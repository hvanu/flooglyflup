use axum::{
    Json, Router,
    extract::{Query, State},
    http::StatusCode,
    response::{Html, IntoResponse},
    routing::get,
};
use clap::Parser;
use serde::{Deserialize, Serialize};
use std::{path::PathBuf, sync::Arc};
use tantivy::{
    DocId, Index, ReloadPolicy, Score, SegmentReader, collector::TopDocs,
    query::QueryParser,
    schema::*, snippet::SnippetGenerator,
};
use tower_http::cors::CorsLayer;
use tracing::{error, info};

#[derive(Parser, Debug)]
#[command(name = "server")]
#[command(about = "Web server for FlooglyFlup search engine")]
struct Args {
    #[arg(short, long, default_value = "index")]
    index_path: PathBuf,

    #[arg(short, long, default_value_t = 3000)]
    port: u16,

    #[arg(long, default_value = "127.0.0.1")]
    host: String,
}

#[derive(Clone)]
struct AppState {
    searcher: Arc<tantivy::Searcher>,
    query_parser: Arc<QueryParser>,
    schema: Schema,
    suggestions: Arc<SuggestionIndex>,
}

struct SuggestionIndex {
    // Sorted by `word` for binary-search prefix lookup.
    entries: Vec<SuggestionEntry>,
}

#[derive(Debug, Deserialize)]
struct SuggestionArtifact {
    entries: Vec<SuggestionEntry>,
}

#[derive(Debug, Deserialize)]
struct SuggestionEntry {
    word: String,
    count: u32,
}

impl SuggestionIndex {
    fn load(index_path: &std::path::Path) -> anyhow::Result<Self> {
        let suggestions_path = index_path.join("suggestions.postcard");
        let bytes = std::fs::read(&suggestions_path)?;
        let artifact: SuggestionArtifact = postcard::from_bytes(&bytes)?;

        Ok(Self {
            entries: artifact.entries,
        })
    }

    fn suggest(&self, prefix: &str, limit: usize) -> Vec<String> {
        if limit == 0 {
            return Vec::new();
        }

        let prefix_lower = prefix.to_lowercase();
        let start = self
            .entries
            .partition_point(|entry| entry.word.as_str() < prefix_lower.as_str());

        let mut top: Vec<&SuggestionEntry> = Vec::new();
        for entry in self.entries[start..]
            .iter()
            .take_while(|entry| entry.word.starts_with(&prefix_lower))
        {
            let mut insert_at = None;
            for (i, existing) in top.iter().enumerate() {
                if entry.count > existing.count
                    || (entry.count == existing.count
                        && entry.word.as_str() < existing.word.as_str())
                {
                    insert_at = Some(i);
                    break;
                }
            }

            match insert_at {
                Some(i) => {
                    top.insert(i, entry);
                    if top.len() > limit {
                        top.pop();
                    }
                }
                None if top.len() < limit => top.push(entry),
                _ => {}
            }
        }

        top.into_iter().map(|entry| entry.word.clone()).collect()
    }
}

#[derive(Debug, Deserialize)]
struct SearchQuery {
    q: String,
    #[serde(default = "default_limit")]
    limit: usize,
    #[serde(default)]
    lang: Option<String>,
    #[serde(default)]
    date_from: Option<String>,
    #[serde(default)]
    date_to: Option<String>,
    #[serde(default)]
    code_only: bool,
    #[serde(default)]
    min_quality: Option<f64>,
}

fn default_limit() -> usize {
    10
}

#[derive(Debug, Serialize)]
struct SearchResult {
    url: String,
    title: String,
    snippet: String,
    score: f32,
    base_url: String,
    language: String,
    published_date: Option<String>,
    authority_score: f64,
    domain_quality: f64,
    content_quality: f64,
    has_code: bool,
    word_count: usize,
    combined_score: f64,
}

#[derive(Debug, Serialize)]
struct SearchResponse {
    query: String,
    results: Vec<SearchResult>,
    total: usize,
    time_ms: u128,
}

#[derive(Debug, Serialize)]
struct ErrorResponse {
    error: String,
}

#[derive(Debug, Deserialize)]
struct SuggestQuery {
    q: String,
    #[serde(default = "default_suggest_limit")]
    limit: usize,
}

fn default_suggest_limit() -> usize {
    10
}

#[derive(Debug, Serialize)]
struct SuggestResponse {
    query: String,
    suggestions: Vec<String>,
}

// Parse advanced search operators from query
// Supports: site:domain, lang:code, has:code. TODO: date:YYYY-MM-DD,
fn parse_search_operators(query: &str) -> (String, Option<String>, Option<String>, bool) {
    let mut cleaned_query = query.to_string();
    let mut site_filter = None;
    let mut lang_filter = None;
    let mut has_code = false;

    if let Some(site_start) = query.find("site:") {
        let rest = &query[site_start + 5..];
        let site_end = rest.find(' ').unwrap_or(rest.len());
        site_filter = Some(rest[..site_end].to_string());
        cleaned_query = cleaned_query.replace(&format!("site:{}", &rest[..site_end]), "");
    }

    if let Some(lang_start) = query.find("lang:") {
        let rest = &query[lang_start + 5..];
        let lang_end = rest.find(' ').unwrap_or(rest.len());
        lang_filter = Some(rest[..lang_end].to_string());
        cleaned_query = cleaned_query.replace(&format!("lang:{}", &rest[..lang_end]), "");
    }

    if query.contains("has:code") {
        has_code = true;
        cleaned_query = cleaned_query.replace("has:code", "");
    }

    (
        cleaned_query.trim().to_string(),
        site_filter,
        lang_filter,
        has_code,
    )
}

async fn search_handler(
    State(state): State<AppState>,
    Query(params): Query<SearchQuery>,
) -> Result<Json<SearchResponse>, (StatusCode, Json<ErrorResponse>)> {
    let start = std::time::Instant::now();

    if params.q.trim().is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "Query cannot be empty".to_string(),
            }),
        ));
    }

    let (cleaned_query, site_filter, lang_filter_from_query, has_code_filter) =
        parse_search_operators(&params.q);

    let lang_filter = params.lang.or(lang_filter_from_query);
    let code_filter = params.code_only || has_code_filter;

    let (query, parse_errors) = state.query_parser.parse_query_lenient(&cleaned_query);
    if !parse_errors.is_empty() {
        info!(
            "Lenient query parse dropped {} part(s) of {:?}: {:?}",
            parse_errors.len(),
            cleaned_query,
            parse_errors
        );
    }

    let limit = params.limit.min(100);

    let authority_score_field = state.schema.get_field("authority_score").ok();

    // Oversample beyond `limit` because site/lang/date/has:code filters below are
    // applied post-search and can drop some of the top-ranked hits.
    let search_limit = (limit * 3).min(300);
    let min_quality = params.min_quality.unwrap_or(0.0);

    // Rank directly in Tantivy: tweak_score lets us fold FlupRank authority
    // and domain/content quality into the score used to
    // pick the top-K. The sort key is (combined_score, original_score) so ties on
    // the combined score fall back to text relevance.
    let top_docs_by_custom_score =
        TopDocs::with_limit(search_limit).tweak_score(move |segment_reader: &SegmentReader| {
            let fast_fields = segment_reader.fast_fields();
            let authority_reader = fast_fields
                .f64("authority_score")
                .unwrap()
                .first_or_default_col(0.0);
            let domain_reader = fast_fields
                .f64("domain_quality")
                .unwrap()
                .first_or_default_col(1.0);
            let content_reader = fast_fields
                .f64("content_quality")
                .unwrap()
                .first_or_default_col(1.0);

            move |doc: DocId, original_score: Score| {
                let authority: f64 = authority_reader.get_val(doc);
                let domain: f64 = domain_reader.get_val(doc);
                let content: f64 = content_reader.get_val(doc);

                // 50% text relevance + 20% authority (FlupRank) + 20% domain quality + 10% content quality
                let mut combined_score = (original_score as f64 * 0.5)
                    + (authority * 0.20)
                    + (domain * 0.20)
                    + (content * 0.10);

                // Penalize (but don't eliminate) docs below the minimum quality threshold
                if domain < min_quality || content < min_quality {
                    combined_score *= 0.5;
                }

                (combined_score, original_score)
            }
        });

    let top_docs = state
        .searcher
        .search(&query, &top_docs_by_custom_score)
        .map_err(|e| {
            error!("Search error: {}", e);
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: "Search failed".to_string(),
                }),
            )
        })?;

    let url_field = state.schema.get_field("url").unwrap();
    let title_field = state.schema.get_field("title").unwrap();
    let content_field = state.schema.get_field("content").unwrap();
    let base_url_field = state.schema.get_field("base_url").ok();

    let snippet_generator = SnippetGenerator::create(&state.searcher, &*query, content_field)
        .map(|mut generator| {
            generator.set_max_num_chars(200);
            generator
        })
        .map_err(|e| error!("Failed to build snippet generator: {}", e))
        .ok();
    let language_field = state.schema.get_field("language").ok();
    let published_date_field = state.schema.get_field("published_date").ok();
    let domain_quality_field = state.schema.get_field("domain_quality").ok();
    let content_quality_field = state.schema.get_field("content_quality").ok();
    let has_code_field = state.schema.get_field("has_code").ok();
    let word_count_field = state.schema.get_field("word_count").ok();

    let mut results = Vec::new();
    for ((combined_score, score), doc_address) in top_docs {
        let retrieved_doc: tantivy::TantivyDocument =
            state.searcher.doc(doc_address).map_err(|e| {
                error!("Doc retrieval error: {}", e);
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(ErrorResponse {
                        error: "Failed to retrieve document".to_string(),
                    }),
                )
            })?;

        let url = retrieved_doc
            .get_first(url_field)
            .and_then(|f| f.as_str())
            .unwrap_or("")
            .to_string();

        let title = retrieved_doc
            .get_first(title_field)
            .and_then(|f| f.as_str())
            .unwrap_or("Untitled")
            .to_string();

        let content = retrieved_doc
            .get_first(content_field)
            .and_then(|f| f.as_str())
            .unwrap_or("")
            .to_string();

        let base_url = base_url_field
            .and_then(|field| retrieved_doc.get_first(field))
            .and_then(|f| f.as_str())
            .unwrap_or("")
            .to_string();

        let language = language_field
            .and_then(|field| retrieved_doc.get_first(field))
            .and_then(|f| f.as_str())
            .unwrap_or("unknown")
            .to_string();

        let published_date = published_date_field
            .and_then(|field| retrieved_doc.get_first(field))
            .and_then(|f| f.as_str())
            .filter(|s| !s.is_empty())
            .map(|s| s.to_string());

        let authority_score = authority_score_field
            .and_then(|field| retrieved_doc.get_first(field))
            .and_then(|f| f.as_f64())
            .unwrap_or(0.0);

        let domain_quality = domain_quality_field
            .and_then(|field| retrieved_doc.get_first(field))
            .and_then(|f| f.as_f64())
            .unwrap_or(1.0);

        let content_quality = content_quality_field
            .and_then(|field| retrieved_doc.get_first(field))
            .and_then(|f| f.as_f64())
            .unwrap_or(1.0);

        let has_code = has_code_field
            .and_then(|field| retrieved_doc.get_first(field))
            .and_then(|f| f.as_bool())
            .unwrap_or(false);

        let word_count = word_count_field
            .and_then(|field| retrieved_doc.get_first(field))
            .and_then(|f| f.as_u64())
            .unwrap_or(0) as usize;

        // Apply filters
        if let Some(ref site) = site_filter
            && !base_url.contains(site)
        {
            continue;
        }

        if let Some(ref lang) = lang_filter
            && language != *lang
        {
            continue;
        }

        if code_filter && !has_code {
            continue;
        }

        if let Some(ref date_from) = params.date_from
            && let Some(ref pub_date) = published_date
            && pub_date < date_from
        {
            continue;
        }

        if let Some(ref date_to) = params.date_to
            && let Some(ref pub_date) = published_date
            && pub_date > date_to
        {
            continue;
        }

        let snippet = snippet_generator
            .as_ref()
            .map(|generator| generator.snippet_from_doc(&retrieved_doc))
            .filter(|snippet| !snippet.fragment().is_empty())
            .map(|snippet| snippet.to_html())
            .unwrap_or_else(|| {
                if content.len() > 200 {
                    format!("{}...", &content[..200])
                } else {
                    content
                }
            });

        results.push(SearchResult {
            url,
            title,
            snippet,
            score,
            base_url,
            language,
            published_date,
            authority_score,
            domain_quality,
            content_quality,
            has_code,
            word_count,
            combined_score,
        });
    }

    results.truncate(limit);

    let time_ms = start.elapsed().as_millis();

    Ok(Json(SearchResponse {
        query: params.q,
        total: results.len(),
        results,
        time_ms,
    }))
}

async fn suggest_handler(
    State(state): State<AppState>,
    Query(params): Query<SuggestQuery>,
) -> Json<SuggestResponse> {
    let prefix = params.q.trim();
    let limit = params.limit.clamp(1, 20);

    let suggestions = if prefix.is_empty() {
        Vec::new()
    } else {
        state.suggestions.suggest(prefix, limit)
    };

    Json(SuggestResponse {
        query: params.q,
        suggestions,
    })
}

async fn index_handler() -> Html<&'static str> {
    Html(include_str!("../static/index.html"))
}

async fn style_handler() -> impl IntoResponse {
    (
        [("content-type", "text/css")],
        include_str!("../static/assets/style.css"),
    )
}

async fn health_handler() -> impl IntoResponse {
    Json(serde_json::json!({
        "status": "healthy"
    }))
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_target(false)
        .with_level(true)
        .init();

    let args = Args::parse();

    info!("Opening index at {:?}", args.index_path);

    let index = Index::open_in_dir(&args.index_path)?;
    let schema = index.schema();

    let title_field = schema.get_field("title")?;
    let content_field = schema.get_field("content")?;

    let query_parser = QueryParser::for_index(&index, vec![title_field, content_field]);

    let reader = index
        .reader_builder()
        .reload_policy(ReloadPolicy::OnCommitWithDelay)
        .try_into()?;

    let searcher = reader.searcher();

    info!("Loading suggestion index artifact...");
    let suggestions = SuggestionIndex::load(&args.index_path)?;
    info!(
        "Suggestion index loaded with {} entries",
        suggestions.entries.len()
    );

    let state = AppState {
        searcher: Arc::new(searcher),
        query_parser: Arc::new(query_parser),
        schema,
        suggestions: Arc::new(suggestions),
    };

    let app = Router::new()
        .route("/", get(index_handler))
        .route("/style.css", get(style_handler))
        .route("/api/search", get(search_handler))
        .route("/api/suggest", get(suggest_handler))
        .route("/health", get(health_handler))
        .layer(CorsLayer::permissive())
        .with_state(state);

    let addr = format!("{}:{}", args.host, args.port);
    info!("Starting server on http://{}", addr);

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
