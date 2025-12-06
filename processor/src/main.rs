use anyhow::Result;
use clap::Parser;
use scraper::{Html, Selector};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::{Arc, OnceLock};
use tantivy::schema::*;
use tantivy::{Index, doc};
use tracing::{info, warn};
use url::Url;
use whatlang::{Lang, detect};

use arrow::array::{BinaryArray, StringArray};
use arrow::datatypes::Schema as ArrowSchema;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

fn detect_language(text: &str) -> String {
    if text.len() < 20 {
        return "unknown".to_string();
    }

    match detect(text) {
        Some(info) => match info.lang() {
            Lang::Eng => "en",
            Lang::Spa => "es",
            Lang::Fra => "fr",
            Lang::Deu => "de",
            Lang::Por => "pt",
            Lang::Ita => "it",
            Lang::Rus => "ru",
            Lang::Jpn => "ja",
            Lang::Kor => "ko",
            Lang::Cmn => "zh",
            Lang::Ara => "ar",
            Lang::Hin => "hi",
            Lang::Nld => "nl",
            Lang::Swe => "sv",
            Lang::Pol => "pl",
            _ => "other",
        }
        .to_string(),
        None => "unknown".to_string(),
    }
}

fn normalize_date_str(raw: &str) -> Option<String> {
    let raw = raw.trim();
    if raw.is_empty() {
        return None;
    }

    if let Ok(dt) = chrono::DateTime::parse_from_rfc3339(raw) {
        return Some(dt.format("%Y-%m-%d").to_string());
    }

    if let Ok(date) = chrono::NaiveDate::parse_from_str(raw, "%Y-%m-%d") {
        return Some(date.format("%Y-%m-%d").to_string());
    }

    const FORMATS: &[&str] = &[
        "%Y-%m-%dT%H:%M:%S%.f",
        "%Y/%m/%d",
        "%B %d, %Y",
        "%d %B %Y",
        "%b %d, %Y",
    ];
    for fmt in FORMATS {
        if let Ok(date) = chrono::NaiveDate::parse_from_str(raw, fmt) {
            return Some(date.format("%Y-%m-%d").to_string());
        }
        if let Ok(dt) = chrono::NaiveDateTime::parse_from_str(raw, fmt) {
            return Some(dt.format("%Y-%m-%d").to_string());
        }
    }

    None
}

fn extract_publish_date(_html: &str, document: &Html) -> Option<String> {
    let meta_selectors = [
        "meta[property='article:published_time']",
        "meta[property='og:published_time']",
        "meta[name='publish_date']",
        "meta[name='date']",
        "meta[name='DC.date']",
        "time[datetime]",
    ];

    for selector_str in &meta_selectors {
        if let Ok(selector) = Selector::parse(selector_str) {
            for element in document.select(&selector) {
                if let Some(content) = element
                    .value()
                    .attr("content")
                    .or_else(|| element.value().attr("datetime"))
                {
                    if let Some(normalized) = normalize_date_str(content) {
                        return Some(normalized);
                    }
                }
            }
        }
    }

    // No reasonably certain date found; leave it unset
    None
}

const HIGH_QUALITY_RAW: &str = include_str!("../data/high-quality.txt");
const LOW_QUALITY_RAW: &str = include_str!("../data/low-quality.txt");

const HIGH_QUALITY_SCORE: f64 = 1.5;
const LOW_QUALITY_SCORE: f64 = 0.5;

// Parse a newline-delimited domain pattern list, ignoring blank lines and `#` comments.
fn parse_domain_list(raw: &str) -> Vec<String> {
    raw.lines()
        .map(|line| line.split('#').next().unwrap_or("").trim())
        .filter(|line| !line.is_empty())
        .map(|line| line.to_lowercase())
        .collect()
}

fn high_quality_domains() -> &'static [String] {
    static LIST: OnceLock<Vec<String>> = OnceLock::new();
    LIST.get_or_init(|| parse_domain_list(HIGH_QUALITY_RAW))
}

fn low_quality_domains() -> &'static [String] {
    static LIST: OnceLock<Vec<String>> = OnceLock::new();
    LIST.get_or_init(|| parse_domain_list(LOW_QUALITY_RAW))
}

fn score_domain_quality(domain: &str) -> f64 {
    let domain_lower = domain.to_lowercase();
    let matches_any =
        |patterns: &[String]| patterns.iter().any(|p| domain_lower.contains(p.as_str()));

    if matches_any(high_quality_domains()) {
        HIGH_QUALITY_SCORE
    } else if matches_any(low_quality_domains()) {
        LOW_QUALITY_SCORE
    } else {
        1.0
    }
}

// Analyze content quality based on technical indicators; for illustration purposes.
// TODO: Add a small ML model for AI-slop detection/quality scoring
fn analyze_content_quality(html: &str, text_content: &str) -> (f64, bool) {
    let mut quality_score = 1.0;
    let mut has_code = false;

    let code_indicators = ["<code", "<pre", "```", "class=\"highlight", "class=\"code"];
    let code_count = code_indicators
        .iter()
        .filter(|&indicator| html.contains(indicator))
        .count();

    if code_count > 0 {
        has_code = true;
        quality_score += 0.3;
    }

    let technical_terms = [
        "algorithm",
        "implementation",
        "theorem",
        "proof",
        "analysis",
        "research",
        "experiment",
        "methodology",
        "hypothesis",
        "abstract",
        "function",
        "class",
        "interface",
        "api",
        "protocol",
        "arxiv",
        "doi:",
        "isbn:",
        "citation",
        "references",
    ];

    let text_lower = text_content.to_lowercase();
    let tech_term_count = technical_terms
        .iter()
        .filter(|&term| text_lower.contains(term))
        .count();

    quality_score += (tech_term_count as f64 * 0.05).min(0.5);

    let word_count = text_content.split_whitespace().count();
    if word_count < 100 {
        quality_score *= 0.5;
    } else if word_count > 1000 {
        quality_score += 0.2;
    }

    (quality_score.min(3.0), has_code)
}

// Compute FlupRank scores (Filtered Link-based User-first Priority Rank ? :) )
// A simplified TrustRank-inspired scoring algorithm
fn compute_fluprank_scores(
    link_graph: &HashMap<String, Vec<String>>,
    inbound_links: &HashMap<String, Vec<String>>,
    iterations: usize,
) -> HashMap<String, f64> {
    let trusted_seeds: HashSet<String> = link_graph
        .keys()
        .filter(|url| {
            if let Ok(parsed) = Url::parse(url)
                && let Some(domain) = parsed.domain()
            {
                return score_domain_quality(domain) >= 1.5;
            }
            false
        })
        .cloned()
        .collect();

    info!("Identified {} trusted seed pages", trusted_seeds.len());

    let mut rank: HashMap<String, f64> = HashMap::new();
    let damping_factor = 0.85;
    let n = link_graph.len() as f64;

    for url in link_graph.keys() {
        rank.insert(
            url.clone(),
            if trusted_seeds.contains(url) {
                2.0 / n
            } else {
                1.0 / n
            },
        );
    }

    // PageRank iterations with trust bias
    for iter in 0..iterations {
        let mut new_rank: HashMap<String, f64> = HashMap::new();
        let mut dangling_sum = 0.0;

        for (url, outlinks) in link_graph.iter() {
            if outlinks.is_empty() {
                dangling_sum += rank.get(url).copied().unwrap_or(0.0);
            }
        }

        for url in link_graph.keys() {
            let teleport_rank = if trusted_seeds.contains(url) {
                (1.0 - damping_factor) * 2.0 / n
            } else {
                (1.0 - damping_factor) / n
            };

            let inbound_contribution: f64 = inbound_links
                .get(url)
                .map(|inbound| {
                    inbound
                        .iter()
                        .filter_map(|source| {
                            let source_rank = rank.get(source).copied().unwrap_or(0.0);
                            let outlink_count = link_graph.get(source)?.len();
                            if outlink_count > 0 {
                                Some(source_rank / outlink_count as f64)
                            } else {
                                None
                            }
                        })
                        .sum()
                })
                .unwrap_or(0.0);

            let dangling_contribution = damping_factor * dangling_sum / n;

            new_rank.insert(
                url.clone(),
                teleport_rank + damping_factor * inbound_contribution + dangling_contribution,
            );
        }

        rank = new_rank;

        if iter % 5 == 0 {
            info!("FlupRank iteration {}/{}", iter + 1, iterations);
        }
    }

    let mut filtered_rank = rank;

    // Normalize scores to 0-1 range for consistency
    if let Some(max_rank) = filtered_rank
        .values()
        .cloned()
        .fold(None, |max, x| Some(max.map_or(x, |m: f64| m.max(x))))
        && max_rank > 0.0
    {
        for val in filtered_rank.values_mut() {
            *val /= max_rank;
        }
    }

    info!("Computed FlupRank scores for {} pages", filtered_rank.len());
    filtered_rank
}

#[derive(Debug, Clone)]
struct Document {
    url: String,
    title: String,
    content: String,
    base_url: String,
    language: String,
    published_date: Option<String>,
    authority_score: f64,
    domain_quality: f64,
    content_quality: f64,
    has_code: bool,
    word_count: usize,
}

#[derive(Debug, Serialize)]
struct SuggestionArtifact {
    entries: Vec<SuggestionEntry>,
}

#[derive(Debug, Serialize, Deserialize)]
struct SuggestionEntry {
    word: String,
    count: u32,
}


fn first_title_segment(title: &str) -> &str {
    const SEPARATORS: [&str; 6] = [" | ", " – ", " — ", " :: ", " » ", " - "];

    let cut = SEPARATORS
        .iter()
        .filter_map(|sep| title.find(sep))
        .min();

    match cut {
        Some(idx) => title[..idx].trim(),
        None => title.trim(),
    }
}

fn build_suggestion_entries(documents: &[Document]) -> Vec<SuggestionEntry> {
    let mut counts: HashMap<String, u32> = HashMap::new();

    for doc in documents {
        let cleaned = first_title_segment(&doc.title);
        if cleaned.is_empty() {
            continue;
        }

        let mut seen_words = HashSet::new();
        for word in cleaned.split(|c: char| !c.is_alphanumeric()) {
            if word.is_empty() {
                continue;
            }

            let word_lower = word.to_lowercase();
            if word_lower.len() < 2 {
                continue;
            }

            if seen_words.insert(word_lower.clone()) {
                *counts.entry(word_lower).or_insert(0) += 1;
            }
        }
    }

    let mut entries: Vec<SuggestionEntry> = counts
        .into_iter()
        .map(|(word, count)| SuggestionEntry { word, count })
        .collect();
    entries.sort_by(|a, b| a.word.cmp(&b.word));
    entries
}

struct ParquetRow {
    url: String,
    body: Vec<u8>,
    outbound_links: Vec<String>,
}

fn read_parquet_rows(path: &PathBuf) -> Result<Vec<ParquetRow>> {
    let file = std::fs::File::open(path)?;
    let reader = ParquetRecordBatchReaderBuilder::try_new(file)?.build()?;
    let mut rows = Vec::new();

    for batch in reader {
        let batch = batch?;
        let schema: &Arc<ArrowSchema> = batch.schema_ref();
        let url_idx = schema.index_of("url")?;
        let body_idx = schema.index_of("body")?;
        let outbound_links_idx = schema.index_of("outbound_links").ok();

        let urls = batch
            .column(url_idx)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let bodies = batch
            .column(body_idx)
            .as_any()
            .downcast_ref::<BinaryArray>()
            .unwrap();
        let outbound_links = outbound_links_idx.map(|idx| {
            batch
                .column(idx)
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap()
        });

        for i in 0..batch.num_rows() {
            rows.push(ParquetRow {
                url: urls.value(i).to_string(),
                body: bodies.value(i).to_vec(),
                outbound_links: outbound_links
                    .map(|col| col.value(i).lines().map(|s| s.to_string()).collect())
                    .unwrap_or_default(),
            });
        }
    }

    Ok(rows)
}

fn process_page(url: &str, html: &str) -> Document {
    let document = Html::parse_document(html);

    let title_selector = Selector::parse("title").unwrap();
    let title = document
        .select(&title_selector)
        .next()
        .map(|e| e.text().collect::<String>())
        .unwrap_or_else(|| url.to_string());

    let body_selector = Selector::parse("body").unwrap();
    let content = document
        .select(&body_selector)
        .next()
        .map(|e| {
            e.text()
                .collect::<Vec<_>>()
                .join(" ")
                .split_whitespace()
                .collect::<Vec<_>>()
                .join(" ")
        })
        .unwrap_or_default();

    let language = detect_language(&content);
    let published_date = extract_publish_date(html, &document);

    let base_url = Url::parse(url)
        .ok()
        .and_then(|u| u.domain().map(|d| d.to_string()))
        .unwrap_or_else(|| url.to_string());

    let domain_quality = score_domain_quality(&base_url);
    let (content_quality, has_code) = analyze_content_quality(html, &content);
    let word_count = content.split_whitespace().count();

    Document {
        url: url.to_string(),
        title: title.trim().to_string(),
        content: content.chars().take(10000).collect(),
        base_url,
        language,
        published_date,
        authority_score: 0.0, // Filled in after FlupRank is computed
        domain_quality,
        content_quality,
        has_code,
        word_count,
    }
}

#[derive(Parser, Debug)]
#[command(name = "processor")]
#[command(about = "Scores crawled pages and builds the Tantivy index for FlooglyFlup")]
struct Args {
    // Directory containing the crawler's parquet files
    #[arg(short = 'p', long, default_value = "pages")]
    input_dir: PathBuf,

    // Path to the Tantivy index directory to (re)build
    #[arg(short = 'i', long, default_value = "index")]
    index_path: PathBuf,

    #[arg(long, default_value_t = 20)]
    fluprank_iterations: usize,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_target(false)
        .with_level(true)
        .init();

    let args = Args::parse();

    let parquet_files: Vec<PathBuf> = std::fs::read_dir(&args.input_dir)?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| path.extension().and_then(|e| e.to_str()) == Some("parquet"))
        .collect();

    info!(
        "Found {} parquet file(s) in {:?}",
        parquet_files.len(),
        args.input_dir
    );

    let mut documents: Vec<Document> = Vec::new();
    let mut link_graph: HashMap<String, Vec<String>> = HashMap::new();
    let mut inbound_links: HashMap<String, Vec<String>> = HashMap::new();

    for path in &parquet_files {
        let rows = match read_parquet_rows(path) {
            Ok(rows) => rows,
            Err(e) => {
                warn!("Failed to read {:?}: {}", path, e);
                continue;
            }
        };

        for row in rows {
            let html = String::from_utf8_lossy(&row.body).to_string();
            let doc = process_page(&row.url, &html);

            link_graph.insert(doc.url.clone(), row.outbound_links.clone());
            for link in &row.outbound_links {
                inbound_links
                    .entry(link.clone())
                    .or_default()
                    .push(doc.url.clone());
            }

            documents.push(doc);
        }
    }

    info!("Processed {} document(s)", documents.len());

    info!("Computing FlupRank authority scores...");
    let fluprank_scores =
        compute_fluprank_scores(&link_graph, &inbound_links, args.fluprank_iterations);

    for doc in &mut documents {
        if let Some(score) = fluprank_scores.get(&doc.url) {
            doc.authority_score = *score;
        }
    }

    let mut schema_builder = Schema::builder();
    schema_builder.add_text_field("url", TEXT | STORED);
    schema_builder.add_text_field("title", TEXT | STORED);
    schema_builder.add_text_field("content", TEXT | STORED);
    schema_builder.add_text_field("base_url", STRING | STORED);
    schema_builder.add_text_field("language", STRING | STORED);
    schema_builder.add_text_field("published_date", STRING | STORED);
    schema_builder.add_f64_field("authority_score", STORED | FAST);
    schema_builder.add_f64_field("domain_quality", STORED | FAST);
    schema_builder.add_f64_field("content_quality", STORED | FAST);
    schema_builder.add_bool_field("has_code", STORED | INDEXED | FAST);
    schema_builder.add_u64_field("word_count", STORED | FAST);
    let schema = schema_builder.build();

    std::fs::create_dir_all(&args.index_path)?;
    let index = Index::create_in_dir(&args.index_path, schema.clone())
        .or_else(|_| Index::open_in_dir(&args.index_path))?;

    let url_field = schema.get_field("url")?;
    let title_field = schema.get_field("title")?;
    let content_field = schema.get_field("content")?;
    let base_url_field = schema.get_field("base_url")?;
    let language_field = schema.get_field("language")?;
    let published_date_field = schema.get_field("published_date")?;
    let authority_score_field = schema.get_field("authority_score")?;
    let domain_quality_field = schema.get_field("domain_quality")?;
    let content_quality_field = schema.get_field("content_quality")?;
    let has_code_field = schema.get_field("has_code")?;
    let word_count_field = schema.get_field("word_count")?;

    let mut index_writer = index.writer(50_000_000)?;
    index_writer.delete_all_documents()?;

    for doc in &documents {
        let tantivy_doc = doc!(
            url_field => doc.url.as_str(),
            title_field => doc.title.as_str(),
            content_field => doc.content.as_str(),
            base_url_field => doc.base_url.as_str(),
            language_field => doc.language.as_str(),
            published_date_field => doc.published_date.as_deref().unwrap_or(""),
            authority_score_field => doc.authority_score,
            domain_quality_field => doc.domain_quality,
            content_quality_field => doc.content_quality,
            has_code_field => doc.has_code,
            word_count_field => doc.word_count as u64,
        );
        index_writer.add_document(tantivy_doc)?;
    }

    info!("Committing index with {} documents...", documents.len());
    index_writer.commit()?;

    let suggestions = SuggestionArtifact {
        entries: build_suggestion_entries(&documents),
    };
    let suggestions_path = args.index_path.join("suggestions.postcard");
    std::fs::write(&suggestions_path, postcard::to_allocvec(&suggestions)?)?;
    info!(
        "Wrote suggestion artifact with {} entries to {:?}",
        suggestions.entries.len(),
        suggestions_path
    );

    info!("Processing complete!");

    Ok(())
}
