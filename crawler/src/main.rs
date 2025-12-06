use anyhow::Result;
use chrono::Utc;
use clap::Parser;
use reqwest::Client;
use scraper::{Html, Selector};
use sha2::{Digest, Sha256};
use std::collections::hash_map::DefaultHasher;
use std::collections::{HashMap, HashSet, VecDeque};
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::{Semaphore, mpsc};
use tracing::{debug, error, info, warn};
use url::Url;

use arrow::array::{ArrayRef, BinaryArray, Int64Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use parquet::arrow::ArrowWriter;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;

const MAX_CONCURRENT_REQUESTS: usize = 10;
const MAX_DEPTH: usize = 3;
const USER_AGENT: &str = "FlooglyFlup/0.1.0 (Simple Crawler)";

fn normalize_url(url_str: &str) -> Result<String> {
    let mut url = Url::parse(url_str)?;
    url.set_fragment(None);
    let mut normalized = url.to_string();
    if normalized.ends_with('/') && normalized.matches('/').count() > 2 {
        normalized.pop();
    }
    Ok(normalized)
}

fn ensure_scheme(raw: &str) -> String {
    if raw.starts_with("http://") || raw.starts_with("https://") {
        raw.to_string()
    } else {
        format!("https://{}", raw)
    }
}

fn resolve_seed_urls(raw_entries: Vec<String>) -> Vec<String> {
    let mut seeds = Vec::new();

    for entry in raw_entries {
        let entry = entry.trim();
        if entry.is_empty() {
            continue;
        }

        let path = PathBuf::from(entry);
        if path.is_file() {
            info!("Reading seed URLs from file: {:?}", path);
            match std::fs::read_to_string(&path) {
                Ok(contents) => {
                    let mut count = 0;
                    for line in contents.lines() {
                        let line = line.split('#').next().unwrap_or("").trim();
                        if line.is_empty() {
                            continue;
                        }
                        seeds.push(ensure_scheme(line));
                        count += 1;
                    }
                    info!("Loaded {} seed URL(s) from {:?}", count, path);
                }
                Err(e) => error!("Failed to read seed file {:?}: {}", path, e),
            }
        } else {
            seeds.push(ensure_scheme(entry));
        }
    }

    seeds
}

#[derive(Debug, Clone)]
struct PageRecord {
    url: String,
    content_type: String,
    retrieval_date: String,
    crawl_time_ms: i64,
    content_hash: String,
    body: Vec<u8>,
    outbound_links: Vec<String>,
}

fn domain_hash(domain: &str) -> u64 {
    let mut hasher = DefaultHasher::new();
    domain.hash(&mut hasher);
    hasher.finish()
}

fn parquet_path_for_domain(output_dir: &Path, domain: &str) -> PathBuf {
    output_dir.join(format!("{}_{}.parquet", domain_hash(domain), domain))
}

fn parquet_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("url", DataType::Utf8, false),
        Field::new("content_type", DataType::Utf8, false),
        Field::new("retrieval_date", DataType::Utf8, false),
        Field::new("crawl_time_ms", DataType::Int64, false),
        Field::new("content_hash", DataType::Utf8, false),
        Field::new("body", DataType::Binary, false),
        Field::new("outbound_links", DataType::Utf8, false),
    ]))
}

fn records_to_batch(schema: &Arc<Schema>, records: &[PageRecord]) -> Result<RecordBatch> {
    let urls: ArrayRef = Arc::new(StringArray::from(
        records.iter().map(|r| r.url.as_str()).collect::<Vec<_>>(),
    ));
    let content_types: ArrayRef = Arc::new(StringArray::from(
        records
            .iter()
            .map(|r| r.content_type.as_str())
            .collect::<Vec<_>>(),
    ));
    let retrieval_dates: ArrayRef = Arc::new(StringArray::from(
        records
            .iter()
            .map(|r| r.retrieval_date.as_str())
            .collect::<Vec<_>>(),
    ));
    let crawl_times: ArrayRef = Arc::new(Int64Array::from(
        records.iter().map(|r| r.crawl_time_ms).collect::<Vec<_>>(),
    ));
    let content_hashes: ArrayRef = Arc::new(StringArray::from(
        records
            .iter()
            .map(|r| r.content_hash.as_str())
            .collect::<Vec<_>>(),
    ));
    let bodies: ArrayRef = Arc::new(BinaryArray::from(
        records
            .iter()
            .map(|r| r.body.as_slice())
            .collect::<Vec<_>>(),
    ));
    let outbound_links: ArrayRef = Arc::new(StringArray::from(
        records
            .iter()
            .map(|r| r.outbound_links.join("\n"))
            .collect::<Vec<_>>(),
    ));

    Ok(RecordBatch::try_new(
        schema.clone(),
        vec![
            urls,
            content_types,
            retrieval_dates,
            crawl_times,
            content_hashes,
            bodies,
            outbound_links,
        ],
    )?)
}

fn read_existing_records(path: &Path) -> Result<Vec<PageRecord>> {
    if !path.exists() {
        return Ok(Vec::new());
    }

    let file = std::fs::File::open(path)?;
    let reader = ParquetRecordBatchReaderBuilder::try_new(file)?.build()?;
    let mut records = Vec::new();

    for batch in reader {
        let batch = batch?;
        let urls = batch
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let content_types = batch
            .column(1)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let retrieval_dates = batch
            .column(2)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let crawl_times = batch
            .column(3)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        let content_hashes = batch
            .column(4)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let bodies = batch
            .column(5)
            .as_any()
            .downcast_ref::<BinaryArray>()
            .unwrap();

        let outbound_links = batch
            .schema()
            .index_of("outbound_links")
            .ok()
            .map(|idx| {
                batch
                    .column(idx)
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .unwrap()
                    .clone()
            });

        for i in 0..batch.num_rows() {
            records.push(PageRecord {
                url: urls.value(i).to_string(),
                content_type: content_types.value(i).to_string(),
                retrieval_date: retrieval_dates.value(i).to_string(),
                crawl_time_ms: crawl_times.value(i),
                content_hash: content_hashes.value(i).to_string(),
                body: bodies.value(i).to_vec(),
                outbound_links: outbound_links
                    .as_ref()
                    .map(|col| {
                        col.value(i)
                            .lines()
                            .map(|s| s.to_string())
                            .collect::<Vec<_>>()
                    })
                    .unwrap_or_default(),
            });
        }
    }

    Ok(records)
}

fn write_domain_file(output_dir: &Path, domain: &str, new_records: Vec<PageRecord>) -> Result<()> {
    let path = parquet_path_for_domain(output_dir, domain);
    let schema = parquet_schema();

    let mut existing = read_existing_records(&path)?;
    let new_urls: HashSet<&str> = new_records.iter().map(|r| r.url.as_str()).collect();
    existing.retain(|r| !new_urls.contains(r.url.as_str()));
    existing.extend(new_records);

    let props = WriterProperties::builder()
        .set_compression(Compression::ZSTD(Default::default()))
        .build();

    let file = std::fs::File::create(&path)?;
    let mut writer = ArrowWriter::try_new(file, schema.clone(), Some(props))?;
    writer.write(&records_to_batch(&schema, &existing)?)?;
    writer.close()?;

    info!("Wrote {} pages to {:?}", existing.len(), path);
    Ok(())
}

#[derive(Parser, Debug)]
#[command(name = "crawler")]
#[command(about = "Web crawler for FlooglyFlup search engine")]
struct Args {
    #[arg(short, long, default_value = "pages")]
    output_dir: PathBuf,

    #[arg(short, long, value_delimiter = ',')]
    urls: Vec<String>,

    #[arg(short, long, default_value_t = MAX_DEPTH)]
    depth: usize,

    #[arg(short, long, default_value_t = 100)]
    max_pages: usize,
}

struct Crawler {
    client: Client,
    semaphore: Arc<Semaphore>,
    max_depth: usize,
    max_pages: usize,
    page_sender: mpsc::UnboundedSender<(String, PageRecord)>,
}

impl Crawler {
    fn new(
        page_sender: mpsc::UnboundedSender<(String, PageRecord)>,
        max_depth: usize,
        max_pages: usize,
    ) -> Result<Self> {
        let client = Client::builder()
            .user_agent(USER_AGENT)
            .timeout(std::time::Duration::from_secs(10))
            .build()?;

        Ok(Self {
            client,
            semaphore: Arc::new(Semaphore::new(MAX_CONCURRENT_REQUESTS)),
            max_depth,
            max_pages,
            page_sender,
        })
    }

    async fn crawl(&self, start_urls: Vec<String>) -> Result<()> {
        let mut queue: VecDeque<(String, usize)> = VecDeque::new();
        let mut queued_or_visited: HashSet<String> = HashSet::new();
        let mut visited_count = 0usize;

        for url in start_urls {
            match normalize_url(&url) {
                Ok(normalized) => {
                    if queued_or_visited.insert(normalized.clone()) {
                        queue.push_back((normalized, 0));
                    }
                }
                Err(e) => warn!("Skipping invalid seed URL {:?}: {}", url, e),
            }
        }

        info!("Queue initialized with {} seed URL(s)", queue.len());
        if queue.is_empty() {
            warn!("Queue is empty, nothing to crawl");
        }

        while let Some((url, depth)) = queue.pop_front() {
            if visited_count >= self.max_pages {
                info!("Reached maximum page limit of {}", self.max_pages);
                break;
            }

            if depth > self.max_depth {
                info!(
                    "Skipping {} (depth {} exceeds max {})",
                    url, depth, self.max_depth
                );
                continue;
            }

            visited_count += 1;

            debug!(
                "Queue size: {}, visited: {}/{}",
                queue.len(),
                visited_count,
                self.max_pages
            );

            info!(
                "Crawling: {} (depth: {}, queue: {}, visited: {}/{})",
                url,
                depth,
                queue.len(),
                visited_count,
                self.max_pages
            );

            let _permit = self.semaphore.acquire().await?;

            match self.crawl_page(&url).await {
                Ok(links) => {
                    if depth < self.max_depth {
                        let mut added = 0;
                        for link in &links {
                            if queued_or_visited.insert(link.clone()) {
                                debug!("Queuing link: {} (depth {})", link, depth + 1);
                                queue.push_back((link.clone(), depth + 1));
                                added += 1;
                            }
                        }
                        info!(
                            "Found {} link(s) on {} ({} queued for depth {})",
                            links.len(),
                            url,
                            added,
                            depth + 1
                        );
                    } else {
                        info!(
                            "Found {} link(s) on {}, but max depth {} reached, not following",
                            links.len(),
                            url,
                            self.max_depth
                        );
                    }
                }
                Err(e) => {
                    error!("Error crawling {}: {}", url, e);
                }
            }
        }

        info!("Crawl complete!");
        Ok(())
    }

    async fn crawl_page(&self, url: &str) -> Result<Vec<String>> {
        let normalized_url = normalize_url(url).unwrap_or_else(|_| url.to_string());

        info!("Fetching: {}", normalized_url);
        let fetch_start = std::time::Instant::now();
        let response = self.client.get(&normalized_url).send().await?;
        let status = response.status();
        let final_url = response.url().to_string();

        if !status.is_success() {
            warn!("Non-success status {} for {}", status, normalized_url);
            return Ok(Vec::new());
        }

        let content_type = response
            .headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .to_string();

        if !content_type.contains("text/html") {
            debug!("Skipping non-HTML content: {}", normalized_url);
            return Ok(Vec::new());
        }

        let body_bytes = response.bytes().await?;
        let crawl_time_ms = fetch_start.elapsed().as_millis() as i64;

        info!(
            "Fetched: {} -> {} [{}] {} ({} bytes, {}ms)",
            normalized_url,
            final_url,
            status.as_u16(),
            content_type,
            body_bytes.len(),
            crawl_time_ms
        );

        let content_hash = format!("{:x}", Sha256::digest(body_bytes.as_ref()));
        let retrieval_date = Utc::now().to_rfc3339();

        let body_vec = body_bytes.to_vec();
        let html = String::from_utf8_lossy(&body_vec).into_owned();
        let url_for_links = normalized_url.clone();
        let extracted =
            tokio::task::spawn_blocking(move || extract_links(&url_for_links, &html)).await?;

        let base_url = Url::parse(&normalized_url)
            .ok()
            .and_then(|u| u.domain().map(|d| d.to_string()));

        if let Some(domain) = base_url {
            let record = PageRecord {
                url: normalized_url.clone(),
                content_type,
                retrieval_date,
                crawl_time_ms,
                content_hash,
                body: body_vec,
                outbound_links: extracted.all,
            };
            self.page_sender.send((domain, record)).ok();
        }

        Ok(extracted.same_domain)
    }
}

fn is_same_site(link_domain: &str, base_domain: &str) -> bool {
    link_domain == base_domain || link_domain.ends_with(&format!(".{}", base_domain))
}

struct ExtractedLinks {
    same_domain: Vec<String>,
    all: Vec<String>,
}

fn extract_links(page_url: &str, html: &str) -> ExtractedLinks {
    let document = Html::parse_document(html);
    let link_selector = Selector::parse("a[href]").unwrap();

    let parsed_base_url = match Url::parse(page_url) {
        Ok(u) => u,
        Err(_) => {
            return ExtractedLinks {
                same_domain: Vec::new(),
                all: Vec::new(),
            };
        }
    };
    let base_domain = parsed_base_url.domain();

    let mut same_domain = Vec::new();
    let mut all = Vec::new();
    for element in document.select(&link_selector) {
        if let Some(href) = element.value().attr("href")
            && let Ok(absolute_url) = parsed_base_url.join(href)
            && let Ok(normalized_link) = normalize_url(absolute_url.as_str())
            && let Ok(link_url) = Url::parse(&normalized_link)
            && (link_url.scheme() == "http" || link_url.scheme() == "https")
        {
            all.push(normalized_link.clone());
            if base_domain.is_some_and(|base| link_url.domain().is_some_and(|d| is_same_site(d, base)))
            {
                same_domain.push(normalized_link);
            }
        }
    }

    ExtractedLinks { same_domain, all }
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_target(false)
        .with_level(true)
        .init();

    let args = Args::parse();

    if args.urls.is_empty() {
        error!("No URLs provided. Use --urls to specify starting URLs.");
        std::process::exit(1);
    }

    info!("Starting crawler with {} URL argument(s)", args.urls.len());
    info!("Output dir: {:?}", args.output_dir);
    info!("Max depth: {}, Max pages: {}", args.depth, args.max_pages);

    let seed_urls = resolve_seed_urls(args.urls);
    if seed_urls.is_empty() {
        error!("No seed URLs resolved. Check --urls entries and any referenced files.");
        std::process::exit(1);
    }

    std::fs::create_dir_all(&args.output_dir)?;

    let (page_sender, mut page_receiver) = mpsc::unbounded_channel::<(String, PageRecord)>();

    let collector_task = tokio::spawn(async move {
        let mut by_domain: HashMap<String, Vec<PageRecord>> = HashMap::new();
        while let Some((domain, record)) = page_receiver.recv().await {
            by_domain.entry(domain).or_default().push(record);
        }
        by_domain
    });

    let crawler = Crawler::new(page_sender, args.depth, args.max_pages)?;
    crawler.crawl(seed_urls).await?;
    drop(crawler);

    let by_domain = collector_task.await?;

    info!("Writing {} domain(s) to parquet...", by_domain.len());
    for (domain, records) in by_domain {
        if let Err(e) = write_domain_file(&args.output_dir, &domain, records) {
            error!("Failed to write parquet for {}: {}", domain, e);
        }
    }

    info!("Crawling complete!");
    Ok(())
}
