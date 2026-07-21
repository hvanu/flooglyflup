# flooglyflup

A tiny search engine, built to research how all components of the full pipeline 
of a search engine can be made as simple as possible. 

There are just three components, a crawler, a processor, and a web server. The server is a single ~3MB binary. 
Currently it is not a distributed architecture. Should scale to a significant number of pages, but it's not intended to 
index the whole internet. Search is built using Tantivy, with a custom tunable ranking using a combination of text relevance and simplified PageRank, domain quality and content quality. Allows for filtering on language (using `whatlang`) and date. 

## the pipeline

Three crates, one workspace, run in order:

```
crawler    -> pages/*.parquet   (fetch HTML, save to disk (zstd compressed), follow links)
processor  -> index/*           (parse and score pages, build the Tantivy index)
server     -> :3000/api/search  (serve queries over HTTP)
```


## crawler

```
cargo run --bin crawler -- \
    --urls seed-domains.txt \
    --output-dir pages \
    --depth 3 \
    --max-pages 100
```

`--urls` takes a comma-separated mix of literal URLs/domains and paths to
files with one URL/domain per line (`seed-domains.txt`)

The crawler does breadth-first, link following, capped by
`--depth` and `--max-pages`, with a limit on concurrent requests so
we don't hammer anyone's server and just fetches and stores. 
Non-HTML responses are ignored.

Output is one parquet file per root domain in `--output-dir`, named
`{hash(domain)}_{domain}.parquet`, columns `url`, `content_type`,
`retrieval_date`, `crawl_time_ms`, `content_hash`, and `body` (raw HTML,
zstd-compressed by the parquet writer). Re-running against the same
`--output-dir` merges into the existing per-domain file; rows with a
matching URL get overwritten, everything else is additive. 

## processor

```
cargo run --bin processor -- \
    --input-dir pages \
    --index-path index \
    --fluprank-iterations 20
```

The processor reads every parquet file in
`--input-dir`, parses the HTML, and computes a handful of signals per page:

- **language** - via `whatlang`
- **published date** - optional, from meta properties like
  `<meta property="article:published_time">`
- **domain quality** - a lookup against a list of known high and low quality domains.
- **content quality** - Some simplified metric for content quality of the specific page. Todo: AI slop detection using a tiny ml model.
- **FlupRank** - a simplified PageRank knockoff; iterates over the link graph seeded with domain quality


Everything gets written into a single Tantivy index at `--index-path`

## server

```
cargo run --bin server -- \
    --index-path index \
    --port 3000 \
    --host 127.0.0.1
```

A thin `axum` server that exposes `GET /api/search`:

```
curl 'localhost:3000/api/search?q=rust+async&limit=10&lang=en'
```

`q` is the only required param. `limit`, `lang`, `date_from`/`date_to`,
`code_only`, and `min_quality` are all optional filters layered on top of
Tantivy's relevance score, combined into a `combined_score` in each
result. 

