# 1. Data collection (`src/build_author_dataset.py`)

This script scrapes isiZulu news articles from [Isolezwe](https://isolezwe.co.za) author profile pages and builds the raw authorship dataset used by every downstream step.

## Purpose

Existing public isiZulu corpora either lack enough text per author or do not label articles by writer. This scraper collects labelled articles from a single news outlet so that each sample has a known author, the core requirement for authorship attribution.

## Target authors

Eleven journalists are included. Zwelakhe Ngcobo was deliberately excluded to reduce severe class imbalance.

| Author | Profile URL |
|--------|-------------|
| Mhlengi Shangase | `https://www.isolezwe.co.za/authors/journalist/` |
| Zimbili Vilakazi | `https://www.isolezwe.co.za/authors/zee/` |
| Sabelo Nsele | `https://www.isolezwe.co.za/authors/mangethe/` |
| Simangaliso Ntshangase | `https://www.isolezwe.co.za/authors/health-and-lifestyle-journalist/` |
| Fanelesibonge Bengu | `https://www.isolezwe.co.za/authors/fanele-bengu/` |
| Nonkululeko Nhlapo | `https://www.isolezwe.co.za/authors/online/` |
| Nokubongwa Phenyane | `https://www.isolezwe.co.za/authors/maphenyane/` |
| Zakhele Xaba | `https://isolezwe.co.za/authors/zakhele-xaba/` |
| Sibusiso Mdlalose | `https://isolezwe.co.za/authors/mdlalose-wezemidlalo/` |
| Mthokozisi Mncuseni | `https://isolezwe.co.za/authors/your-football-guy/` |
| Charles Khuzwayo | `https://isolezwe.co.za/authors/entertainment/` |

## How it works

The pipeline runs in four stages for each author:

1. **Discover article URLs** (`get_article_links_selenium`): A headless Chrome browser (Selenium) loads the author profile page and clicks **Load more** up to four times to bypass Isolezwe’s default ~25-article listing cap. Parsed links must contain a section path (`/izindaba/`, `/ezemidlalo/`, or `/ezokungcebeleka/`) and a date in the form `YYYY-MM-DD`. Relative URLs are expanded to `https://www.isolezwe.co.za`; duplicates and very short URLs are dropped.

2. **Cap per author**: At most **80** candidate links per author (`MAX_ARTICLES_PER_AUTHOR`).

3. **Extract article text** (`scrape_article_text`): Each URL is fetched with `requests`; all `<p>` tags are joined. Paragraphs shorter than 20 characters are skipped (ads, share buttons). Lines starting with `Image:` are stripped of photo-credit boilerplate. Articles with fewer than 150 characters of body text are discarded.

4. **Save to CSV**: Successful rows are written to `data/raw/isizulu_authors_dataset.csv`.

A **1.2-second** delay between article requests reduces load on the server.

## Configuration

| Constant | Value | Role |
|----------|-------|------|
| `MAX_ARTICLES_PER_AUTHOR` | 80 | Upper bound on URLs scraped per journalist |
| `CLICKS_NEEDED` | 4 | Number of **Load more** clicks on each profile page |
| `OUTPUT_FILE` | `data/raw/isizulu_authors_dataset.csv` | Raw dataset path |

## Output

| File | Description |
|------|-------------|
| `data/raw/isizulu_authors_dataset.csv` | Raw dataset with columns `author`, `url`, `text` |

### Current dataset (last run)

| Statistic | Value |
|-----------|-------|
| Total articles | 877 |
| Authors | 11 |
| Articles per author | 77–80 (most authors have 80) |

Per-author counts from the latest scrape:

| Author | Articles |
|--------|----------|
| Charles Khuzwayo | 80 |
| Fanelesibonge Bengu | 80 |
| Mhlengi Shangase | 80 |
| Mthokozisi Mncuseni | 80 |
| Nokubongwa Phenyane | 80 |
| Nonkululeko Nhlapo | 80 |
| Sabelo Nsele | 80 |
| Sibusiso Mdlalose | 80 |
| Zakhele Xaba | 80 |
| Zimbili Vilakazi | 80 |
| Simangaliso Ntshangase | 77 |

Compared with the earlier static-scrape version (~263 articles, ~25 per author), the Selenium upgrade roughly **triples** corpus size and balances classes more evenly.

## Dependencies

- `selenium`: Headless Chrome automation for **Load more**
- `requests`: HTTP fetching of article pages
- `beautifulsoup4`: HTML parsing
- `pandas`: CSV export

You also need **Google Chrome** and a matching **ChromeDriver** on your `PATH` for Selenium. These packages are listed in `requirements.txt`.

## Usage

From the project root (with network access):

**macOS / Linux:**

```bash
python3 src/build_author_dataset.py
```

**Windows (Command Prompt or PowerShell):**

```cmd
python src\build_author_dataset.py
```

Re-scraping overwrites `data/raw/isizulu_authors_dataset.csv`. After a fresh scrape, re-run the downstream pipeline: `preprocessing.py` → `tokenizer.py` → `baselines.py` (or `python run_pipeline.py`).

## Notes and limitations

- **Network and browser required**: Isolezwe must be reachable; Selenium needs a working Chrome/ChromeDriver setup.
- **Site structure**: Link selectors and paragraph heuristics may break if Isolezwe changes its HTML layout.
- **Scraping etiquette**: Built-in delays limit request rate; avoid reducing sleeps when re-running at scale.
- **No cross-author deduplication**: The same article URL is not expected twice for one author; overlap across authors is not checked at scrape time.

## Next step

Run [`src/preprocessing.py`](2.1-preprocessing.md) to chunk long articles, apply stratified splits, and write `data/splits/*.csv`.
