# 1. Data collection (`src/build_author_dataset.py`)

This script scrapes isiZulu news articles from [Isolezwe](https://isolezwe.co.za) author profile pages and builds the raw authorship dataset used by every downstream step.

## Purpose

Existing public isiZulu corpora either lack enough text per author or do not label articles by writer. This scraper collects labelled articles from a single news outlet so that each sample has a known author — the core requirement for authorship attribution.

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

1. **Discover article URLs** (`get_article_links`) — Fetches the author profile page and collects links that look like real articles. A link must contain a section path (`/izindaba/`, `/ezemidlalo/`, or `/ezokungcebeleka/`) and a date in the form `YYYY-MM-DD`. Relative URLs are expanded to full `https://www.isolezwe.co.za` URLs; duplicates and very short URLs are dropped.

2. **Cap per author** — At most 50 candidate links are kept per author so classes stay roughly balanced.

3. **Extract article text** (`scrape_article_text`) — Each URL is fetched; all `<p>` tags are joined. Paragraphs shorter than 20 characters are skipped (photo credits, ads). Articles with fewer than 150 characters of body text are discarded.

4. **Save to CSV** — Successful rows are written to `data/raw/isizulu_authors_dataset.csv`.

A 1.5-second delay between article requests reduces load on the server.

## Output

| File | Description |
|------|-------------|
| `data/raw/isizulu_authors_dataset.csv` | Raw dataset with columns `author`, `url`, `text` |

### Current dataset (last run)

| Statistic | Value |
|-----------|-------|
| Total articles | 263 |
| Authors | 11 |
| Total words | ~69,700 |
| Mean words per article | ~265 |
| Articles per author | 19–25 (most authors have 25) |

Per-author counts from the latest scrape:

| Author | Articles |
|--------|----------|
| Mhlengi Shangase | 25 |
| Zimbili Vilakazi | 25 |
| Sabelo Nsele | 25 |
| Fanelesibonge Bengu | 25 |
| Nokubongwa Phenyane | 25 |
| Zakhele Xaba | 25 |
| Mthokozisi Mncuseni | 25 |
| Charles Khuzwayo | 25 |
| Sibusiso Mdlalose | 24 |
| Nonkululeko Nhlapo | 20 |
| Simangaliso Ntshangase | 19 |

## Dependencies

- `requests` — HTTP fetching
- `beautifulsoup4` — HTML parsing
- `pandas` — CSV export

These are used by the scraper but are not listed in `requirements.txt` (only the ML pipeline dependencies are pinned there). Install them if you re-run the scraper:

```bash
pip install requests beautifulsoup4 pandas
```

## Usage

From the project root (with network access):

```bash
python3 src/build_author_dataset.py
```

Re-scraping will overwrite `data/raw/isizulu_authors_dataset.csv`. After a fresh scrape, re-run the downstream pipeline in order: `preprocessing.py` → `tokenizer.py` → `baselines.py`.

## Notes and limitations

- **Network required** — The script cannot run offline; Isolezwe must be reachable.
- **Site structure** — Link and paragraph heuristics may break if Isolezwe changes its HTML layout.
- **Small corpus** — ~263 articles is modest by NLP standards, but appropriate for exploring low-resource African language authorship attribution.
- **No deduplication across authors** — The same article URL should only appear once per author profile; cross-author duplicates are not explicitly checked at scrape time (overlap is checked later during splitting).

## Next step

Run [`src/preprocessing.py`](2.2-preprocessing.md) to create stratified train/validation/test splits from this CSV.
