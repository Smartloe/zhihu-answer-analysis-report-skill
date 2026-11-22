---
name: zhihu-answer-analysis-report
description: Standalone workflow that crawls every answer under a Zhihu question, then runs jieba tokenization, word cloud generation, SnowNLP sentiment analysis, LDA topic clustering, inferred author network analysis, and ECharts visualizations to produce a Markdown report. Use when asked to analyze a Zhihu question page, an answer URL that should be expanded to the full question, or an existing local scrape directory and deliver a report with charts and text insights.
---

# Zhihu Answer Analysis Report

Use this skill when the user wants a report, not just raw scraping output. The script resolves its own `core/` package from `__file__`; relative source/output arguments are resolved against the current working directory.

## Quick Start

1. Install the project dependencies (a virtualenv is recommended when the system Python is externally managed):

```bash
python3 -m pip install -r requirements.txt
# If pip reports "externally-managed-environment":
# python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
```

Installing the project itself is optional when you run the bundled script from the repo root; it adds `core/` to `sys.path` automatically.

2. For live crawling, make sure `cookies.json` contains a real `z_c0` (plus `d_c0`). A missing or expired cookie produces a `403`; export a fresh one from a logged-in browser. `cookie_pool/**/*.json` is also supported.

3. Run the bundled script.

For a Zhihu question URL:

```bash
python3 scripts/zhihu_answer_report.py \
  "https://www.zhihu.com/question/2010315360377799327"
```

For a Zhihu answer URL that should be expanded to the full question:

```bash
python3 scripts/zhihu_answer_report.py \
  "https://www.zhihu.com/question/2010315360377799327/answer/2011017391602151702"
```

For an existing local scrape directory:

```bash
python3 scripts/zhihu_answer_report.py \
  "./data/reports/<bundle>/raw"
```

For an existing `answers.jsonl` dataset:

```bash
python3 scripts/zhihu_answer_report.py \
  "./data/reports/<bundle>/analysis/answers.jsonl"
```

## Workflow

1. Normalize the input.
   Question URL: crawl that question.
   Answer URL: extract the question id and switch to the question page so the full answer set is analyzed.
   Local path: skip fetching and analyze existing Markdown outputs or `analysis/answers.jsonl`.

2. Fetch all answers when the input is a URL.
   Use the repository's protocol-first modules instead of inventing a second crawler.
   Respect cookies, humanized delays, and existing image/Markdown/SQLite behavior.

3. Build the analysis dataset.
   Convert each answer to clean text.
   Use `assets/stopwords_zh.txt` as the default stopword list.
   Preserve per-answer metadata such as author, date, URL, character count, and upvotes when available.

4. Generate outputs.
   `report.md`: the final Markdown report.
   `analysis/wordcloud.png`: static word cloud image.
   `analysis/dashboard.html`: self-contained, offline-capable ECharts dashboard with a light/dark toggle, sentiment charts, LDA topic clustering, and an inferred author network graph.
   `analysis/summary.json`: structured metrics.
   `analysis/answers.jsonl`: row-level answer dataset for reuse.
   `analysis/corpus.txt`: flattened text corpus used by the analysis.
   `raw/entries/*/index.md` and `raw/zhihu.db`: raw scrape artifacts when running from a URL.

## Default Command Pattern

Use the script directly unless the user asks for a custom flow:

```bash
python3 scripts/zhihu_answer_report.py <source>
```

Useful flags:

- `--answer-cap N`: cap the crawl when the question is extremely large.
- `--output-dir PATH`: choose a custom report directory.
- `--no-images`: skip image downloads during fetch mode to speed up long runs.
- `--top-k N`: number of top keywords shown in tables and charts.
- `--min-word-length N`: minimum token length kept after jieba segmentation.
- `--stopwords FILE`: merge custom stopwords with the bundled list.
- `--font-path FILE`: override the Chinese font used for the word cloud.
- `--lda-topics N`: number of LDA topics (set to 0 to disable).
- `--lda-words N`: top words per LDA topic.
- `--lda-max-iter N`: LDA training iterations.
- `--network-max-nodes N`: number of author nodes kept in the inferred author network (`0` disables it).
- `--network-max-edges N`: number of edges kept in the inferred author network.

## Decision Notes

- If the question has a very large answer count, surface the tradeoff before running a full scrape. Thousands of answers can take a long time and raise anti-bot risk.
- If dependencies are missing, install them first rather than rewriting the analysis stack.
- If the user already has local outputs, prefer path mode over re-scraping.
- If the word cloud font cannot be resolved automatically, pass `--font-path` explicitly or set `ZHIHU_FONT_PATH`.
- If requests fail with `Resolving timed out` / `curl: (28)`, the bundled libcurl cannot read the system resolver. The script auto-detects DNS servers; override with `ZHIHU_DNS_SERVERS="223.5.5.5,119.29.29.29"` or disable the fix with `ZHIHU_DISABLE_DNS_FIX=1`.
- If the Zhihu API returns `403`, the `z_c0` cookie is missing or expired. Export a fresh cookie rather than retrying in a loop.
- The author network should be described as an inferred association graph, not Zhihu's real follow or comment relationship graph.

## Report Contract

Use this default report structure unless the user asks for something more specific:

```markdown
# [Question Title] 知乎回答分析报告

## 核心结论
[2-4 bullet points]

## 数据概览
[answer count, author count, character stats, date range, positive/neutral/negative split]

## 词云与高频词
[embedded word cloud image + top keywords table]

## 主题聚类 (LDA)
[topic clusters with top words and coverage share]

## 作者关联网络
[inferred author network based on topic and keyword overlap; clarify that it is not a real follow graph]

## ECharts 可视化
[link to the dashboard plus an embedded iframe that renders the local ECharts dashboard]

## 情感分析
[average score, bucket distribution, interpretation]

## 高频作者
[top authors by answer count]

## 代表性回答
[top answers by upvotes if available, otherwise by length]

## 数据文件
[optional references to summary.json and answers.jsonl]

> 方法与局限
[one note describing SnowNLP, jieba, LDA and the inferred network's limits]
```

## Resource Notes

- `scripts/zhihu_answer_report.py` is the main deterministic entry point.
- `assets/stopwords_zh.txt` contains the default Chinese stopwords and report-noise tokens. Extend it when a topic introduces domain-specific filler words.
- `assets/dashboard.css` and `assets/dashboard.js` define the dashboard's editorial layout, themes and chart behaviour. `assets/echarts.min.js` is a vendored ECharts build so the dashboard works offline. All three are inlined into `analysis/dashboard.html` at generation time; edit the assets rather than the generated HTML.
- `tests/` contains the smoke-test suite. Run `python3 -m pytest tests/ -q`; network cases are opt-in with `RUN_NETWORK_TESTS=1`.
