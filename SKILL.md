---
name: zhihu-answer-analysis-report
description: Standalone workflow that crawls every answer under a Zhihu question, then runs jieba tokenization, word cloud generation, SnowNLP sentiment analysis, and ECharts visualizations to produce a Markdown report. Use when asked to analyze a Zhihu question page, an answer URL that should be expanded to the full question, or an existing local scrape directory and deliver a report with charts and text insights.
---

# Zhihu Answer Analysis Report

Use this skill when the user wants a report, not just raw scraping output. This repository is self-contained and assumes the working directory is the repo root.

## Quick Start

1. Install the project dependencies:

```bash
python3 -m pip install -e .
```

2. Run the bundled script.

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
  "./data/entries"
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
   `analysis/dashboard.html`: responsive ECharts dashboard with a desktop two-column card layout and mobile single-column fallback.
   `analysis/summary.json`: structured metrics.
   `analysis/answers.jsonl`: row-level answer dataset for reuse.
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
- `--stopwords FILE`: merge custom stopwords with the bundled list.
- `--font-path FILE`: override the Chinese font used for the word cloud.

## Decision Notes

- If the question has a very large answer count, surface the tradeoff before running a full scrape. Thousands of answers can take a long time and raise anti-bot risk.
- If dependencies are missing, install them first rather than rewriting the analysis stack.
- If the user already has local outputs, prefer path mode over re-scraping.
- If the word cloud font cannot be resolved automatically, pass `--font-path` explicitly.

## Report Contract

Use this default report structure unless the user asks for something more specific:

```markdown
# [Question Title] 知乎回答分析报告

## 核心结论
[2-4 bullet points]

## 数据概览
[answer count, character count, average length, positive/neutral/negative split]

## 词云与高频词
[embedded word cloud image + top keywords table]

## ECharts 可视化
[embedded iframe that renders the local ECharts dashboard directly inside the Markdown report]

## 情感分析
[average score, bucket distribution, interpretation]

## 代表性回答
[top answers by upvotes if available, otherwise by length]

## 数据文件
[optional references to summary.json and answers.jsonl]
```

## Resource Notes

- `scripts/zhihu_answer_report.py` is the main deterministic entry point.
- `assets/stopwords_zh.txt` contains the default Chinese stopwords and report-noise tokens. Extend it when a topic introduces domain-specific filler words.
