#!/usr/bin/env python3
"""
Generate a Markdown report for all answers under a Zhihu question.

The script supports two modes:
1. URL mode: fetch every answer under a Zhihu question, or normalize an answer URL to its parent question.
2. Path mode: analyze an existing local scrape directory or an existing answers.jsonl dataset.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import statistics
import sys
from collections import Counter
from datetime import datetime
from html import unescape
from pathlib import Path
from typing import Any, Iterable


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parent.parent
SKILL_ROOT = REPO_ROOT

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.api_client import ZhihuAPIClient
from core.cookie_manager import cookie_manager
from core.config import get_config
from core.converter import ZhihuConverter
from core.db import ZhihuDatabase
from core.scraper import ZhihuDownloader
from core.utils import sanitize_filename


QUESTION_RE = re.compile(r"https?://www\.zhihu\.com/question/(\d+)(?:/answer/(\d+))?")
SOURCE_RE = re.compile(r"\[([^\]]+)\]\((https?://[^)]+)\)")
DATE_RE = re.compile(r"^\> \*\*Date / 日期\*\*: ([0-9-]+)", re.M)
AUTHOR_RE = re.compile(r"^\> \*\*Author / 作者\*\*: (.+)$", re.M)
TITLE_RE = re.compile(r"^# (.+)$", re.M)


class PreparedSource:
    def __init__(
        self,
        mode: str,
        source: str,
        *,
        question_id: str | None = None,
        question_url: str | None = None,
        question_title: str | None = None,
        total_answers: int | None = None,
        path: Path | None = None,
    ) -> None:
        self.mode = mode
        self.source = source
        self.question_id = question_id
        self.question_url = question_url
        self.question_title = question_title
        self.total_answers = total_answers
        self.path = path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Crawl Zhihu question answers and generate a Markdown analysis report.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("source", help="Zhihu question/answer URL, or an existing local output path.")
    parser.add_argument(
        "--output-dir",
        help="Directory for the generated report bundle. Defaults to a dated folder in data/reports for URL mode.",
    )
    parser.add_argument(
        "--answer-cap",
        type=int,
        default=None,
        help="Optional cap for the number of answers to fetch in URL mode.",
    )
    parser.add_argument(
        "--no-images",
        action="store_true",
        help="Skip image downloads during fetch mode.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=30,
        help="Number of top keywords shown in tables and charts.",
    )
    parser.add_argument(
        "--min-word-length",
        type=int,
        default=2,
        help="Minimum token length kept in jieba results.",
    )
    parser.add_argument(
        "--stopwords",
        help="Optional extra stopwords file, one token per line.",
    )
    parser.add_argument(
        "--font-path",
        help="Override the font used by the word cloud.",
    )
    return parser.parse_args()


def load_analysis_dependencies() -> tuple[Any, Any, Any]:
    missing: list[str] = []
    try:
        import jieba
    except ImportError:
        jieba = None
        missing.append("jieba")

    try:
        from snownlp import SnowNLP
    except ImportError:
        SnowNLP = None
        missing.append("snownlp")

    try:
        from wordcloud import WordCloud
    except ImportError:
        WordCloud = None
        missing.append("wordcloud")

    if missing:
        raise SystemExit(
            "Missing analysis dependencies: "
            f"{', '.join(missing)}\n"
            f"Install them with:\npython3 -m pip install {' '.join(missing)}"
        )

    return jieba, SnowNLP, WordCloud


def prepare_source(source: str) -> PreparedSource:
    source = source.strip()
    candidate_path = Path(source).expanduser()
    if candidate_path.exists():
        return PreparedSource(mode="path", source=source, path=candidate_path.resolve())

    match = QUESTION_RE.match(source)
    if not match:
        raise SystemExit("Only Zhihu question URLs, answer URLs, or local paths are supported.")

    question_id = match.group(1)
    question_url = f"https://www.zhihu.com/question/{question_id}"
    return PreparedSource(
        mode="url",
        source=source,
        question_id=question_id,
        question_url=question_url,
    )


def require_cookies_for_url_mode() -> None:
    if cookie_manager.has_sessions():
        return

    template_path = REPO_ROOT / "cookies.example.json"
    target_path = REPO_ROOT / "cookies.json"
    raise SystemExit(
        "Zhihu URL mode requires a valid login cookie, but no usable session was found.\n"
        f"Create `{target_path}` from `{template_path}` and fill in real `z_c0` / `d_c0` values, "
        "or add one or more JSON files under `cookie_pool/`."
    )


def fetch_question_overview(prepared: PreparedSource) -> PreparedSource:
    if prepared.mode != "url" or not prepared.question_id:
        return prepared

    client = ZhihuAPIClient()
    page = client.get_question_answers_page(prepared.question_id, limit=1, offset=0)
    answers = page.get("data", [])
    title = f"question-{prepared.question_id}"
    if answers:
        title = answers[0].get("question", {}).get("title", title)

    totals = page.get("paging", {}).get("totals") or len(answers)
    prepared.question_title = title
    prepared.total_answers = int(totals)
    return prepared


def default_output_dir(prepared: PreparedSource) -> Path:
    if prepared.mode == "path" and prepared.path:
        base_dir = prepared.path.parent if prepared.path.is_file() else prepared.path
        return base_dir / "analysis-report"

    today = datetime.now().strftime("%Y-%m-%d")
    safe_title = sanitize_filename(prepared.question_title or "zhihu-report", max_length=80)
    bundle_name = f"[{today}] {safe_title} (question-{prepared.question_id})"
    return REPO_ROOT / "data" / "reports" / bundle_name


def resolve_output_dir(prepared: PreparedSource, explicit_output: str | None) -> Path:
    if explicit_output:
        return Path(explicit_output).expanduser().resolve()
    return default_output_dir(prepared).resolve()


def build_output_folder_name(item_date: str, title: str, author: str, item_key: str) -> str:
    cfg = get_config()
    folder_template = cfg.output.folder_format or "[{date}] {title}"
    try:
        rendered = folder_template.format(date=item_date, title=title, author=author)
    except KeyError:
        rendered = f"[{item_date}] {title}"

    rendered = sanitize_filename(rendered, max_length=120)
    return f"{rendered} ({item_key})"


def extract_text_from_markdown(markdown: str) -> str:
    body = markdown.split("\n---\n", 1)[1] if "\n---\n" in markdown else markdown
    body = re.sub(r"```.*?```", " ", body, flags=re.S)
    body = re.sub(r"`([^`]+)`", r"\1", body)
    body = re.sub(r"!\[[^\]]*\]\([^)]+\)", " ", body)
    body = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1", body)
    body = re.sub(r"<[^>]+>", " ", body)
    body = re.sub(r"^>\s*.*$", " ", body, flags=re.M)
    body = re.sub(r"^#+\s*", " ", body, flags=re.M)
    body = re.sub(r"^\s*[-*]\s+", " ", body, flags=re.M)
    body = body.replace("|", " ")
    body = unescape(body)
    body = re.sub(r"\s+", " ", body)
    return body.strip()


def load_stopwords(extra_stopwords: str | None) -> set[str]:
    stopwords_path = SKILL_ROOT / "assets" / "stopwords_zh.txt"
    stopwords = {
        line.strip()
        for line in stopwords_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    if extra_stopwords:
        extra_path = Path(extra_stopwords).expanduser().resolve()
        stopwords.update(
            line.strip()
            for line in extra_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    return stopwords


def resolve_wordcloud_font(font_override: str | None) -> str:
    candidates = []
    if font_override:
        candidates.append(Path(font_override).expanduser())

    candidates.extend(
        Path(candidate)
        for candidate in [
            "/System/Library/Fonts/Supplemental/Songti.ttc",
            "/System/Library/Fonts/STHeiti Medium.ttc",
            "/System/Library/Fonts/STHeiti Light.ttc",
            "/System/Library/Fonts/PingFang.ttc",
            "/System/Library/AssetsV2/com_apple_MobileAsset_Font8/259e8f5a322e8dae602d51ac00aefb3d6b05c224.asset/AssetData/SimSong.ttc",
        ]
    )

    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())

    raise SystemExit("No usable Chinese font found for the word cloud. Pass --font-path explicitly.")


async def fetch_and_save_answers(
    prepared: PreparedSource,
    output_dir: Path,
    download_images: bool,
) -> list[dict[str, Any]]:
    if not prepared.question_url:
        raise SystemExit("Question URL is required in fetch mode.")

    target_answers = prepared.total_answers or 0
    if target_answers <= 0:
        raise SystemExit("The question returned zero answers.")

    raw_root = output_dir / "raw"
    content_root = raw_root / "entries"
    content_root.mkdir(parents=True, exist_ok=True)

    downloader = ZhihuDownloader(prepared.question_url)
    items = await downloader.fetch_page(limit=target_answers)
    if not isinstance(items, list) or not items:
        raise SystemExit("No answers were fetched.")

    cfg = get_config()
    images_subdir = cfg.output.images_subdir or "images"
    db = ZhihuDatabase(str(raw_root / "zhihu.db"))
    saved_records: list[dict[str, Any]] = []
    today = datetime.now().strftime("%Y-%m-%d")

    try:
        for item in items:
            title = sanitize_filename(item.get("title", "Untitled"), max_length=80)
            author = sanitize_filename(item.get("author", "Unknown"), max_length=40)
            item_date = item.get("date") or today
            item_key = sanitize_filename(
                f"{item.get('type', 'answer')}-{item.get('id', 'unknown')}",
                max_length=80,
            )
            folder_name = build_output_folder_name(item_date, title, author, item_key)
            folder = content_root / folder_name
            folder.mkdir(parents=True, exist_ok=True)

            img_map: dict[str, str] = {}
            if download_images:
                img_urls = ZhihuConverter.extract_image_urls(item.get("html", ""))
                if img_urls:
                    img_map = await ZhihuDownloader.download_images(
                        img_urls,
                        folder / images_subdir,
                        relative_prefix=images_subdir,
                        concurrency=cfg.crawler.images.concurrency,
                        timeout=cfg.crawler.images.timeout,
                    )

            markdown = ZhihuConverter(img_map=img_map).convert(item.get("html", ""))
            source_url = item.get("url") or prepared.question_url
            header = (
                f"# {item.get('title', 'Untitled')}\n\n"
                f"> **Author / 作者**: {item.get('author', 'Unknown')}  \n"
                f"> **Source / 来源**: [{source_url}]({source_url})  \n"
                f"> **Date / 日期**: {item_date}\n\n"
                "---\n\n"
            )
            full_markdown = header + markdown
            out_path = folder / "index.md"
            out_path.write_text(full_markdown, encoding="utf-8")
            db.save_article(item, full_markdown)

            saved_records.append(
                {
                    "id": str(item.get("id", "")),
                    "title": item.get("title", ""),
                    "author": item.get("author", ""),
                    "url": source_url,
                    "date": item_date,
                    "type": item.get("type", "answer"),
                    "upvotes": item.get("upvotes"),
                    "markdown_path": str(out_path.resolve()),
                    "text": extract_text_from_markdown(markdown),
                }
            )
    finally:
        db.close()

    return saved_records


def parse_markdown_file(path: Path) -> dict[str, Any]:
    content = path.read_text(encoding="utf-8")
    title_match = TITLE_RE.search(content)
    title = title_match.group(1).strip() if title_match else path.parent.name

    author_match = AUTHOR_RE.search(content)
    author = author_match.group(1).strip() if author_match else "Unknown"

    source_match = SOURCE_RE.search(content)
    url = source_match.group(2).strip() if source_match else ""

    date_match = DATE_RE.search(content)
    item_date = date_match.group(1).strip() if date_match else ""

    answer_match = re.search(r"/answer/(\d+)", url)
    answer_id = answer_match.group(1) if answer_match else ""

    return {
        "id": answer_id,
        "title": title,
        "author": author,
        "url": url,
        "date": item_date,
        "type": "answer",
        "upvotes": None,
        "markdown_path": str(path.resolve()),
        "text": extract_text_from_markdown(content),
    }


def load_records_from_path(path: Path) -> list[dict[str, Any]]:
    path = path.resolve()
    if path.is_file() and path.name == "answers.jsonl":
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    dataset_path = path / "analysis" / "answers.jsonl"
    if dataset_path.exists():
        return load_records_from_path(dataset_path)

    if path.is_file() and path.suffix == ".md":
        markdown_files = [path]
    else:
        markdown_files = sorted(path.rglob("index.md"))

    if not markdown_files:
        raise SystemExit(f"No Markdown answers found under: {path}")

    return [parse_markdown_file(markdown_path) for markdown_path in markdown_files]


def normalize_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for record in records:
        copied = dict(record)
        copied["markdown_path"] = str(Path(copied["markdown_path"]).resolve())
        copied["text"] = extract_text_from_markdown(copied.get("text", ""))
        normalized.append(copied)
    return normalized


def tokenize_records(
    records: list[dict[str, Any]],
    jieba: Any,
    stopwords: set[str],
    min_word_length: int,
) -> Counter:
    frequencies: Counter = Counter()
    for record in records:
        text = record.get("text", "").strip()
        tokens: list[str] = []
        for token in jieba.cut(text, cut_all=False):
            token = token.strip().lower()
            if not token:
                continue
            if len(token) < min_word_length:
                continue
            if token in stopwords:
                continue
            if re.fullmatch(r"[_\W]+", token):
                continue
            if token.isdigit():
                continue
            tokens.append(token)

        record["char_count"] = len(text)
        record["token_count"] = len(tokens)
        frequencies.update(tokens)

    return frequencies


def score_sentiment(records: list[dict[str, Any]], SnowNLP: Any) -> None:
    for record in records:
        text = record.get("text", "")
        sample = text[:6000]
        if not sample:
            record["sentiment"] = None
            continue
        try:
            record["sentiment"] = round(float(SnowNLP(sample).sentiments), 4)
        except Exception:
            record["sentiment"] = None


def sentiment_bucket(score: float | None) -> str:
    if score is None:
        return "unknown"
    if score < 0.2:
        return "0.0-0.2"
    if score < 0.4:
        return "0.2-0.4"
    if score < 0.6:
        return "0.4-0.6"
    if score < 0.8:
        return "0.6-0.8"
    return "0.8-1.0"


def sentiment_label(score: float | None) -> str:
    if score is None:
        return "unknown"
    if score < 0.4:
        return "negative"
    if score <= 0.6:
        return "neutral"
    return "positive"


def build_summary(
    records: list[dict[str, Any]],
    frequencies: Counter,
    prepared: PreparedSource,
    top_k: int,
) -> dict[str, Any]:
    sentiments = [record["sentiment"] for record in records if record.get("sentiment") is not None]
    char_counts = [record.get("char_count", 0) for record in records]
    upvote_records = [record for record in records if isinstance(record.get("upvotes"), (int, float))]

    label_counts = Counter(sentiment_label(record.get("sentiment")) for record in records)
    bucket_counts = Counter(sentiment_bucket(record.get("sentiment")) for record in records)
    timeline = Counter(record.get("date") or "unknown" for record in records)
    authors = Counter(record.get("author") or "Unknown" for record in records)

    if upvote_records:
        top_answers = sorted(
            upvote_records,
            key=lambda item: (item.get("upvotes") or 0, item.get("char_count") or 0),
            reverse=True,
        )[:5]
    else:
        top_answers = sorted(
            records,
            key=lambda item: (item.get("char_count") or 0, item.get("token_count") or 0),
            reverse=True,
        )[:5]

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source": prepared.source,
        "question_url": prepared.question_url,
        "question_id": prepared.question_id,
        "question_title": prepared.question_title,
        "answer_count": len(records),
        "total_characters": sum(char_counts),
        "average_characters": round(sum(char_counts) / len(char_counts), 2) if char_counts else 0,
        "median_characters": statistics.median(char_counts) if char_counts else 0,
        "average_sentiment": round(sum(sentiments) / len(sentiments), 4) if sentiments else None,
        "sentiment_stddev": round(statistics.pstdev(sentiments), 4) if len(sentiments) > 1 else 0,
        "sentiment_labels": dict(label_counts),
        "sentiment_buckets": dict(bucket_counts),
        "top_keywords": [
            {"word": word, "count": count}
            for word, count in frequencies.most_common(top_k)
        ],
        "top_authors": [
            {"author": author, "count": count}
            for author, count in authors.most_common(10)
        ],
        "timeline": [
            {"date": date, "count": count}
            for date, count in sorted(timeline.items())
        ],
        "top_answers": [
            {
                "title": item.get("title", ""),
                "author": item.get("author", ""),
                "date": item.get("date", ""),
                "url": item.get("url", ""),
                "upvotes": item.get("upvotes"),
                "char_count": item.get("char_count", 0),
                "sentiment": item.get("sentiment"),
                "markdown_path": item.get("markdown_path"),
            }
            for item in top_answers
        ],
    }


def create_wordcloud(
    frequencies: Counter,
    output_path: Path,
    WordCloud: Any,
    font_path: str,
) -> None:
    top_frequencies = dict(frequencies.most_common(200))
    if not top_frequencies:
        raise SystemExit("No usable keywords were produced for the word cloud.")

    cloud = WordCloud(
        width=1600,
        height=900,
        background_color="white",
        font_path=font_path,
        max_words=200,
        collocations=False,
    )
    cloud.generate_from_frequencies(top_frequencies)
    cloud.to_file(str(output_path))


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    lines = [json.dumps(record, ensure_ascii=False) for record in records]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_dashboard(path: Path, summary: dict[str, Any], records: list[dict[str, Any]]) -> None:
    top_keywords = summary["top_keywords"][:20]
    timeline = summary["timeline"]
    sentiment_buckets = summary["sentiment_buckets"]

    scatter_points = []
    for record in records:
        if record.get("sentiment") is None:
            continue
        scatter_points.append(
            [
                record.get("char_count", 0),
                0 if record.get("upvotes") is None else record.get("upvotes"),
                record.get("sentiment"),
                record.get("title", ""),
            ]
        )

    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{summary.get("question_title") or "Zhihu Dashboard"}</title>
  <script src="https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js"></script>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f8f5ef;
      --panel: #fffdf9;
      --ink: #1f2328;
      --line: #d9cfc1;
    }}
    body {{
      margin: 0;
      padding: 24px;
      background: radial-gradient(circle at top, #fff4df, var(--bg));
      color: var(--ink);
      font-family: "SF Pro Text", "PingFang SC", "Helvetica Neue", sans-serif;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 28px;
    }}
    p {{
      margin: 0 0 24px;
      color: #5a544c;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 18px;
    }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 16px;
      box-shadow: 0 10px 30px rgba(46, 30, 13, 0.08);
    }}
    .chart {{
      width: 100%;
      height: 360px;
    }}
  </style>
</head>
<body>
  <h1>{summary.get("question_title") or "知乎回答分析"}</h1>
  <p>回答数 {summary.get("answer_count", 0)}，平均情感分 {summary.get("average_sentiment")}, 平均字数 {summary.get("average_characters")}。</p>
  <div class="grid">
    <section class="card"><div id="keywords" class="chart"></div></section>
    <section class="card"><div id="sentiment" class="chart"></div></section>
    <section class="card"><div id="timeline" class="chart"></div></section>
    <section class="card"><div id="scatter" class="chart"></div></section>
  </div>
  <script>
    const topKeywords = {json.dumps(top_keywords, ensure_ascii=False)};
    const sentimentBuckets = {json.dumps(sentiment_buckets, ensure_ascii=False)};
    const timeline = {json.dumps(timeline, ensure_ascii=False)};
    const scatterPoints = {json.dumps(scatter_points, ensure_ascii=False)};

    const keywordChart = echarts.init(document.getElementById('keywords'));
    keywordChart.setOption({{
      title: {{ text: '高频关键词 Top 20' }},
      tooltip: {{}},
      xAxis: {{
        type: 'category',
        data: topKeywords.map(item => item.word),
        axisLabel: {{ rotate: 35 }}
      }},
      yAxis: {{ type: 'value' }},
      series: [{{
        type: 'bar',
        data: topKeywords.map(item => item.count),
        itemStyle: {{ color: '#c96d3a' }},
        barMaxWidth: 28
      }}]
    }});

    const bucketOrder = ['0.0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1.0', 'unknown'];
    const sentimentChart = echarts.init(document.getElementById('sentiment'));
    sentimentChart.setOption({{
      title: {{ text: '情感分布' }},
      tooltip: {{}},
      xAxis: {{ type: 'category', data: bucketOrder }},
      yAxis: {{ type: 'value' }},
      series: [{{
        type: 'line',
        smooth: true,
        data: bucketOrder.map(name => sentimentBuckets[name] || 0),
        lineStyle: {{ color: '#287271', width: 3 }},
        areaStyle: {{ color: 'rgba(40, 114, 113, 0.18)' }}
      }}]
    }});

    const timelineChart = echarts.init(document.getElementById('timeline'));
    timelineChart.setOption({{
      title: {{ text: '回答时间线' }},
      tooltip: {{}},
      xAxis: {{ type: 'category', data: timeline.map(item => item.date), axisLabel: {{ rotate: 35 }} }},
      yAxis: {{ type: 'value' }},
      series: [{{
        type: 'bar',
        data: timeline.map(item => item.count),
        itemStyle: {{ color: '#4c7aaf' }},
        barMaxWidth: 26
      }}]
    }});

    const scatterChart = echarts.init(document.getElementById('scatter'));
    scatterChart.setOption({{
      title: {{ text: '字数 / 赞同 / 情感' }},
      tooltip: {{
        formatter: params => {{
          const data = params.data;
          return `${{data[3]}}<br>字数: ${{data[0]}}<br>赞同: ${{data[1]}}<br>情感: ${{data[2]}}`;
        }}
      }},
      xAxis: {{ type: 'value', name: '字数' }},
      yAxis: {{ type: 'value', name: '赞同数' }},
      visualMap: {{
        min: 0,
        max: 1,
        dimension: 2,
        orient: 'horizontal',
        left: 'center',
        bottom: 0,
        inRange: {{ color: ['#6b1d1d', '#d6c65b', '#276749'] }}
      }},
      series: [{{
        type: 'scatter',
        symbolSize: value => Math.max(10, Math.sqrt(value[1] + 1) * 3),
        data: scatterPoints
      }}]
    }});

    window.addEventListener('resize', () => {{
      keywordChart.resize();
      sentimentChart.resize();
      timelineChart.resize();
      scatterChart.resize();
    }});
  </script>
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")


def make_relative_link(target: str | Path, base_dir: Path) -> str:
    return os.path.relpath(str(target), start=str(base_dir))


def build_key_findings(summary: dict[str, Any]) -> list[str]:
    findings = [
        f"共分析 {summary['answer_count']} 条回答，累计 {summary['total_characters']} 字。",
        f"平均回答长度为 {summary['average_characters']} 字，中位数为 {summary['median_characters']} 字。",
    ]
    if summary.get("average_sentiment") is not None:
        findings.append(
            "SnowNLP 平均情感分为 "
            f"{summary['average_sentiment']}，正向 {summary['sentiment_labels'].get('positive', 0)} 条，"
            f"中性 {summary['sentiment_labels'].get('neutral', 0)} 条，"
            f"负向 {summary['sentiment_labels'].get('negative', 0)} 条。"
        )
    keywords = ", ".join(item["word"] for item in summary["top_keywords"][:8])
    if keywords:
        findings.append(f"最显著的高频词包括：{keywords}。")
    return findings


def write_report(
    report_path: Path,
    summary: dict[str, Any],
    output_dir: Path,
) -> None:
    wordcloud_rel = make_relative_link(output_dir / "analysis" / "wordcloud.png", report_path.parent)
    dashboard_rel = make_relative_link(output_dir / "analysis" / "dashboard.html", report_path.parent)
    summary_rel = make_relative_link(output_dir / "analysis" / "summary.json", report_path.parent)
    dataset_rel = make_relative_link(output_dir / "analysis" / "answers.jsonl", report_path.parent)

    lines = [
        f"# {summary.get('question_title') or '知乎回答分析报告'}",
        "",
        f"> **Generated At / 生成时间**: {summary['generated_at']}  ",
        f"> **Source / 输入来源**: `{summary.get('source')}`  ",
        f"> **Question URL / 问题链接**: {summary.get('question_url') or 'N/A'}  ",
        f"> **Answer Count / 回答数**: {summary['answer_count']}",
        "",
        "## 核心结论",
        "",
    ]

    for finding in build_key_findings(summary):
        lines.append(f"- {finding}")

    lines.extend(
        [
            "",
            "## 数据概览",
            "",
            "| 指标 | 数值 |",
            "|---|---|",
            f"| 回答数 | {summary['answer_count']} |",
            f"| 总字数 | {summary['total_characters']} |",
            f"| 平均字数 | {summary['average_characters']} |",
            f"| 中位数字数 | {summary['median_characters']} |",
            f"| 平均情感分 | {summary.get('average_sentiment', 'N/A')} |",
            f"| 情感标准差 | {summary.get('sentiment_stddev', 'N/A')} |",
            f"| 正向回答数 | {summary['sentiment_labels'].get('positive', 0)} |",
            f"| 中性回答数 | {summary['sentiment_labels'].get('neutral', 0)} |",
            f"| 负向回答数 | {summary['sentiment_labels'].get('negative', 0)} |",
            "",
            "## 词云与高频词",
            "",
            f"![词云图]({wordcloud_rel})",
            "",
            "| 关键词 | 频次 |",
            "|---|---|",
        ]
    )

    for item in summary["top_keywords"][:20]:
        lines.append(f"| {item['word']} | {item['count']} |")

    lines.extend(
        [
            "",
            "## 情感分析",
            "",
            "情感分使用 SnowNLP 生成，范围接近 0 到 1。分值越接近 1，文本越偏正向；越接近 0，文本越偏负向。",
            "",
            "| 区间 | 回答数 |",
            "|---|---|",
        ]
    )

    for bucket in ["0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0", "unknown"]:
        lines.append(f"| {bucket} | {summary['sentiment_buckets'].get(bucket, 0)} |")

    lines.extend(
        [
            "",
            "## ECharts 可视化",
            "",
            "<div style=\"margin: 16px 0 24px;\">",
            f"<iframe src=\"{dashboard_rel}\" title=\"ECharts Dashboard\" "
            "style=\"width: 100%; min-height: 1580px; border: 1px solid #e5e7eb; border-radius: 16px; background: #ffffff;\" "
            "loading=\"lazy\"></iframe>",
            "</div>",
            "",
        ]
    )

    lines.extend(
        [
            "## 高频作者",
            "",
            "| 作者 | 回答数 |",
            "|---|---|",
        ]
    )

    for item in summary["top_authors"][:10]:
        lines.append(f"| {item['author']} | {item['count']} |")

    lines.extend(
        [
            "",
            "## 代表性回答",
            "",
            "| 标题 | 作者 | 日期 | 赞同数 | 字数 | 情感分 | Markdown |",
            "|---|---|---|---|---|---|---|",
        ]
    )

    for item in summary["top_answers"]:
        markdown_rel = make_relative_link(item["markdown_path"], report_path.parent)
        safe_title = item["title"].replace("|", "\\|")
        safe_author = item["author"].replace("|", "\\|")
        lines.append(
            f"| {safe_title} | {safe_author} | {item.get('date') or ''} | "
            f"{item.get('upvotes') if item.get('upvotes') is not None else 'N/A'} | "
            f"{item.get('char_count', 0)} | "
            f"{item.get('sentiment') if item.get('sentiment') is not None else 'N/A'} | "
            f"[index.md]({markdown_rel}) |"
        )

    lines.extend(
        [
            "",
            "## 数据文件",
            "",
            f"- [结构化摘要 summary.json]({summary_rel})",
            f"- [答案数据 answers.jsonl]({dataset_rel})",
        ]
    )

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_outputs(
    output_dir: Path,
    summary: dict[str, Any],
    records: list[dict[str, Any]],
    frequencies: Counter,
    WordCloud: Any,
    font_path: str,
) -> None:
    analysis_dir = output_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    answers_jsonl = analysis_dir / "answers.jsonl"
    summary_json = analysis_dir / "summary.json"
    corpus_txt = analysis_dir / "corpus.txt"
    wordcloud_png = analysis_dir / "wordcloud.png"
    dashboard_html = analysis_dir / "dashboard.html"
    report_md = output_dir / "report.md"

    write_jsonl(answers_jsonl, records)
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    corpus_txt.write_text("\n\n".join(record.get("text", "") for record in records), encoding="utf-8")
    create_wordcloud(frequencies, wordcloud_png, WordCloud, font_path)
    write_dashboard(dashboard_html, summary, records)
    write_report(report_md, summary, output_dir)


async def main() -> None:
    args = parse_args()
    prepared = prepare_source(args.source)
    if prepared.mode == "url":
        require_cookies_for_url_mode()
        prepared = fetch_question_overview(prepared)
        if args.answer_cap is not None and prepared.total_answers is not None:
            prepared.total_answers = min(prepared.total_answers, args.answer_cap)

    output_dir = resolve_output_dir(prepared, args.output_dir)

    jieba, SnowNLP, WordCloud = load_analysis_dependencies()
    stopwords = load_stopwords(args.stopwords)
    font_path = resolve_wordcloud_font(args.font_path)

    if prepared.mode == "url":
        output_dir.mkdir(parents=True, exist_ok=True)
        records = await fetch_and_save_answers(
            prepared=prepared,
            output_dir=output_dir,
            download_images=not args.no_images,
        )
    else:
        records = load_records_from_path(prepared.path)
        if not prepared.question_title and records:
            prepared.question_title = records[0].get("title") or prepared.path.name

    records = normalize_records(records)
    if not records:
        raise SystemExit("No answer records were available for analysis.")

    frequencies = tokenize_records(records, jieba, stopwords, args.min_word_length)
    score_sentiment(records, SnowNLP)
    summary = build_summary(records, frequencies, prepared, args.top_k)
    write_outputs(output_dir, summary, records, frequencies, WordCloud, font_path)

    print(f"Report bundle created at: {output_dir}")
    print(f"Markdown report: {output_dir / 'report.md'}")
    print(f"ECharts dashboard: {output_dir / 'analysis' / 'dashboard.html'}")


if __name__ == "__main__":
    asyncio.run(main())
