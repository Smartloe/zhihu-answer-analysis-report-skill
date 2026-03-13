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
    known_dates = sorted(date for date in timeline if date != "unknown")

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
        "author_count": len(authors),
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
        "date_range": (
            {"start": known_dates[0], "end": known_dates[-1]}
            if known_dates
            else None
        ),
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
    sentiment_labels = summary.get("sentiment_labels", {})
    question_title = summary.get("question_title") or "知乎回答分析"
    average_sentiment = summary.get("average_sentiment")
    average_sentiment_display = average_sentiment if average_sentiment is not None else "N/A"
    positive_count = sentiment_labels.get("positive", 0)
    neutral_count = sentiment_labels.get("neutral", 0)
    negative_count = sentiment_labels.get("negative", 0)
    author_count = summary.get("author_count") or len({record.get("author") or "Unknown" for record in records})
    leading_keyword = top_keywords[0]["word"] if top_keywords else "N/A"
    date_range = summary.get("date_range") or {}
    date_start = date_range.get("start")
    date_end = date_range.get("end")
    if date_start and date_end:
        date_range_display = f"{date_start} - {date_end}" if date_start != date_end else date_start
    elif date_start or date_end:
        date_range_display = date_start or date_end
    else:
        date_range_display = "未提供"
    bucket_order = ["0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0", "unknown"]
    dominant_bucket = max(
        [name for name in bucket_order if name != "unknown"],
        key=lambda name: sentiment_buckets.get(name, 0),
        default="unknown",
    )
    dominant_bucket_count = sentiment_buckets.get(dominant_bucket, 0)
    timeline_peak = max(timeline, key=lambda item: item.get("count", 0), default={"date": "N/A", "count": 0})
    timeline_peak_label = f"{timeline_peak.get('date', 'N/A')} / {timeline_peak.get('count', 0)} 条"

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
    scatter_count = len(scatter_points)

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
      --bg: #f6f1e7;
      --panel: #fffdf9;
      --panel-soft: rgba(255, 255, 255, 0.78);
      --ink: #1f2328;
      --muted: #665f56;
      --line: rgba(120, 95, 61, 0.16);
      --accent: #c96d3a;
      --accent-soft: rgba(201, 109, 58, 0.12);
      --teal: #287271;
      --blue: #4c7aaf;
      --positive: #2f855a;
      --neutral: #b08929;
      --negative: #9b3d3d;
      --shadow: 0 18px 48px rgba(67, 46, 22, 0.12);
    }}
    * {{
      box-sizing: border-box;
    }}
    body {{
      margin: 0;
      min-height: 100vh;
      padding: clamp(18px, 3vw, 32px);
      background:
        radial-gradient(circle at top left, rgba(255, 228, 181, 0.8), transparent 30%),
        radial-gradient(circle at top right, rgba(76, 122, 175, 0.18), transparent 24%),
        linear-gradient(180deg, #fbf6ed 0%, var(--bg) 100%);
      color: var(--ink);
      font-family: "Avenir Next", "PingFang SC", "Noto Sans SC", sans-serif;
      position: relative;
    }}
    body::before {{
      content: "";
      position: fixed;
      inset: 0;
      pointer-events: none;
      background-image:
        linear-gradient(rgba(108, 93, 72, 0.035) 1px, transparent 1px),
        linear-gradient(90deg, rgba(108, 93, 72, 0.035) 1px, transparent 1px);
      background-size: 24px 24px;
      mask-image: linear-gradient(180deg, rgba(0, 0, 0, 0.42), transparent 72%);
    }}
    .shell {{
      max-width: 1440px;
      margin: 0 auto;
      display: flex;
      flex-direction: column;
      gap: 18px;
    }}
    .hero {{
      display: flex;
      align-items: stretch;
      justify-content: space-between;
      gap: 20px;
      padding: clamp(22px, 3vw, 32px);
      background:
        linear-gradient(135deg, rgba(255, 244, 223, 0.96), rgba(255, 253, 249, 0.94)),
        var(--panel);
      border: 1px solid var(--line);
      border-radius: 28px;
      box-shadow: var(--shadow);
      overflow: hidden;
      position: relative;
      isolation: isolate;
    }}
    .hero::after {{
      content: "";
      position: absolute;
      right: -8%;
      bottom: -36%;
      width: 320px;
      height: 320px;
      border-radius: 999px;
      background: radial-gradient(circle, rgba(76, 122, 175, 0.22), transparent 70%);
      z-index: -1;
    }}
    .hero-copy {{
      flex: 1;
      min-width: 0;
    }}
    .hero-kicker {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 6px 12px;
      border-radius: 999px;
      background: var(--accent-soft);
      color: var(--accent);
      font-size: 12px;
      font-weight: 700;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }}
    h1 {{
      margin: 14px 0 10px;
      font-size: clamp(30px, 5vw, 44px);
      line-height: 1.08;
    }}
    .lead {{
      margin: 0;
      max-width: 64ch;
      color: var(--muted);
      font-size: 15px;
      line-height: 1.7;
    }}
    .hero-meta {{
      margin-top: 18px;
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
    }}
    .meta-chip {{
      min-width: 148px;
      padding: 12px 14px;
      border-radius: 18px;
      background: rgba(255, 255, 255, 0.74);
      border: 1px solid rgba(120, 95, 61, 0.1);
      box-shadow: 0 10px 24px rgba(67, 46, 22, 0.06);
      display: flex;
      flex-direction: column;
      gap: 4px;
    }}
    .meta-chip span {{
      color: var(--muted);
      font-size: 11px;
      font-weight: 700;
      letter-spacing: 0.06em;
      text-transform: uppercase;
    }}
    .meta-chip strong {{
      font-size: 16px;
      line-height: 1.25;
    }}
    .hero-note {{
      width: min(300px, 100%);
      padding: 20px 22px;
      border-radius: 24px;
      background: var(--panel-soft);
      border: 1px solid rgba(76, 122, 175, 0.18);
      box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.55);
      display: flex;
      flex-direction: column;
      justify-content: space-between;
      gap: 10px;
      backdrop-filter: blur(12px);
    }}
    .note-label,
    .stat-label,
    .panel-tag {{
      font-size: 12px;
      font-weight: 700;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: var(--accent);
    }}
    .note-value {{
      font-size: clamp(26px, 3.5vw, 34px);
      font-weight: 700;
      line-height: 1.1;
    }}
    .hero-breakdown {{
      display: grid;
      gap: 10px;
      margin-top: 8px;
    }}
    .breakdown-row {{
      display: grid;
      grid-template-columns: auto 1fr auto;
      align-items: center;
      gap: 10px;
      font-size: 13px;
      color: var(--muted);
    }}
    .breakdown-dot {{
      width: 10px;
      height: 10px;
      border-radius: 999px;
    }}
    .breakdown-dot.positive {{
      background: var(--positive);
    }}
    .breakdown-dot.neutral {{
      background: var(--neutral);
    }}
    .breakdown-dot.negative {{
      background: var(--negative);
    }}
    .breakdown-row strong {{
      color: var(--ink);
    }}
    .note-caption,
    .stat-note,
    .panel-desc {{
      color: var(--muted);
      font-size: 13px;
      line-height: 1.6;
    }}
    .stats {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 18px;
    }}
    .stat-card,
    .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 22px;
      box-shadow: var(--shadow);
    }}
    .stat-card {{
      padding: 18px 20px;
      display: flex;
      flex-direction: column;
      gap: 8px;
    }}
    .stat-card:nth-child(1) {{
      background: linear-gradient(180deg, rgba(201, 109, 58, 0.09), rgba(255, 253, 249, 0.96) 55%);
    }}
    .stat-card:nth-child(2) {{
      background: linear-gradient(180deg, rgba(76, 122, 175, 0.1), rgba(255, 253, 249, 0.96) 55%);
    }}
    .stat-card:nth-child(3) {{
      background: linear-gradient(180deg, rgba(40, 114, 113, 0.1), rgba(255, 253, 249, 0.96) 55%);
    }}
    .stat-card:nth-child(4) {{
      background: linear-gradient(180deg, rgba(176, 137, 41, 0.11), rgba(255, 253, 249, 0.96) 55%);
    }}
    .stat-value {{
      font-size: clamp(26px, 3.6vw, 38px);
      font-weight: 700;
      line-height: 1.05;
    }}
    .dashboard {{
      display: grid;
      grid-template-columns: repeat(12, minmax(0, 1fr));
      gap: 18px;
      align-items: stretch;
    }}
    .panel {{
      grid-column: span 6;
      padding: 18px;
      display: flex;
      flex-direction: column;
      min-width: 0;
      overflow: hidden;
    }}
    .panel-heat {{
      background: linear-gradient(180deg, rgba(201, 109, 58, 0.09), rgba(255, 253, 249, 0.98) 34%);
    }}
    .panel-tone {{
      background: linear-gradient(180deg, rgba(40, 114, 113, 0.1), rgba(255, 253, 249, 0.98) 34%);
    }}
    .panel-rhythm {{
      background: linear-gradient(180deg, rgba(76, 122, 175, 0.1), rgba(255, 253, 249, 0.98) 34%);
    }}
    .panel-map {{
      background: linear-gradient(180deg, rgba(107, 29, 29, 0.08), rgba(255, 253, 249, 0.98) 34%);
    }}
    .panel-wide {{
      grid-column: span 7;
    }}
    .panel-narrow {{
      grid-column: span 5;
    }}
    .panel-head {{
      display: flex;
      flex-direction: column;
      gap: 6px;
      margin-bottom: 14px;
    }}
    .panel-topline {{
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 12px;
    }}
    .panel-head h2 {{
      margin: 0;
      font-size: 20px;
      line-height: 1.25;
    }}
    .panel-metric {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-height: 34px;
      padding: 6px 12px;
      border-radius: 999px;
      background: rgba(255, 255, 255, 0.84);
      border: 1px solid rgba(120, 95, 61, 0.12);
      color: var(--ink);
      font-size: 12px;
      font-weight: 700;
      white-space: nowrap;
    }}
    .chart {{
      width: 100%;
      min-width: 0;
    }}
    .chart-lg {{
      height: 420px;
    }}
    .chart-md {{
      height: 360px;
    }}
    @media (max-width: 1180px) {{
      .stats {{
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }}
      .panel,
      .panel-wide,
      .panel-narrow {{
        grid-column: span 6;
      }}
    }}
    @media (max-width: 820px) {{
      body {{
        padding: 16px;
      }}
      .hero {{
        flex-direction: column;
      }}
      .hero-note {{
        width: 100%;
      }}
      .hero-meta {{
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }}
      .stats {{
        grid-template-columns: 1fr;
      }}
      .dashboard {{
        grid-template-columns: 1fr;
      }}
      .panel,
      .panel-wide,
      .panel-narrow {{
        grid-column: auto;
      }}
      .chart-lg,
      .chart-md {{
        height: 320px;
      }}
      .panel-topline {{
        flex-direction: column;
      }}
      .panel-metric {{
        white-space: normal;
      }}
    }}
    @media (max-width: 560px) {{
      .hero-meta {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <main class="shell">
    <section class="hero">
      <div class="hero-copy">
        <span class="hero-kicker">Zhihu Insight Dashboard</span>
        <h1>{question_title}</h1>
        <p class="lead">共分析 {summary.get("answer_count", 0)} 条回答，累计 {summary.get("total_characters", 0)} 字。看板默认采用双列卡片布局，在窄屏场景下自动切换为单列，避免图表被硬塞进同一行。</p>
        <div class="hero-meta">
          <article class="meta-chip">
            <span>时间跨度</span>
            <strong>{date_range_display}</strong>
          </article>
          <article class="meta-chip">
            <span>活跃作者</span>
            <strong>{author_count}</strong>
          </article>
          <article class="meta-chip">
            <span>领跑关键词</span>
            <strong>{leading_keyword}</strong>
          </article>
        </div>
      </div>
      <aside class="hero-note">
        <span class="note-label">情绪切片</span>
        <strong class="note-value">{average_sentiment_display}</strong>
        <span class="note-caption">平均情感分，越接近 1 越偏正向。</span>
        <div class="hero-breakdown">
          <div class="breakdown-row">
            <span class="breakdown-dot positive"></span>
            <span>正向</span>
            <strong>{positive_count}</strong>
          </div>
          <div class="breakdown-row">
            <span class="breakdown-dot neutral"></span>
            <span>中性</span>
            <strong>{neutral_count}</strong>
          </div>
          <div class="breakdown-row">
            <span class="breakdown-dot negative"></span>
            <span>负向</span>
            <strong>{negative_count}</strong>
          </div>
        </div>
      </aside>
    </section>

    <section class="stats">
      <article class="stat-card">
        <span class="stat-label">回答总数</span>
        <strong class="stat-value">{summary.get("answer_count", 0)}</strong>
        <span class="stat-note">累计字数 {summary.get("total_characters", 0)}</span>
      </article>
      <article class="stat-card">
        <span class="stat-label">活跃作者</span>
        <strong class="stat-value">{author_count}</strong>
        <span class="stat-note">报告正文会列出前 10 位高频作者</span>
      </article>
      <article class="stat-card">
        <span class="stat-label">平均字数</span>
        <strong class="stat-value">{summary.get("average_characters", 0)}</strong>
        <span class="stat-note">中位数字数 {summary.get("median_characters", 0)}</span>
      </article>
      <article class="stat-card">
        <span class="stat-label">平均情感分</span>
        <strong class="stat-value">{average_sentiment_display}</strong>
        <span class="stat-note">标准差 {summary.get("sentiment_stddev", "N/A")}</span>
      </article>
    </section>

    <section class="dashboard">
      <article class="panel panel-wide panel-heat">
        <div class="panel-head">
          <div class="panel-topline">
            <div>
              <span class="panel-tag">Topic Language</span>
              <h2>高频关键词 Top 20</h2>
            </div>
            <span class="panel-metric">Top 1: {leading_keyword}</span>
          </div>
          <p class="panel-desc">观察回答里的核心讨论主题与词频热度。</p>
        </div>
        <div id="keywords" class="chart chart-lg"></div>
      </article>

      <article class="panel panel-narrow panel-tone">
        <div class="panel-head">
          <div class="panel-topline">
            <div>
              <span class="panel-tag">Sentiment Curve</span>
              <h2>情感分布</h2>
            </div>
            <span class="panel-metric">主峰: {dominant_bucket} / {dominant_bucket_count} 条</span>
          </div>
          <p class="panel-desc">按分数区间观察回答整体偏正向还是偏负向。</p>
        </div>
        <div id="sentiment" class="chart chart-lg"></div>
      </article>

      <article class="panel panel-narrow panel-rhythm">
        <div class="panel-head">
          <div class="panel-topline">
            <div>
              <span class="panel-tag">Activity Rhythm</span>
              <h2>回答时间线</h2>
            </div>
            <span class="panel-metric">峰值: {timeline_peak_label}</span>
          </div>
          <p class="panel-desc">按日期查看回答发布节奏是否有集中爆发。</p>
        </div>
        <div id="timeline" class="chart chart-md"></div>
      </article>

      <article class="panel panel-wide panel-map">
        <div class="panel-head">
          <div class="panel-topline">
            <div>
              <span class="panel-tag">Engagement Map</span>
              <h2>字数 / 赞同 / 情感</h2>
            </div>
            <span class="panel-metric">样本点: {scatter_count}</span>
          </div>
          <p class="panel-desc">同时观察篇幅、赞同数与情感分之间的关系。</p>
        </div>
        <div id="scatter" class="chart chart-md"></div>
      </article>
    </section>
  </main>
  <script>
    const topKeywords = {json.dumps(top_keywords, ensure_ascii=False)};
    const sentimentBuckets = {json.dumps(sentiment_buckets, ensure_ascii=False)};
    const timeline = {json.dumps(timeline, ensure_ascii=False)};
    const scatterPoints = {json.dumps(scatter_points, ensure_ascii=False)};
    const bucketOrder = {json.dumps(bucket_order, ensure_ascii=False)};
    const axisLabelColor = '#615949';
    const splitLineColor = 'rgba(109, 94, 72, 0.16)';
    const tooltipTheme = {{
      backgroundColor: 'rgba(31, 26, 21, 0.94)',
      borderWidth: 0,
      padding: [10, 12],
      textStyle: {{ color: '#fff', fontSize: 12 }}
    }};

    const keywordChart = echarts.init(document.getElementById('keywords'));
    keywordChart.setOption({{
      animationDuration: 700,
      grid: {{ left: 48, right: 22, top: 16, bottom: 74 }},
      tooltip: {{ ...tooltipTheme, trigger: 'axis', axisPointer: {{ type: 'shadow' }} }},
      xAxis: {{
        type: 'category',
        data: topKeywords.map(item => item.word),
        axisLabel: {{ rotate: 34, color: axisLabelColor }},
        axisLine: {{ lineStyle: {{ color: '#cdbba5' }} }},
        axisTick: {{ show: false }}
      }},
      yAxis: {{
        type: 'value',
        axisLabel: {{ color: axisLabelColor }},
        splitLine: {{ lineStyle: {{ color: splitLineColor }} }}
      }},
      series: [{{
        type: 'bar',
        data: topKeywords.map(item => item.count),
        itemStyle: {{
          color: '#c96d3a',
          borderRadius: [10, 10, 0, 0],
          shadowBlur: 14,
          shadowColor: 'rgba(201, 109, 58, 0.18)'
        }},
        emphasis: {{ focus: 'series' }},
        barMaxWidth: 28
      }}]
    }});

    const sentimentChart = echarts.init(document.getElementById('sentiment'));
    sentimentChart.setOption({{
      animationDuration: 700,
      grid: {{ left: 42, right: 18, top: 16, bottom: 42 }},
      tooltip: {{ ...tooltipTheme, trigger: 'axis' }},
      xAxis: {{
        type: 'category',
        data: bucketOrder,
        axisLabel: {{ color: axisLabelColor }},
        axisLine: {{ lineStyle: {{ color: '#cdbba5' }} }},
        axisTick: {{ show: false }}
      }},
      yAxis: {{
        type: 'value',
        axisLabel: {{ color: axisLabelColor }},
        splitLine: {{ lineStyle: {{ color: splitLineColor }} }}
      }},
      series: [{{
        type: 'line',
        smooth: true,
        data: bucketOrder.map(name => sentimentBuckets[name] || 0),
        lineStyle: {{ color: '#287271', width: 3 }},
        areaStyle: {{ color: 'rgba(40, 114, 113, 0.18)' }},
        itemStyle: {{ color: '#287271' }},
        symbol: 'circle',
        symbolSize: 8
      }}]
    }});

    const timelineChart = echarts.init(document.getElementById('timeline'));
    timelineChart.setOption({{
      animationDuration: 700,
      grid: {{ left: 48, right: 18, top: 16, bottom: 74 }},
      tooltip: {{ ...tooltipTheme, trigger: 'axis', axisPointer: {{ type: 'shadow' }} }},
      xAxis: {{
        type: 'category',
        data: timeline.map(item => item.date),
        axisLabel: {{ rotate: 34, color: axisLabelColor }},
        axisLine: {{ lineStyle: {{ color: '#cdbba5' }} }},
        axisTick: {{ show: false }}
      }},
      yAxis: {{
        type: 'value',
        axisLabel: {{ color: axisLabelColor }},
        splitLine: {{ lineStyle: {{ color: splitLineColor }} }}
      }},
      series: [{{
        type: 'bar',
        data: timeline.map(item => item.count),
        itemStyle: {{
          color: '#4c7aaf',
          borderRadius: [10, 10, 0, 0],
          shadowBlur: 16,
          shadowColor: 'rgba(76, 122, 175, 0.18)'
        }},
        barMaxWidth: 26
      }}]
    }});

    const scatterChart = echarts.init(document.getElementById('scatter'));
    scatterChart.setOption({{
      animationDuration: 700,
      grid: {{ left: 54, right: 28, top: 20, bottom: 72 }},
      tooltip: {{
        ...tooltipTheme,
        formatter: params => {{
          const data = params.data;
          const title = data[3] || '未命名回答';
          return `${{title}}<br>字数: ${{data[0]}}<br>赞同: ${{data[1]}}<br>情感: ${{data[2]}}`;
        }}
      }},
      xAxis: {{
        type: 'value',
        name: '字数',
        nameTextStyle: {{ color: axisLabelColor }},
        axisLabel: {{ color: axisLabelColor }},
        splitLine: {{ lineStyle: {{ color: splitLineColor }} }}
      }},
      yAxis: {{
        type: 'value',
        name: '赞同数',
        nameTextStyle: {{ color: axisLabelColor }},
        axisLabel: {{ color: axisLabelColor }},
        splitLine: {{ lineStyle: {{ color: splitLineColor }} }}
      }},
      visualMap: {{
        min: 0,
        max: 1,
        dimension: 2,
        orient: 'horizontal',
        left: 'center',
        bottom: 0,
        textStyle: {{ color: axisLabelColor }},
        inRange: {{ color: ['#6b1d1d', '#d6c65b', '#276749'] }}
      }},
      series: [{{
        type: 'scatter',
        symbolSize: value => Math.min(52, Math.max(10, Math.sqrt(value[1] + 1) * 3)),
        itemStyle: {{
          borderColor: 'rgba(255, 255, 255, 0.9)',
          borderWidth: 1,
          opacity: 0.9
        }},
        data: scatterPoints
      }}]
    }});

    const charts = [keywordChart, sentimentChart, timelineChart, scatterChart];
    const syncFrameHeight = () => {{
      const frame = window.frameElement;
      if (!frame) {{
        return;
      }}
      try {{
        frame.style.height = `${{document.documentElement.scrollHeight + 24}}px`;
      }} catch (_error) {{
        // Ignore iframe sizing failures when opened directly.
      }}
    }};

    const resizeCharts = () => {{
      charts.forEach(chart => chart.resize());
      syncFrameHeight();
    }};

    window.addEventListener('load', syncFrameHeight);
    window.addEventListener('resize', resizeCharts);
    if ('ResizeObserver' in window) {{
      const observer = new ResizeObserver(() => syncFrameHeight());
      observer.observe(document.body);
    }}
    setTimeout(syncFrameHeight, 300);
    setTimeout(syncFrameHeight, 900);
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
            "style=\"width: 100%; min-height: 1720px; border: 1px solid #e5e7eb; border-radius: 16px; background: #ffffff;\" "
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
