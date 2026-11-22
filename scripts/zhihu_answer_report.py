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
import math
import os
import re
import statistics
import sys
from collections import Counter
from datetime import datetime
from html import escape, unescape
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import quote


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
from core.utils import extract_urls, sanitize_filename


QUESTION_RE = re.compile(
    r"https?://(?:www\.|m\.)?zhihu\.com/question/(\d+)(?:/answer/(\d+))?"
)
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


def _positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("必须是大于 0 的整数")
    return number


def _non_negative_int(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("必须是大于等于 0 的整数")
    return number


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
        type=_positive_int,
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
        type=_positive_int,
        default=30,
        help="Number of top keywords shown in tables and charts.",
    )
    parser.add_argument(
        "--min-word-length",
        type=_non_negative_int,
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
    parser.add_argument(
        "--lda-topics",
        type=_non_negative_int,
        default=6,
        help="Number of LDA topics to infer. Use 0 to disable LDA.",
    )
    parser.add_argument(
        "--lda-words",
        type=_positive_int,
        default=8,
        help="Top words to keep per LDA topic.",
    )
    parser.add_argument(
        "--lda-max-iter",
        type=_positive_int,
        default=15,
        help="Max iterations for the LDA optimizer.",
    )
    parser.add_argument(
        "--network-max-nodes",
        type=_non_negative_int,
        default=18,
        help="Maximum number of author nodes kept in the inferred author network. Use 0 to disable it.",
    )
    parser.add_argument(
        "--network-max-edges",
        type=_non_negative_int,
        default=28,
        help="Maximum number of author links kept in the inferred author network.",
    )
    return parser.parse_args()


def load_analysis_dependencies() -> tuple[Any, Any, Any, Any, Any]:
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

    try:
        from sklearn.decomposition import LatentDirichletAllocation
        from sklearn.feature_extraction.text import CountVectorizer
    except ImportError:
        LatentDirichletAllocation = None
        CountVectorizer = None
        missing.append("scikit-learn")

    if missing:
        raise SystemExit(
            "Missing analysis dependencies: "
            f"{', '.join(missing)}\n"
            f"Install them with:\npython3 -m pip install {' '.join(missing)}"
        )

    return jieba, SnowNLP, WordCloud, LatentDirichletAllocation, CountVectorizer


def prepare_source(source: str) -> PreparedSource:
    source = source.strip()
    candidate_path = Path(source).expanduser()
    if candidate_path.exists():
        return PreparedSource(mode="path", source=source, path=candidate_path.resolve())

    match = QUESTION_RE.match(source)
    if not match:
        # Accept pasted share text such as "如何评价… https://www.zhihu.com/question/123"
        embedded = [url for url in extract_urls(source) if QUESTION_RE.match(url)]
        if embedded:
            source = embedded[0]
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

    totals = (page.get("paging") or {}).get("totals")
    prepared.question_title = title
    prepared.total_answers = int(totals) if totals else None
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
    except (KeyError, IndexError, ValueError):
        rendered = f"[{item_date}] {title}"

    rendered = sanitize_filename(rendered, max_length=120)
    return f"{rendered} ({item_key})"


def extract_text_from_markdown(markdown: str | None) -> str:
    """Strip the generated scrape header, then flatten Markdown to plain text.

    Only a header that starts with ``# title`` and contains ``> **Author`` /
    ``> **Source`` / ``> **Date`` metadata is removed. A bare ``---`` inside the
    answer body must never truncate the text.
    """
    markdown = "" if markdown is None else str(markdown)
    separator_index = markdown.find("\n---\n")
    if separator_index != -1:
        header = markdown[:separator_index]
        has_heading = header.lstrip().startswith("# ")
        has_metadata = bool(
            re.search(r"^> \*\*(?:Author|Source|Date)\b", header, re.M)
        )
        if has_heading and has_metadata:
            markdown = markdown[separator_index + len("\n---\n"):]

    body = markdown
    body = re.sub(r"```.*?```", " ", body, flags=re.S)
    body = re.sub(r"`([^`]+)`", r"\1", body)
    body = re.sub(r"!\[[^\]]*\]\([^)]+\)", " ", body)
    body = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1", body)
    body = re.sub(r"<[^>]+>", " ", body)
    body = re.sub(r"^>\s*.*$", " ", body, flags=re.M)
    body = re.sub(r"^#+\s*", " ", body, flags=re.M)
    body = re.sub(r"^\s*[-*]\s+", " ", body, flags=re.M)
    body = re.sub(r"^\s*---+\s*$", " ", body, flags=re.M)
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
        if not extra_path.is_file():
            raise SystemExit(f"额外停用词文件不存在：{extra_path}")
        stopwords.update(
            line.strip()
            for line in extra_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    return stopwords


def resolve_wordcloud_font(font_override: str | None) -> str:
    candidates: list[Path] = []
    env_font = os.environ.get("ZHIHU_FONT_PATH", "").strip()
    if font_override:
        override_path = Path(font_override).expanduser()
        if not override_path.is_file():
            raise SystemExit(f"指定的字体文件不存在：{override_path}")
        candidates.append(override_path)
    if env_font:
        candidates.append(Path(env_font).expanduser())

    candidates.extend(
        Path(candidate)
        for candidate in [
            # macOS
            "/System/Library/Fonts/Supplemental/Songti.ttc",
            "/System/Library/Fonts/STHeiti Medium.ttc",
            "/System/Library/Fonts/STHeiti Light.ttc",
            "/System/Library/Fonts/PingFang.ttc",
            "/System/Library/AssetsV2/com_apple_MobileAsset_Font8/259e8f5a322e8dae602d51ac00aefb3d6b05c224.asset/AssetData/SimSong.ttc",
            # Linux
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
            "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
            # Windows
            "C:/Windows/Fonts/msyh.ttc",
            "C:/Windows/Fonts/simhei.ttf",
            "C:/Windows/Fonts/simsun.ttc",
        ]
    )

    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())

    # Last resort on Linux: ask fontconfig for any installed CJK font.
    if sys.platform.startswith("linux"):
        try:
            import subprocess

            output = subprocess.run(
                ["fc-match", "-f", "%{file}", "sans-serif:lang=zh-cn"],
                capture_output=True,
                text=True,
                timeout=3,
            ).stdout.strip()
            if output and Path(output).exists():
                return output
        except Exception:
            pass

    raise SystemExit(
        "未找到可用的中文字体，无法生成词云。请通过 --font-path 指定字体文件，"
        "或设置环境变量 ZHIHU_FONT_PATH。"
    )


async def fetch_and_save_answers(
    prepared: PreparedSource,
    output_dir: Path,
    download_images: bool,
) -> list[dict[str, Any]]:
    if not prepared.question_url:
        raise SystemExit("Question URL is required in fetch mode.")

    target_answers = prepared.total_answers
    if target_answers is None:
        target_answers = 200
        print("⚠️ 未能获取回答总数，默认最多抓取 200 条；可用 --answer-cap 调整。")
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
    db_failures = 0
    today = datetime.now().strftime("%Y-%m-%d")

    try:
        for item in items:
            title = sanitize_filename(str(item.get("title") or "Untitled"), max_length=80)
            author = sanitize_filename(str(item.get("author") or "Unknown"), max_length=40)
            item_date = str(item.get("date") or today)
            item_key = sanitize_filename(
                f"{item.get('type') or 'answer'}-{item.get('id') or 'unknown'}",
                max_length=80,
            )
            folder_name = build_output_folder_name(item_date, title, author, item_key)
            folder = content_root / folder_name
            folder.mkdir(parents=True, exist_ok=True)

            img_map: dict[str, str] = {}
            if download_images:
                img_urls = ZhihuConverter.extract_image_urls(item.get("html") or "")
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
            if not db.save_article(item, full_markdown):
                db_failures += 1

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

    if db_failures:
        print(f"⚠️ {db_failures} 条回答写入 raw/zhihu.db 失败，Markdown 与报告不受影响。")

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
    if path.is_file() and path.suffix.lower() in {".jsonl", ".json"}:
        if path.suffix.lower() == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, list):
                return [item for item in payload if isinstance(item, dict)]
        else:
            records = []
            for line_number, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), start=1
            ):
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise SystemExit(
                        f"answers 数据集第 {line_number} 行不是合法 JSON：{exc}"
                    ) from exc
                if isinstance(item, dict):
                    records.append(item)
            return records

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


def _coerce_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _coerce_int(value: Any) -> int | None:
    """Parse ints from numbers and human counters such as ``1.2万`` / ``1,234``."""
    if value is None or isinstance(value, bool):
        return None if value is None else int(value)
    if isinstance(value, (int, float)):
        return int(value)

    text = _coerce_text(value).strip().replace(",", "").replace("，", "").replace(" ", "")
    if not text:
        return None
    multiplier = 1
    if text.endswith("万"):
        multiplier = 10000
        text = text[:-1]
    elif text.endswith(("千", "k", "K")):
        multiplier = 1000
        text = text[:-1]
    match = re.match(r"-?\d+(?:\.\d+)?", text)
    if not match:
        return None
    try:
        return int(float(match.group(0)) * multiplier)
    except ValueError:
        return None


def normalize_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise SystemExit(f"答案数据集第 {index + 1} 条不是 JSON 对象。")
        copied = dict(record)
        for key in ("id", "title", "author", "url", "date", "type"):
            copied[key] = _coerce_text(copied.get(key))
        copied["author"] = copied["author"].strip() or "Unknown"
        copied["upvotes"] = _coerce_int(copied.get("upvotes"))

        markdown_path = _coerce_text(copied.get("markdown_path")).strip()
        copied["markdown_path"] = (
            str(Path(markdown_path).expanduser().resolve()) if markdown_path else ""
        )
        copied["text"] = extract_text_from_markdown(copied.get("text"))
        normalized.append(copied)
    return normalized


def tokenize_text(
    text: str,
    jieba: Any,
    stopwords: set[str],
    min_word_length: int,
) -> list[str]:
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
    return tokens


def tokenize_records(
    records: list[dict[str, Any]],
    jieba: Any,
    stopwords: set[str],
    min_word_length: int,
) -> Counter:
    frequencies: Counter = Counter()
    for record in records:
        text = record.get("text", "").strip()
        tokens = tokenize_text(text, jieba, stopwords, min_word_length)
        record["char_count"] = len(text)
        record["token_count"] = len(tokens)
        frequencies.update(tokens)

    return frequencies


def run_lda_topics(
    records: list[dict[str, Any]],
    jieba: Any,
    stopwords: set[str],
    min_word_length: int,
    topic_count: int,
    top_words: int,
    max_iter: int,
    LDA: Any,
    CountVectorizer: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str | None]:
    if topic_count <= 0:
        return [], [], "已关闭 LDA 主题聚类。"

    docs: list[list[str]] = []
    doc_indices: list[int] = []
    for idx, record in enumerate(records):
        tokens = tokenize_text(record.get("text", ""), jieba, stopwords, min_word_length)
        if tokens:
            docs.append(tokens)
            doc_indices.append(idx)

    minimum_docs = max(8, topic_count * 2)
    if len(docs) < minimum_docs:
        return [], [], (
            f"有效回答仅 {len(docs)} 条，至少需要 {minimum_docs} 条才能进行 LDA 主题聚类。"
        )

    min_df = 2 if len(docs) >= 20 else 1
    vectorizer = CountVectorizer(
        tokenizer=lambda tokens: tokens,
        preprocessor=lambda tokens: tokens,
        token_pattern=None,
        min_df=min_df,
        max_df=0.95,
    )
    try:
        doc_term = vectorizer.fit_transform(docs)
    except ValueError:
        # CountVectorizer raises when max_df/min_df prune every term.
        return [], [], "有效词汇量不足（词项被文档频率过滤），无法形成稳定的主题聚类。"
    if doc_term.shape[1] < 6:
        return [], [], "有效词汇量不足，无法形成稳定的主题聚类。"

    max_topics = min(topic_count, doc_term.shape[0], doc_term.shape[1])
    if max_topics < 2:
        return [], [], "可用于聚类的主题数量不足（少于 2 个）。"
    if max_topics != topic_count:
        topic_count = max_topics

    lda = LDA(
        n_components=topic_count,
        max_iter=max_iter,
        learning_method="batch",
        random_state=42,
    )
    doc_topics = lda.fit_transform(doc_term)
    topic_assignments = doc_topics.argmax(axis=1)
    topic_scores = doc_topics.max(axis=1)

    for index, topic_id, score in zip(doc_indices, topic_assignments, topic_scores):
        records[index]["topic_id"] = int(topic_id) + 1
        records[index]["topic_score"] = round(float(score), 4)

    feature_names = vectorizer.get_feature_names_out()
    topic_counts = Counter(int(topic_id) for topic_id in topic_assignments)
    topic_summaries: list[dict[str, Any]] = []
    topic_distribution: list[dict[str, Any]] = []
    total_docs = sum(topic_counts.values()) or 1

    for idx in range(topic_count):
        top_idx = lda.components_[idx].argsort()[::-1][:top_words]
        words = [feature_names[word_idx] for word_idx in top_idx]
        count = topic_counts.get(idx, 0)
        share = round(count / total_docs, 4)
        topic_summaries.append(
            {
                "topic_id": idx + 1,
                "top_words": words,
                "doc_count": count,
                "doc_share": share,
            }
        )
        topic_distribution.append(
            {
                "topic": f"Topic {idx + 1}",
                "count": count,
                "keywords": " ".join(words[:3]),
            }
        )

    return topic_summaries, topic_distribution, None


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


def build_author_network(
    records: list[dict[str, Any]],
    jieba: Any,
    stopwords: set[str],
    min_word_length: int,
    lda_topics: list[dict[str, Any]] | None,
    max_nodes: int,
    max_edges: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], str | None]:
    if max_nodes <= 0 or max_edges <= 0:
        return {"nodes": [], "links": [], "categories": []}, [], "已关闭作者关联网络。"

    topic_word_map = {
        int(topic.get("topic_id")): list(topic.get("top_words") or [])
        for topic in (lda_topics or [])
        if topic.get("topic_id") is not None
    }
    author_profiles: dict[str, dict[str, Any]] = {}

    for record in records:
        author = _coerce_text(record.get("author")).strip() or "Unknown"
        profile = author_profiles.setdefault(
            author,
            {
                "author": author,
                "answers": 0,
                "upvotes": 0,
                "char_total": 0,
                "sentiments": [],
                "topics": Counter(),
                "keywords": Counter(),
            },
        )
        profile["answers"] += 1
        profile["upvotes"] += _coerce_int(record.get("upvotes")) or 0
        profile["char_total"] += _coerce_int(record.get("char_count")) or 0

        sentiment = record.get("sentiment")
        if sentiment is not None:
            profile["sentiments"].append(float(sentiment))

        topic_id = record.get("topic_id")
        if topic_id is not None:
            profile["topics"][int(topic_id)] += 1

        tokens = tokenize_text(record.get("text", ""), jieba, stopwords, min_word_length)
        if tokens:
            profile["keywords"].update(tokens)

    if len(author_profiles) < 3:
        return (
            {"nodes": [], "links": [], "categories": []},
            [],
            f"仅检测到 {len(author_profiles)} 位作者，至少需要 3 位才能构建作者关联网络。",
        )

    ranked_profiles = sorted(
        author_profiles.values(),
        key=lambda item: (item["answers"], item["upvotes"], item["char_total"]),
        reverse=True,
    )[:max_nodes]

    if len(ranked_profiles) < 3:
        return (
            {"nodes": [], "links": [], "categories": []},
            [],
            f"保留的作者仅 {len(ranked_profiles)} 位，至少需要 3 位才能构建作者关联网络。",
        )

    categories: list[dict[str, str]] = []
    category_map: dict[str, int] = {}
    max_answers = max((profile["answers"] for profile in ranked_profiles), default=1)

    for profile in ranked_profiles:
        dominant_topic_id = None
        if profile["topics"]:
            dominant_topic_id = int(profile["topics"].most_common(1)[0][0])
        profile["dominant_topic_id"] = dominant_topic_id
        top_keywords = [word for word, _ in profile["keywords"].most_common(12)]
        profile["top_keywords"] = top_keywords
        profile["keyword_set"] = set(top_keywords)
        profile["avg_sentiment"] = (
            round(sum(profile["sentiments"]) / len(profile["sentiments"]), 4)
            if profile["sentiments"]
            else None
        )
        topic_words = topic_word_map.get(dominant_topic_id, [])
        profile["topic_keywords_display"] = " / ".join(topic_words[:3]) if topic_words else "未聚类"
        profile["keywords_display"] = " / ".join(top_keywords[:5]) if top_keywords else "未提取关键词"

        category_name = f"Topic {dominant_topic_id}" if dominant_topic_id else "未聚类"
        if category_name not in category_map:
            category_map[category_name] = len(categories)
            categories.append({"name": category_name})
        profile["category_index"] = category_map[category_name]

        answer_ratio = profile["answers"] / max_answers if max_answers else 0
        profile["symbol_size"] = round(18 + math.log1p(profile["upvotes"]) * 3.6 + answer_ratio * 10, 2)

    links: list[dict[str, Any]] = []
    for idx, source in enumerate(ranked_profiles):
        for target in ranked_profiles[idx + 1:]:
            shared_keywords = sorted(source["keyword_set"] & target["keyword_set"])
            union_keywords = source["keyword_set"] | target["keyword_set"]
            keyword_score = (len(shared_keywords) / len(union_keywords)) if union_keywords else 0.0
            same_topic = (
                source["dominant_topic_id"] is not None
                and source["dominant_topic_id"] == target["dominant_topic_id"]
            )
            sentiment_bonus = 0.0
            if source["avg_sentiment"] is not None and target["avg_sentiment"] is not None:
                sentiment_gap = abs(source["avg_sentiment"] - target["avg_sentiment"])
                sentiment_bonus = max(0.0, 0.08 - sentiment_gap * 0.08)

            score = keyword_score + (0.12 if same_topic else 0.0) + sentiment_bonus
            if score < 0.2:
                continue
            if not shared_keywords and not same_topic:
                continue

            topic_name = f"Topic {source['dominant_topic_id']}" if same_topic else ""
            relation_parts: list[str] = []
            if same_topic:
                relation_parts.append(topic_name)
            if shared_keywords:
                relation_parts.append("共享关键词: " + " / ".join(shared_keywords[:4]))

            links.append(
                {
                    "source": source["author"],
                    "target": target["author"],
                    "value": round(score, 4),
                    "shared_keywords": " / ".join(shared_keywords[:4]),
                    "topic": topic_name,
                    "relation": "；".join(relation_parts) if relation_parts else "文本相似",
                    "lineStyle": {
                        "width": round(1.2 + score * 8, 2),
                        "opacity": round(min(0.72, 0.22 + score), 2),
                    },
                }
            )

    links.sort(
        key=lambda item: (
            item["value"],
            len(item.get("shared_keywords", "").split(" / ")) if item.get("shared_keywords") else 0,
        ),
        reverse=True,
    )
    links = links[:max_edges]

    if not links:
        return (
            {"nodes": [], "links": [], "categories": []},
            [],
            "作者已识别，但回答之间的关键词与主题重合度不足以形成可靠关联。",
        )

    connected_authors = {
        author
        for link in links
        for author in (link["source"], link["target"])
    }
    ranked_profiles = [profile for profile in ranked_profiles if profile["author"] in connected_authors]

    nodes = []
    for profile in ranked_profiles:
        topic_name = f"Topic {profile['dominant_topic_id']}" if profile["dominant_topic_id"] else "未聚类"
        sentiment_value = profile["avg_sentiment"] if profile["avg_sentiment"] is not None else "N/A"
        nodes.append(
            {
                "id": profile["author"],
                "name": profile["author"],
                "category": profile["category_index"],
                "symbolSize": profile["symbol_size"],
                "value": profile["upvotes"],
                "answers": profile["answers"],
                "upvotes": profile["upvotes"],
                "avg_sentiment": sentiment_value,
                "topic": topic_name,
                "topic_keywords": profile["topic_keywords_display"],
                "keywords": profile["keywords_display"],
            }
        )

    highlights = [
        {
            "source": link["source"],
            "target": link["target"],
            "score": link["value"],
            "relation": link["relation"],
            "shared_keywords": link["shared_keywords"] or "同主题",
        }
        for link in links[:6]
    ]

    note = (
        "作者关系图基于回答主题、关键词重合度与情感接近度推断，"
        "不代表知乎真实关注、评论或私下社交关系。"
    )
    return {"nodes": nodes, "links": links, "categories": categories}, highlights, note


def build_summary(
    records: list[dict[str, Any]],
    frequencies: Counter,
    prepared: PreparedSource,
    top_k: int,
    lda_topics: list[dict[str, Any]] | None = None,
    lda_distribution: list[dict[str, Any]] | None = None,
    lda_note: str | None = None,
    author_network: dict[str, Any] | None = None,
    author_network_highlights: list[dict[str, Any]] | None = None,
    author_network_note: str | None = None,
) -> dict[str, Any]:
    sentiments = [record["sentiment"] for record in records if record.get("sentiment") is not None]
    char_counts = [record.get("char_count", 0) for record in records]
    upvote_records = [record for record in records if isinstance(record.get("upvotes"), (int, float))]

    label_counts = Counter(sentiment_label(record.get("sentiment")) for record in records)
    bucket_counts = Counter(sentiment_bucket(record.get("sentiment")) for record in records)
    timeline = Counter(_coerce_text(record.get("date")) or "unknown" for record in records)
    authors = Counter(_coerce_text(record.get("author")) or "Unknown" for record in records)
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
        "requested_answers": prepared.total_answers,
        "complete": prepared.total_answers is None or len(records) >= prepared.total_answers,
        "author_count": len(authors),
        "total_characters": sum(char_counts),
        "average_characters": round(sum(char_counts) / len(char_counts), 2) if char_counts else 0,
        "median_characters": statistics.median(char_counts) if char_counts else 0,
        "max_characters": max(char_counts) if char_counts else 0,
        "min_characters": min(char_counts) if char_counts else 0,
        "average_sentiment": round(sum(sentiments) / len(sentiments), 4) if sentiments else None,
        "sentiment_stddev": round(statistics.pstdev(sentiments), 4) if len(sentiments) > 1 else 0,
        "sentiment_labels": dict(label_counts),
        "sentiment_buckets": dict(bucket_counts),
        "top_keywords": [
            {"word": word, "count": count}
            for word, count in frequencies.most_common(top_k)
        ],
        "lda_topics": lda_topics or [],
        "lda_topic_distribution": lda_distribution or [],
        "lda_note": lda_note,
        "author_network": author_network or {"nodes": [], "links": [], "categories": []},
        "author_network_highlights": author_network_highlights or [],
        "author_network_note": author_network_note,
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
        raise SystemExit("没有可用于词云的关键词，请检查回答文本或停用词配置。")

    palette = ["#b8402a", "#1f5a4a", "#35597c", "#a9761c", "#7a3e5d"]

    def color_func(word: str, *args: Any, **kwargs: Any) -> str:
        # Deterministic colour per word keeps the image stable across runs.
        return palette[sum(ord(char) for char in word) % len(palette)]

    cloud = WordCloud(
        width=1800,
        height=1050,
        background_color="#fbf7ee",
        font_path=font_path,
        max_words=120,
        collocations=False,
        prefer_horizontal=1.0,
        relative_scaling=0.4,
        max_font_size=170,
        min_font_size=14,
        margin=16,
        color_func=color_func,
    )
    cloud.generate_from_frequencies(top_frequencies)
    cloud.to_file(str(output_path))


def create_placeholder_wordcloud(output_path: Path, font_path: str, message: str) -> bool:
    """Write a readable placeholder so a report never ships with a broken image."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception:
        return False

    image = Image.new("RGB", (1800, 1050), "#fbf7ee")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype(font_path, 54)
    except Exception:
        font = ImageFont.load_default()

    text = f"词云暂不可用\n{message}"
    try:
        bbox = draw.multiline_textbbox((0, 0), text, font=font, spacing=20, align="center")
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        position = ((image.width - width) / 2, (image.height - height) / 2)
        draw.multiline_text(
            position, text, font=font, fill="#b8402a", spacing=20, align="center"
        )
        image.save(output_path)
        return True
    except Exception:
        return False


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    lines = [json.dumps(record, ensure_ascii=False) for record in records]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _html_text(value: Any) -> str:
    """Escape arbitrary text before inserting it into the dashboard markup."""
    return escape(str(value if value is not None else ""), quote=True)


def _json_for_script(payload: Any) -> str:
    """Serialize data for an inline <script type="application/json"> block."""
    return json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")


def _dashboard_table(caption: str, headers: list[str], rows: list[list[Any]]) -> str:
    if not rows:
        return ""
    head = "".join(f"<th>{_html_text(item)}</th>" for item in headers)
    body = []
    for row in rows:
        cells = []
        for index, cell in enumerate(row):
            css = ' class="num"' if index > 0 else ""
            cells.append(f"<td{css}>{_html_text(cell)}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return (
        f'<details class="data-table"><summary>{_html_text(caption)}</summary>'
        '<div class="table-wrap"><table><thead><tr>'
        f"{head}</tr></thead><tbody>{''.join(body)}</tbody></table></div></details>"
    )


def _dashboard_panel(
    kicker: str,
    title: str,
    description: str,
    metric: str | None,
    body: str,
    column: str = "col-6",
) -> str:
    metric_html = (
        f'<span class="panel-metric">{_html_text(metric)}</span>' if metric else ""
    )
    return f"""
      <article class="panel {column}">
        <header class="panel-head">
          <div class="panel-kicker"><span>{_html_text(kicker)}</span>{metric_html}</div>
          <h3>{_html_text(title)}</h3>
          <p class="panel-desc">{_html_text(description)}</p>
        </header>
        {body}
      </article>
"""


def _dashboard_section(index: str, title: str, description: str, body: str) -> str:
    return f"""
    <section class="section">
      <div class="section-head">
        <span class="section-index">{_html_text(index)}</span>
        <h2>{_html_text(title)}</h2>
        <p>{_html_text(description)}</p>
      </div>
      {body}
    </section>
"""


def write_dashboard(path: Path, summary: dict[str, Any], records: list[dict[str, Any]]) -> None:
    top_keywords = summary.get("top_keywords", [])
    timeline = summary.get("timeline", [])
    sentiment_buckets = summary.get("sentiment_buckets", {})
    sentiment_labels = summary.get("sentiment_labels", {})
    lda_topics = summary.get("lda_topics") or []
    lda_distribution = summary.get("lda_topic_distribution") or []
    lda_note = summary.get("lda_note")
    author_network = summary.get("author_network") or {}
    network_nodes = author_network.get("nodes") or []
    network_links = author_network.get("links") or []
    network_highlights = summary.get("author_network_highlights") or []
    network_note = summary.get("author_network_note") or (
        "作者关系图基于回答主题、关键词重合度与情感接近度推断，不代表知乎真实关注、评论或私下社交关系。"
    )

    question_title = summary.get("question_title") or "知乎回答分析"
    question_url = summary.get("question_url")
    question_id = summary.get("question_id")
    generated_at = summary.get("generated_at") or ""
    answer_count = summary.get("answer_count", 0)
    author_count = summary.get("author_count") or len(
        {record.get("author") or "Unknown" for record in records}
    )
    total_characters = summary.get("total_characters", 0)
    average_characters = summary.get("average_characters", 0)
    median_characters = summary.get("median_characters", 0)
    average_sentiment = summary.get("average_sentiment")
    average_sentiment_display = (
        average_sentiment if average_sentiment is not None else "N/A"
    )
    positive_count = sentiment_labels.get("positive", 0)
    neutral_count = sentiment_labels.get("neutral", 0)
    negative_count = sentiment_labels.get("negative", 0)

    date_range = summary.get("date_range") or {}
    date_start = date_range.get("start")
    date_end = date_range.get("end")
    if date_start and date_end:
        date_range_display = (
            f"{date_start} — {date_end}" if date_start != date_end else date_start
        )
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
    sentiment_metric = (
        f"主峰 · {dominant_bucket.replace('-', '–')} / {dominant_bucket_count} 条"
        if dominant_bucket_count
        else "暂无情感样本"
    )
    leading_keyword = top_keywords[0]["word"] if top_keywords else "N/A"
    timeline_peak = max(
        timeline, key=lambda item: item.get("count", 0), default={"date": "N/A", "count": 0}
    )
    timeline_metric = (
        f"峰值 · {timeline_peak.get('date', 'N/A')} / {timeline_peak.get('count', 0)} 条"
        if timeline_peak.get("count")
        else "暂无时间数据"
    )

    scatter_points = []
    for record in records:
        if record.get("sentiment") is None:
            continue
        scatter_points.append(
            [
                record.get("char_count", 0),
                record.get("upvotes") or 0,
                record.get("sentiment"),
                record.get("title", ""),
                record.get("author", ""),
            ]
        )

    findings = build_key_findings(summary)
    notes_html = "".join(f"<li>{_html_text(item)}</li>" for item in findings)

    keyword_rows = [
        [item.get("word", ""), item.get("count", 0)]
        for item in top_keywords[:20]
    ]
    sentiment_rows = [
        [
            "未知" if bucket == "unknown" else bucket.replace("-", "–"),
            sentiment_buckets.get(bucket, 0),
        ]
        for bucket in bucket_order
    ]
    timeline_rows = [
        [item.get("date", "未知"), item.get("count", 0)] for item in timeline
    ]

    keywords_body = '<div id="keywords" class="chart chart-tall"></div>'
    if not top_keywords:
        keywords_body = (
            '<div class="empty-state"><strong>暂无可用关键词。</strong> '
            "样本过短或全部命中停用词时，词频图会自动留空。</div>"
        )
    keywords_body += _dashboard_table(
        "关键词数据表", ["关键词", "出现次数"], keyword_rows
    )

    sentiment_body = '<div id="sentiment" class="chart"></div>'
    sentiment_body += _dashboard_table(
        "情感分桶数据表", ["区间", "回答数"], sentiment_rows
    )

    timeline_body = (
        '<div id="timeline" class="chart"></div>'
        if timeline
        else (
            '<div id="timeline" class="chart" data-empty="1"></div>'
            '<div class="empty-state"><strong>暂无发布时间。</strong> '
            "样本缺少日期字段时，时间线会自动留空。</div>"
        )
    )
    timeline_body += _dashboard_table(
        "时间线数据表", ["日期", "回答数"], timeline_rows
    )

    if scatter_points:
        scatter_body = '<div id="scatter" class="chart chart-tall"></div>'
    else:
        scatter_body = (
            '<div id="scatter" class="chart chart-tall" data-empty="1"></div>'
            '<div class="empty-state"><strong>暂无情感分样本。</strong> '
            "缺少可分析文本时，互动散点图会自动留空。</div>"
        )

    keywords_panel = _dashboard_panel(
        "Topic Language",
        "高频关键词 Top 20",
        "横向条形按出现频次排序，直接标注数值，避免长词标签旋转后互相遮挡。",
        f"Top 1 · {leading_keyword}",
        keywords_body,
        "col-7",
    )
    sentiment_panel = _dashboard_panel(
        "Sentiment Distribution",
        "情感分分布",
        "按 SnowNLP 分值区间统计回答数量，颜色从负向过渡到正向。",
        sentiment_metric,
        sentiment_body,
        "col-5",
    )
    timeline_panel = _dashboard_panel(
        "Activity Rhythm",
        "回答时间线",
        "按发布日期观察回答是否集中在同一天爆发。",
        timeline_metric,
        timeline_body,
        "col-5",
    )
    upvote_available = any((point[1] or 0) > 0 for point in scatter_points)
    scatter_description = (
        "每个点是一条回答：横轴为字数，纵轴为赞同数，颜色表示情感倾向。"
        if upvote_available
        else "每个点是一条回答：横轴为字数，颜色表示情感倾向。当前数据源不含赞同数，纵轴统一为 0。"
    )
    scatter_panel = _dashboard_panel(
        "Engagement Map",
        "篇幅 × 赞同 × 情感",
        scatter_description,
        f"样本点 · {len(scatter_points)}",
        scatter_body,
        "col-7",
    )

    topic_cards = []
    for topic in lda_topics:
        words = " · ".join(topic.get("top_words") or []) or "未提取关键词"
        share = float(topic.get("doc_share") or 0) * 100
        topic_cards.append(
            f"""
          <div class="topic-card">
            <div class="topic-title">Topic {_html_text(topic.get('topic_id', ''))}</div>
            <div class="topic-words">{_html_text(words)}</div>
            <div class="topic-meta">覆盖 {_html_text(topic.get('doc_count', 0))} 条 · 占比 {share:.1f}%</div>
          </div>
"""
        )

    if lda_topics:
        topics_body = (
            '<div id="topics" class="chart chart-tall"></div>'
            f'<div class="topic-grid">{"".join(topic_cards)}</div>'
        )
    else:
        note = lda_note or "未启用 LDA 或数据不足，未生成主题聚类结果。"
        topics_body = (
            f'<div class="empty-state"><strong>本次未生成主题聚类。</strong> '
            f"{_html_text(note)}</div>"
        )

    topics_panel = _dashboard_panel(
        "Latent Dirichlet Allocation",
        "LDA 主题聚类",
        "从回答文本中归纳话题簇；样本不足时自动跳过统计推断。",
        f"主题数 · {len(lda_topics)}" if lda_topics else "已跳过",
        topics_body,
        "col-12",
    )

    network_cards = []
    for link in network_highlights:
        network_cards.append(
            f"""
          <div class="network-card">
            <div class="network-pair">{_html_text(link.get('source', 'Unknown'))} ↔ {_html_text(link.get('target', 'Unknown'))}</div>
            <div class="network-score">关联强度 {_html_text(link.get('score', 0))}</div>
            <div class="network-reason">{_html_text(link.get('relation') or '文本相似')}</div>
          </div>
"""
        )

    if network_nodes and network_links:
        network_body = '<div id="author-network" class="chart chart-network"></div>'
        network_body += f'<div class="network-grid">{"".join(network_cards)}</div>'
        network_metric = f"节点 {len(network_nodes)} · 连线 {len(network_links)}"
    else:
        note = network_note or "当前样本未检测到足够强的作者关联。"
        network_body = (
            f'<div class="empty-state"><strong>本次未生成作者网络。</strong> '
            f"{_html_text(note)}</div>"
        )
        network_metric = "已跳过"

    network_panel = _dashboard_panel(
        "Inferred Author Network",
        "作者关联网络",
        "基于主题、关键词与情感接近度推断；样本不足时自动跳过。",
        network_metric,
        network_body,
        "col-12",
    )

    payload = {
        "topKeywords": top_keywords,
        "sentimentBuckets": sentiment_buckets,
        "bucketOrder": bucket_order,
        "timeline": timeline,
        "scatter": scatter_points,
        "ldaDistribution": lda_distribution,
        "authorNetwork": author_network,
    }

    css_path = SKILL_ROOT / "assets" / "dashboard.css"
    js_path = SKILL_ROOT / "assets" / "dashboard.js"
    echarts_path = SKILL_ROOT / "assets" / "echarts.min.js"
    dashboard_css = css_path.read_text(encoding="utf-8") if css_path.exists() else ""
    dashboard_js = js_path.read_text(encoding="utf-8") if js_path.exists() else ""
    if echarts_path.exists():
        # Vendored copy keeps the generated dashboard fully offline-capable.
        echarts_tag = f"<script>\n{echarts_path.read_text(encoding='utf-8')}\n  </script>"
    else:
        echarts_tag = (
            '<script src="https://cdn.jsdelivr.net/npm/echarts@5.5.1/dist/echarts.min.js"></script>'
        )

    if question_url:
        source_html = (
            f'<a class="source-link" href="{_html_text(question_url)}" '
            'target="_blank" rel="noreferrer">查看原问题 ↗</a>'
        )
    else:
        source_html = f'<span class="source-link">{_html_text(summary.get("source") or "本地数据")}</span>'

    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="color-scheme" content="light dark">
  <script>
    (function () {{
      try {{
        var saved = window.localStorage.getItem("zhihu-report-theme");
        if (saved === "dark" || saved === "light") {{
          document.documentElement.setAttribute("data-theme", saved);
        }}
      }} catch (error) {{}}
    }})();
  </script>
  <title>{_html_text(question_title)} · 知乎回答分析</title>
  {echarts_tag}
  <style>
{dashboard_css}
  </style>
</head>
<body>
  <main class="page">
    <header class="masthead">
      <div class="masthead-bar">
        <span class="brand"><span class="brand-mark"></span>知乎回答分析报告</span>
        <span class="masthead-meta">
          <span>NO. {_html_text(question_id or "—")}</span>
          <span>{_html_text(generated_at[:10])}</span>
          <button id="theme-toggle" class="theme-toggle" type="button" aria-label="切换深浅色主题">☾</button>
        </span>
      </div>
      <h1 class="headline">{_html_text(question_title)}</h1>
      <p class="dek">共分析 {_html_text(answer_count)} 条回答，累计 {_html_text(total_characters)} 字，覆盖 {_html_text(author_count)} 位作者。样本时间跨度为 {_html_text(date_range_display)}。</p>
      <div class="masthead-foot">
        {source_html}
        <span class="dot">◆</span>
        <span>生成于 {_html_text(generated_at)}</span>
        <span class="dot">◆</span>
        <span>情感均值 {_html_text(average_sentiment_display)}</span>
      </div>
    </header>

    <section class="ledger" aria-label="关键指标">
      <div class="ledger-item">
        <span class="ledger-label">回答总数</span>
        <strong class="ledger-value">{_html_text(answer_count)}</strong>
        <span class="ledger-note">累计 {_html_text(total_characters)} 字</span>
      </div>
      <div class="ledger-item">
        <span class="ledger-label">参与作者</span>
        <strong class="ledger-value">{_html_text(author_count)}</strong>
        <span class="ledger-note">样本内独立作者</span>
      </div>
      <div class="ledger-item">
        <span class="ledger-label">平均篇幅</span>
        <strong class="ledger-value">{_html_text(average_characters)}</strong>
        <span class="ledger-note">中位数 {_html_text(median_characters)} 字</span>
      </div>
      <div class="ledger-item">
        <span class="ledger-label">平均情感分</span>
        <strong class="ledger-value">{_html_text(average_sentiment_display)}</strong>
        <span class="ledger-note">正向 {_html_text(positive_count)} · 中性 {_html_text(neutral_count)} · 负向 {_html_text(negative_count)}</span>
      </div>
    </section>

    <section class="notes" aria-label="核心结论">
      <div class="notes-head">
        <span class="section-index">00</span>
        <h2>核心结论</h2>
      </div>
      <ol class="note-list">{notes_html}</ol>
    </section>

{_dashboard_section(
        "01",
        "主题语言",
        "词频、情绪、时间与互动四个切面，构成这份样本的基本轮廓。",
        f'<div class="grid">{keywords_panel}{sentiment_panel}{timeline_panel}{scatter_panel}</div>',
    )}
{_dashboard_section(
        "02",
        "主题聚类",
        "LDA 从回答文本中归纳话题簇；样本不足时自动跳过并说明原因。",
        f'<div class="grid">{topics_panel}</div>',
    )}
{_dashboard_section(
        "03",
        "作者网络",
        "基于主题、关键词与情感接近度推断的作者关联，并非知乎真实社交关系。",
        f'<div class="grid">{network_panel}</div>',
    )}

    <footer class="colophon">
      <span><strong>知乎回答分析报告</strong> · {_html_text(question_title)}</span>
      <span>{_html_text(answer_count)} 条回答 · {_html_text(generated_at)}</span>
    </footer>
  </main>

  <script id="report-data" type="application/json">{_json_for_script(payload)}</script>
  <script>
{dashboard_js}
  </script>
  <noscript><p class="empty-state">本页图表需要 JavaScript；Markdown 报告中的表格仍可阅读。</p></noscript>
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")


def make_relative_link(target: str | Path, base_dir: Path) -> str:
    try:
        return os.path.relpath(str(target), start=str(base_dir))
    except ValueError:
        # Windows: different drives have no relative path.
        return Path(target).resolve().as_uri()


def _md_cell(value: Any) -> str:
    """Keep Markdown table cells on one line and neutralize inline HTML."""
    text = "" if value is None else str(value)
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return text.replace("|", "\\|").replace("\n", " ").strip()


def _md_target(path: str | Path, base_dir: Path) -> str:
    """Percent-encode a local relative path so spaces and CJK survive Markdown."""
    relative = make_relative_link(path, base_dir)
    if relative.startswith("file:"):
        return relative
    relative = relative.replace(os.sep, "/")
    return quote(relative, safe="/")


def build_key_findings(summary: dict[str, Any]) -> list[str]:
    answer_count = summary.get("answer_count", 0)
    author_count = summary.get("author_count", 0)
    total_characters = summary.get("total_characters", 0)
    findings = [
        f"共分析 {answer_count} 条回答、{author_count} 位作者，累计 {total_characters} 字。",
        (
            f"单条回答平均 {summary.get('average_characters', 0)} 字，"
            f"中位数 {summary.get('median_characters', 0)} 字，"
            f"最长 {summary.get('max_characters', 0)} 字、最短 {summary.get('min_characters', 0)} 字。"
        ),
    ]

    average_sentiment = summary.get("average_sentiment")
    labels = summary.get("sentiment_labels") or {}
    if average_sentiment is not None:
        if average_sentiment >= 0.65:
            tone = "整体偏正向"
        elif average_sentiment <= 0.35:
            tone = "整体偏负向"
        else:
            tone = "整体接近中性"
        findings.append(
            f"SnowNLP 平均情感分 {average_sentiment}，{tone}；"
            f"正向 {labels.get('positive', 0)} 条、中性 {labels.get('neutral', 0)} 条、"
            f"负向 {labels.get('negative', 0)} 条。"
        )

    keywords = "、".join(item["word"] for item in summary.get("top_keywords", [])[:6])
    if keywords:
        findings.append(f"高频词集中在：{keywords}。")

    topics = summary.get("lda_topics") or []
    if topics:
        leading_topic = max(topics, key=lambda item: item.get("doc_count", 0))
        words = "、".join(leading_topic.get("top_words", [])[:4])
        share = float(leading_topic.get("doc_share") or 0) * 100
        findings.append(
            f"LDA 归纳出 {len(topics)} 个主题，覆盖最多的是 Topic {leading_topic.get('topic_id')}"
            f"（{leading_topic.get('doc_count', 0)} 条，{share:.1f}%），关键词为 {words}。"
        )

    network_highlights = summary.get("author_network_highlights") or []
    if network_highlights:
        top_link = network_highlights[0]
        findings.append(
            f"作者关联网络里，{top_link['source']} 与 {top_link['target']} 的推断关联最强，"
            f"依据为 {top_link.get('shared_keywords') or '同主题'}。"
        )

    return findings


def write_report(
    report_path: Path,
    summary: dict[str, Any],
    output_dir: Path,
) -> None:
    wordcloud_rel = _md_target(output_dir / "analysis" / "wordcloud.png", report_path.parent)
    dashboard_rel = _md_target(output_dir / "analysis" / "dashboard.html", report_path.parent)
    summary_rel = _md_target(output_dir / "analysis" / "summary.json", report_path.parent)
    dataset_rel = _md_target(output_dir / "analysis" / "answers.jsonl", report_path.parent)

    question_title = summary.get("question_title") or "知乎回答分析报告"
    question_url = summary.get("question_url") or ""
    labels = summary.get("sentiment_labels") or {}
    date_range = summary.get("date_range") or {}
    if date_range.get("start") and date_range.get("end"):
        date_range_display = (
            f"{date_range['start']} — {date_range['end']}"
            if date_range["start"] != date_range["end"]
            else date_range["start"]
        )
    else:
        date_range_display = "未提供"

    requested_answers = summary.get("requested_answers")
    if requested_answers is None:
        completeness_display = "本地数据"
    else:
        completeness_display = (
            "完整" if summary.get("complete", True) else "部分"
        ) + f"（请求 {requested_answers} 条）"

    lines = [
        f"# {_md_cell(question_title)}",
        "",
        f"> **生成时间**：{summary['generated_at']}  ",
        f"> **输入来源**：`{summary.get('source')}`  ",
        f"> **问题链接**：{question_url or 'N/A'}  ",
        (
            f"> **样本规模**：{summary['answer_count']} 条回答 · "
            f"{summary.get('author_count', 0)} 位作者 · "
            f"{summary['total_characters']} 字 · {date_range_display}"
        ),
        "",
        "> 阅读顺序：核心结论 → 数据概览 → 词云与高频词 → 主题聚类 → 作者网络 → 可视化看板 → 情感分析 → 代表性回答。",
        "",
        "---",
        "",
        "## 核心结论",
        "",
    ]

    for finding in build_key_findings(summary):
        lines.append(f"- {_md_cell(finding)}")

    lines.extend(
        [
            "",
            "---",
            "",
            "## 数据概览",
            "",
            "| 指标 | 数值 |",
            "|---|---|",
            f"| 回答数 | {summary['answer_count']} |",
            f"| 作者数 | {summary.get('author_count', 0)} |",
            f"| 总字数 | {summary['total_characters']} |",
            f"| 平均字数 | {summary['average_characters']} |",
            f"| 中位数字数 | {summary['median_characters']} |",
            f"| 最长回答 | {summary.get('max_characters', 0)} 字 |",
            f"| 最短回答 | {summary.get('min_characters', 0)} 字 |",
            f"| 样本完整性 | {completeness_display} |",
            f"| 时间跨度 | {_md_cell(date_range_display)} |",
            f"| 平均情感分 | {summary.get('average_sentiment', 'N/A')} |",
            f"| 情感标准差 | {summary.get('sentiment_stddev', 'N/A')} |",
            f"| 正向 / 中性 / 负向 | {labels.get('positive', 0)} / {labels.get('neutral', 0)} / {labels.get('negative', 0)} |",
            "",
            "---",
            "",
            "## 词云与高频词",
            "",
            f"![词云图]({wordcloud_rel})",
            "",
            "*图：按词频加权的关键词词云，字号越大表示在样本中出现越频繁。*",
            "",
            "| 关键词 | 频次 |",
            "|---|---|",
        ]
    )

    for item in summary["top_keywords"][:20]:
        lines.append(f"| {_md_cell(item['word'])} | {item['count']} |")

    lines.extend(["", "---", "", "## 主题聚类 (LDA)", ""])

    if summary.get("lda_topics"):
        lines.extend(
            [
                "| 主题 | 关键词 | 覆盖回答数 | 占比 |",
                "|---|---|---|---|",
            ]
        )
        for topic in summary["lda_topics"]:
            words = "、".join(topic.get("top_words", []))
            share = float(topic.get("doc_share") or 0) * 100
            lines.append(
                f"| Topic {topic['topic_id']} | {_md_cell(words)} | "
                f"{topic.get('doc_count', 0)} | {share:.1f}% |"
            )
    else:
        note = summary.get("lda_note") or "未启用 LDA 或数据不足，未生成主题聚类结果。"
        lines.append(_md_cell(note))

    lines.extend(["", "---", "", "## 作者关联网络", ""])

    author_network_note = summary.get("author_network_note") or (
        "作者关系图基于主题与关键词相似度推断，不代表知乎真实关注或互动关系。"
    )
    lines.append(_md_cell(author_network_note))

    author_network_highlights = summary.get("author_network_highlights") or []
    if author_network_highlights:
        lines.extend(
            [
                "",
                "| 作者 A | 作者 B | 关联强度 | 关联依据 |",
                "|---|---|---|---|",
            ]
        )
        for item in author_network_highlights:
            lines.append(
                f"| {_md_cell(item.get('source', ''))} | "
                f"{_md_cell(item.get('target', ''))} | "
                f"{item.get('score', 0)} | {_md_cell(item.get('relation') or '')} |"
            )
    else:
        if not summary.get("author_network_note"):
            lines.extend(["", "当前样本中未检测到足够强的作者关联边，暂不输出关系对。"])

    lines.extend(
        [
            "",
            "---",
            "",
            "## ECharts 可视化",
            "",
            f"[在新窗口打开交互式看板]({dashboard_rel})",
            "",
            "> 看板支持浅色/深色主题切换。若预览器不允许嵌入 iframe，请使用上面的链接直接打开。",
            "",
            '<div style="margin: 16px 0 24px;">',
            f'<iframe id="zhihu-dashboard" src="{dashboard_rel}" title="ECharts Dashboard" '
            'style="width: 100%; height: 3200px; border: 1px solid #d8cdb8; border-radius: 2px; background: #fbf7ee;" '
            'loading="lazy"></iframe>',
            "</div>",
            "<script>",
            "(function () {",
            '  var frame = document.getElementById("zhihu-dashboard");',
            "  if (!frame) { return; }",
            '  window.addEventListener("message", function (event) {',
            "    if (event.source !== frame.contentWindow) { return; }",
            "    var data = event.data || {};",
            '    if (data.type === "zhihu-dashboard-height" && typeof data.height === "number") {',
            '      frame.style.height = Math.max(1600, Math.min(data.height, 14000)) + "px";',
            "    }",
            "  });",
            "})();",
            "</script>",
            "",
            "---",
            "",
            "## 情感分析",
            "",
            "情感分由 SnowNLP 生成，取值范围接近 0 到 1：越接近 1 越偏正向，越接近 0 越偏负向。",
            "",
            "| 区间 | 回答数 |",
            "|---|---|",
        ]
    )

    for bucket in ["0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0", "unknown"]:
        name = "未知" if bucket == "unknown" else bucket
        lines.append(f"| {name} | {summary['sentiment_buckets'].get(bucket, 0)} |")

    lines.extend(
        [
            "",
            "---",
            "",
            "## 高频作者",
            "",
            "| 作者 | 回答数 |",
            "|---|---|",
        ]
    )

    for item in summary["top_authors"][:10]:
        lines.append(f"| {_md_cell(item['author'])} | {item['count']} |")

    lines.extend(
        [
            "",
            "---",
            "",
            "## 代表性回答",
            "",
            "按赞同数排序；缺少赞同数据时按篇幅排序。",
            "",
            "| 标题 | 作者 | 日期 | 赞同数 | 字数 | 情感分 | 原文 |",
            "|---|---|---|---|---|---|---|",
        ]
    )

    for item in summary["top_answers"]:
        markdown_path = item.get("markdown_path") or ""
        if markdown_path:
            source_cell = f"[index.md]({_md_target(markdown_path, report_path.parent)})"
        else:
            source_cell = "—"
        lines.append(
            f"| {_md_cell(item['title'])} | {_md_cell(item['author'])} | "
            f"{_md_cell(item.get('date') or '')} | "
            f"{item.get('upvotes') if item.get('upvotes') is not None else 'N/A'} | "
            f"{item.get('char_count', 0)} | "
            f"{item.get('sentiment') if item.get('sentiment') is not None else 'N/A'} | "
            f"{source_cell} |"
        )

    lines.extend(
        [
            "",
            "---",
            "",
            "## 数据文件",
            "",
            f"- [结构化摘要 summary.json]({summary_rel})",
            f"- [答案数据 answers.jsonl]({dataset_rel})",
            f"- [交互式看板 dashboard.html]({dashboard_rel})",
            "",
            "---",
            "",
            "> **方法与局限**：词频经 jieba 分词并过滤停用词；情感分来自 SnowNLP 的通用语料模型，"
            "对反讽、行业黑话和长文本的判断可能偏差；LDA 主题与作者网络均为统计推断，"
            "作者网络只反映文本相似度，不代表知乎真实关注、评论或私下社交关系。",
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
    try:
        create_wordcloud(frequencies, wordcloud_png, WordCloud, font_path)
    except SystemExit as exc:
        print(f"⚠️ 词云生成失败：{exc}")
        if not create_placeholder_wordcloud(wordcloud_png, font_path, str(exc)):
            raise
    write_dashboard(dashboard_html, summary, records)
    write_report(report_md, summary, output_dir)


async def main() -> None:
    args = parse_args()
    prepared = prepare_source(args.source)

    if prepared.mode == "url":
        require_cookies_for_url_mode()

    # Validate local dependencies before touching the network.
    jieba, SnowNLP, WordCloud, LDA, CountVectorizer = load_analysis_dependencies()
    stopwords = load_stopwords(args.stopwords)
    font_path = resolve_wordcloud_font(args.font_path)

    if prepared.mode == "url":
        prepared = fetch_question_overview(prepared)
        if args.answer_cap is not None:
            if prepared.total_answers is None:
                prepared.total_answers = args.answer_cap
            else:
                prepared.total_answers = min(prepared.total_answers, args.answer_cap)

    output_dir = resolve_output_dir(prepared, args.output_dir)

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
    lda_topics, lda_distribution, lda_note = run_lda_topics(
        records=records,
        jieba=jieba,
        stopwords=stopwords,
        min_word_length=args.min_word_length,
        topic_count=args.lda_topics,
        top_words=args.lda_words,
        max_iter=args.lda_max_iter,
        LDA=LDA,
        CountVectorizer=CountVectorizer,
    )
    author_network, author_network_highlights, author_network_note = build_author_network(
        records=records,
        jieba=jieba,
        stopwords=stopwords,
        min_word_length=args.min_word_length,
        lda_topics=lda_topics,
        max_nodes=args.network_max_nodes,
        max_edges=args.network_max_edges,
    )
    summary = build_summary(
        records,
        frequencies,
        prepared,
        args.top_k,
        lda_topics=lda_topics,
        lda_distribution=lda_distribution,
        lda_note=lda_note,
        author_network=author_network,
        author_network_highlights=author_network_highlights,
        author_network_note=author_network_note,
    )
    write_outputs(output_dir, summary, records, frequencies, WordCloud, font_path)

    print(f"Report bundle created at: {output_dir}")
    print(f"Markdown report: {output_dir / 'report.md'}")
    print(f"ECharts dashboard: {output_dir / 'analysis' / 'dashboard.html'}")


if __name__ == "__main__":
    asyncio.run(main())
