"""End-to-end smoke tests for ``scripts/zhihu_answer_report.py``.

These cover the two documented input modes (local Markdown scrape and an
existing ``answers.jsonl`` dataset), the CLI surface, record loading, and the
completeness of ``analysis/summary.json``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


def _load_summary(bundle: Path) -> dict:
    return json.loads(
        (bundle / "analysis" / "summary.json").read_text(encoding="utf-8")
    )


# ---------------------------------------------------------------------------
# parse_args / prepare_source helpers
# ---------------------------------------------------------------------------


def test_parse_args_defaults(report_module, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["zhihu_answer_report.py", "some-source"])
    args = report_module.parse_args()

    assert args.source == "some-source"
    assert args.output_dir is None
    assert args.answer_cap is None
    assert args.no_images is False
    assert args.top_k == 30
    assert args.min_word_length == 2
    assert args.stopwords is None
    assert args.font_path is None
    assert args.lda_topics == 6
    assert args.lda_words == 8
    assert args.lda_max_iter == 15
    assert args.network_max_nodes == 18
    assert args.network_max_edges == 28


def test_parse_args_accepts_all_analysis_flags(report_module, monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "zhihu_answer_report.py",
            "source",
            "--output-dir",
            "out",
            "--answer-cap",
            "3",
            "--no-images",
            "--top-k",
            "5",
            "--min-word-length",
            "3",
            "--stopwords",
            "sw.txt",
            "--font-path",
            "font.ttc",
            "--lda-topics",
            "0",
            "--lda-words",
            "4",
            "--lda-max-iter",
            "2",
            "--network-max-nodes",
            "0",
            "--network-max-edges",
            "0",
        ],
    )
    args = report_module.parse_args()

    assert args.output_dir == "out"
    assert args.answer_cap == 3
    assert args.no_images is True
    assert args.top_k == 5
    assert args.min_word_length == 3
    assert args.stopwords == "sw.txt"
    assert args.font_path == "font.ttc"
    assert args.lda_topics == 0
    assert args.lda_words == 4
    assert args.lda_max_iter == 2
    assert args.network_max_nodes == 0
    assert args.network_max_edges == 0


def test_prepare_source_existing_path(report_module, sample_raw):
    prepared = report_module.prepare_source(str(sample_raw))

    assert prepared.mode == "path"
    assert prepared.path == sample_raw.resolve()
    assert prepared.question_url is None


def test_prepare_source_question_url(report_module):
    prepared = report_module.prepare_source(
        "https://www.zhihu.com/question/2010761296434458950"
    )

    assert prepared.mode == "url"
    assert prepared.question_id == "2010761296434458950"
    assert prepared.question_url == (
        "https://www.zhihu.com/question/2010761296434458950"
    )


def test_prepare_source_answer_url_normalizes_to_question(report_module):
    prepared = report_module.prepare_source(
        "https://www.zhihu.com/question/2010761296434458950/answer/2011056805845951188"
    )

    assert prepared.mode == "url"
    assert prepared.question_id == "2010761296434458950"


def test_prepare_source_rejects_unsupported_input(report_module):
    with pytest.raises(SystemExit):
        report_module.prepare_source("not-a-url-or-path")


def test_prepare_source_accepts_share_text_with_url(report_module):
    prepared = report_module.prepare_source(
        "如何评价这个问题？ https://www.zhihu.com/question/2010761296434458950 转发一下"
    )

    assert prepared.mode == "url"
    assert prepared.question_id == "2010761296434458950"


def test_prepare_source_accepts_mobile_url(report_module):
    prepared = report_module.prepare_source(
        "https://m.zhihu.com/question/2010761296434458950"
    )

    assert prepared.mode == "url"
    assert prepared.question_id == "2010761296434458950"


def test_detect_dns_servers_honours_environment(report_module, monkeypatch):
    from core.api_client import detect_dns_servers

    monkeypatch.setenv("ZHIHU_DNS_SERVERS", "1.1.1.1,8.8.8.8")
    assert detect_dns_servers() == "1.1.1.1,8.8.8.8"

    monkeypatch.delenv("ZHIHU_DNS_SERVERS")
    monkeypatch.setenv("ZHIHU_DISABLE_DNS_FIX", "1")
    assert detect_dns_servers() is None


# ---------------------------------------------------------------------------
# Stopwords / font / Markdown extraction
# ---------------------------------------------------------------------------


def test_load_stopwords_merges_base_and_extra_file(report_module, tmp_path):
    extra = tmp_path / "extra-stopwords.txt"
    extra.write_text("乒乓球\n\n自定义停用词\n", encoding="utf-8")

    merged = report_module.load_stopwords(str(extra))

    assert "的" in merged  # from assets/stopwords_zh.txt
    assert {"乒乓球", "自定义停用词"} <= merged


def test_load_stopwords_without_extra_file(report_module, stopwords):
    assert "的" in stopwords
    assert "乒乓球" not in stopwords


def test_resolve_wordcloud_font_honours_existing_override(report_module, font_path):
    resolved = report_module.resolve_wordcloud_font(font_path)

    assert resolved == str(Path(font_path).resolve())
    assert Path(resolved).is_file()


def test_extract_text_from_markdown_strips_header_and_markup(report_module):
    markdown = (
        "# 标题\n\n"
        "> **Author / 作者**: 某人  \n"
        "> **Source / 来源**: [link](https://example.com)  \n\n"
        "---\n\n"
        "正文 [链接](https://example.com) 与 ![图](a.png)\n\n"
        "```python\nprint('hidden')\n```\n\n"
        "尾部 `代码` 与 <b>标签</b>\n"
    )

    text = report_module.extract_text_from_markdown(markdown)

    assert "标题" not in text
    assert "Author" not in text
    assert "正文" in text
    assert "链接" in text
    assert "尾部" in text
    assert "hidden" not in text
    assert "a.png" not in text
    assert "https://example.com" not in text
    assert "<b>" not in text


# ---------------------------------------------------------------------------
# load_records_from_path
# ---------------------------------------------------------------------------


def test_load_records_from_raw_markdown_directory(report_module, sample_raw):
    records = report_module.load_records_from_path(sample_raw)

    assert len(records) == len(list(sample_raw.rglob("index.md")))
    first = records[0]
    assert {
        "id",
        "title",
        "author",
        "url",
        "date",
        "type",
        "upvotes",
        "markdown_path",
        "text",
    } <= set(first)
    assert first["title"]
    assert first["author"]
    assert first["markdown_path"].endswith("index.md")
    assert first["text"]


def test_load_records_from_answers_jsonl_file(report_module, sample_jsonl):
    records = report_module.load_records_from_path(sample_jsonl)
    expected = len(
        [line for line in sample_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    )

    assert len(records) == expected
    assert all("text" in record for record in records)
    assert all("markdown_path" in record for record in records)


def test_load_records_from_arbitrary_jsonl_filename(report_module, tmp_path):
    payload = {
        "id": "1",
        "title": "T",
        "author": "A",
        "url": "https://example.com",
        "date": "2026-01-01",
        "type": "answer",
        "upvotes": 1,
        "markdown_path": str(tmp_path / "index.md"),
        "text": "正文",
    }
    dataset = tmp_path / "custom-export.jsonl"
    dataset.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")

    assert report_module.load_records_from_path(dataset) == [payload]


def test_load_records_from_json_array_file(report_module, tmp_path):
    payload = {
        "id": "1",
        "title": "T",
        "author": "A",
        "url": "https://example.com",
        "date": "2026-01-01",
        "type": "answer",
        "upvotes": 1,
        "markdown_path": str(tmp_path / "index.md"),
        "text": "正文",
    }
    dataset = tmp_path / "custom-export.json"
    dataset.write_text(json.dumps([payload], ensure_ascii=False), encoding="utf-8")

    assert report_module.load_records_from_path(dataset) == [payload]


def test_load_records_prefers_analysis_answers_jsonl(report_module, tmp_path):
    analysis = tmp_path / "analysis"
    analysis.mkdir()
    payload = {
        "id": "1",
        "title": "T",
        "author": "A",
        "url": "https://example.com",
        "date": "2026-01-01",
        "type": "answer",
        "upvotes": 1,
        "markdown_path": str(tmp_path / "index.md"),
        "text": "正文",
    }
    (analysis / "answers.jsonl").write_text(
        json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    records = report_module.load_records_from_path(tmp_path)

    assert records == [payload]


def test_load_records_from_single_markdown_file(report_module, sample_raw):
    markdown_file = next(sample_raw.rglob("index.md"))

    records = report_module.load_records_from_path(markdown_file)

    assert len(records) == 1
    assert records[0]["title"]
    assert records[0]["author"]
    assert "zhihu.com" in records[0]["url"]
    assert records[0]["date"]
    assert records[0]["text"]


def test_load_records_missing_directory_raises(report_module, tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()

    with pytest.raises(SystemExit):
        report_module.load_records_from_path(empty)


def test_normalize_records_resolves_paths_and_cleans_text(report_module, tmp_path):
    markdown_file = tmp_path / "nested" / "index.md"
    markdown_file.parent.mkdir()
    markdown_file.write_text(
        "# T\n\n---\n\n正文 [链接](https://example.com) **粗体**\n", encoding="utf-8"
    )
    records = [
        {
            "markdown_path": str(markdown_file),
            "text": "# T\n\n---\n\n正文 [链接](https://example.com) **粗体**",
            "author": "A",
        }
    ]

    normalized = report_module.normalize_records(records)

    assert normalized[0]["markdown_path"] == str(markdown_file.resolve())
    assert "正文" in normalized[0]["text"]
    assert "链接" in normalized[0]["text"]
    assert "https://example.com" not in normalized[0]["text"]
    assert "# T" not in normalized[0]["text"]


# ---------------------------------------------------------------------------
# End-to-end artifacts
# ---------------------------------------------------------------------------


EXPECTED_ARTIFACTS = [
    "report.md",
    "analysis/summary.json",
    "analysis/answers.jsonl",
    "analysis/corpus.txt",
    "analysis/wordcloud.png",
    "analysis/dashboard.html",
]


def test_path_mode_generates_all_artifacts(path_bundle):
    for relative in EXPECTED_ARTIFACTS:
        artifact = path_bundle / relative
        assert artifact.is_file(), f"missing artifact: {relative}"
        assert artifact.stat().st_size > 0, f"empty artifact: {relative}"

    assert (path_bundle / "analysis" / "wordcloud.png").read_bytes()[:8] == (
        b"\x89PNG\r\n\x1a\n"
    )


def test_path_mode_summary_reports_sample_dataset(path_bundle, sample_raw):
    summary = _load_summary(path_bundle)
    expected_answers = len(list(sample_raw.rglob("index.md")))

    assert summary["answer_count"] == expected_answers
    assert summary["author_count"] > 0
    assert summary["author_count"] <= expected_answers
    assert summary["question_title"]
    assert summary["source"].endswith("raw")
    assert summary["question_url"] is None
    assert summary["total_characters"] > 0
    assert summary["average_characters"] > 0
    assert summary["top_keywords"]
    assert summary["timeline"]
    assert summary["date_range"] is not None


def test_path_mode_answers_jsonl_matches_summary(path_bundle):
    summary = _load_summary(path_bundle)
    lines = [
        line
        for line in (path_bundle / "analysis" / "answers.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]

    assert len(lines) == summary["answer_count"]
    records = [json.loads(line) for line in lines]
    assert all("sentiment" in record for record in records)
    assert all("char_count" in record for record in records)
    assert all("token_count" in record for record in records)


def test_answers_jsonl_input_mode_generates_all_artifacts(jsonl_bundle, sample_jsonl):
    for relative in EXPECTED_ARTIFACTS:
        artifact = jsonl_bundle / relative
        assert artifact.is_file(), f"missing artifact: {relative}"
        assert artifact.stat().st_size > 0, f"empty artifact: {relative}"

    summary = _load_summary(jsonl_bundle)
    first_record = json.loads(
        next(line for line in sample_jsonl.read_text(encoding="utf-8").splitlines() if line.strip())
    )
    expected = len(
        [line for line in sample_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    )
    assert summary["answer_count"] == expected
    assert summary["source"].endswith(".jsonl")
    assert summary["question_title"] == first_record["title"]


REQUIRED_SUMMARY_KEYS = {
    "generated_at",
    "source",
    "question_url",
    "question_id",
    "question_title",
    "answer_count",
    "author_count",
    "total_characters",
    "average_characters",
    "median_characters",
    "max_characters",
    "min_characters",
    "average_sentiment",
    "sentiment_stddev",
    "sentiment_labels",
    "sentiment_buckets",
    "top_keywords",
    "lda_topics",
    "lda_topic_distribution",
    "lda_note",
    "author_network",
    "author_network_highlights",
    "author_network_note",
    "top_authors",
    "timeline",
    "date_range",
    "top_answers",
}


def test_summary_has_all_expected_fields(path_bundle):
    summary = _load_summary(path_bundle)

    assert REQUIRED_SUMMARY_KEYS <= set(summary)
    assert set(summary["author_network"]) == {"nodes", "links", "categories"}
    assert all(set(item) == {"word", "count"} for item in summary["top_keywords"])
    assert all(
        set(item) == {"topic_id", "top_words", "doc_count", "doc_share"}
        for item in summary["lda_topics"]
    )
    assert all(
        set(item) == {"date", "count"} for item in summary["timeline"]
    )
    assert all(
        set(item) == {"author", "count"} for item in summary["top_authors"]
    )
    assert all(
        {
            "title",
            "author",
            "date",
            "url",
            "upvotes",
            "char_count",
            "sentiment",
            "markdown_path",
        }
        == set(item)
        for item in summary["top_answers"]
    )
    if summary["lda_topics"]:
        assert sum(topic["doc_share"] for topic in summary["lda_topics"]) == (
            pytest.approx(1.0, abs=1e-3)
        )
        assert sum(topic["doc_count"] for topic in summary["lda_topics"]) == (
            summary["answer_count"]
        )


# ---------------------------------------------------------------------------
# CLI surface
# ---------------------------------------------------------------------------


def test_cli_help_lists_all_options(run_cli):
    result = run_cli("--help")

    assert result.returncode == 0
    assert "usage:" in result.stdout
    for flag in (
        "--output-dir",
        "--answer-cap",
        "--no-images",
        "--top-k",
        "--min-word-length",
        "--stopwords",
        "--font-path",
        "--lda-topics",
        "--lda-words",
        "--lda-max-iter",
        "--network-max-nodes",
        "--network-max-edges",
    ):
        assert flag in result.stdout


def test_cli_lda_zero_and_network_zero_disable_features(flag_bundle):
    output = flag_bundle["output"]
    summary = _load_summary(output)

    assert summary["lda_topics"] == []
    assert summary["lda_topic_distribution"] == []
    assert "LDA" in summary["lda_note"]
    assert "关闭" in summary["lda_note"]
    assert summary["author_network"] == {
        "nodes": [],
        "links": [],
        "categories": [],
    }
    assert summary["author_network_highlights"] == []
    assert "作者" in summary["author_network_note"]
    assert "关闭" in summary["author_network_note"]


def test_cli_custom_stopwords_are_applied(flag_bundle):
    output = flag_bundle["output"]
    summary = _load_summary(output)
    words = {item["word"] for item in summary["top_keywords"]}

    assert "樊振东" not in words
    assert summary["top_keywords"], "stopword filtering should not empty the keyword list"


def test_cli_custom_font_path_still_produces_wordcloud(flag_bundle):
    output = flag_bundle["output"]
    wordcloud = output / "analysis" / "wordcloud.png"

    assert wordcloud.is_file()
    assert wordcloud.stat().st_size > 0
    assert wordcloud.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
