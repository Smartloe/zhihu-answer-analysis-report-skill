"""Tests for the generated dashboard, Markdown report and file outputs."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from urllib.parse import unquote

import pytest


def _records() -> list[dict]:
    return [
        {
            "author": "甲",
            "text": "乒乓球 央视",
            "char_count": 12,
            "token_count": 2,
            "sentiment": 0.9,
            "upvotes": 10,
            "date": "2026-01-01",
            "title": "标题A",
            "url": "https://example.com/a",
            "markdown_path": "/tmp/a/index.md",
        },
        {
            "author": "乙",
            "text": "德甲 体育",
            "char_count": 20,
            "token_count": 2,
            "sentiment": 0.5,
            "upvotes": 5,
            "date": "2026-01-02",
            "title": "标题B",
            "url": "https://example.com/b",
            "markdown_path": "/tmp/b/index.md",
        },
        {
            "author": "甲",
            "text": "中国 国际",
            "char_count": 30,
            "token_count": 2,
            "sentiment": 0.1,
            "upvotes": 3,
            "date": "2026-01-02",
            "title": "标题C",
            "url": "https://example.com/c",
            "markdown_path": "/tmp/c/index.md",
        },
    ]


def _summary(report_module, tmp_path, records=None, title="测试问题"):
    records = _records() if records is None else records
    prepared = report_module.PreparedSource(
        mode="path",
        source=str(tmp_path),
        path=tmp_path,
        question_title=title,
    )
    return report_module.build_summary(
        records,
        Counter({"乒乓球": 3, "央视": 2, "德甲": 1}),
        prepared,
        top_k=20,
    )


def _payload(html: str) -> dict:
    """Read the JSON payload the dashboard hands to its runtime script."""
    match = re.search(
        r'<script id="report-data" type="application/json">(.*?)</script>',
        html,
        re.S,
    )
    assert match is not None, "dashboard JSON payload script was not emitted"
    return json.loads(match.group(1))


def _table_rows(report: str, header: str) -> list[str]:
    lines = report.splitlines()
    assert header in lines, f"table header not found: {header}"
    start = lines.index(header) + 2  # header + separator
    rows = []
    for line in lines[start:]:
        if not line.startswith("|"):
            break
        rows.append(line)
    return rows


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------


def test_write_dashboard_embeds_echarts_and_parseable_data(report_module, tmp_path):
    summary = _summary(report_module, tmp_path)
    dashboard = tmp_path / "analysis" / "dashboard.html"
    dashboard.parent.mkdir()

    report_module.write_dashboard(dashboard, summary, _records())
    html = dashboard.read_text(encoding="utf-8")

    assert "echarts" in html
    assert "echarts.init" in html
    assert html.lstrip().startswith("<!doctype html>")
    assert html.rstrip().endswith("</html>")

    payload = _payload(html)
    assert payload["topKeywords"] == summary["top_keywords"]
    assert payload["sentimentBuckets"] == summary["sentiment_buckets"]
    assert payload["timeline"] == summary["timeline"]
    assert payload["ldaDistribution"] == summary["lda_topic_distribution"]
    assert payload["authorNetwork"] == summary["author_network"]
    assert payload["bucketOrder"] == [
        "0.0-0.2",
        "0.2-0.4",
        "0.4-0.6",
        "0.6-0.8",
        "0.8-1.0",
        "unknown",
    ]
    assert len(payload["scatter"]) == 3
    assert all(len(point) == 5 for point in payload["scatter"])

    assert "乒乓球" in html
    assert "测试问题" in html


def test_write_dashboard_has_no_unrendered_python_braces(report_module, tmp_path):
    summary = _summary(report_module, tmp_path)
    dashboard = tmp_path / "dashboard.html"

    report_module.write_dashboard(dashboard, summary, _records())
    html = dashboard.read_text(encoding="utf-8")

    assert "{{" not in html
    assert "{summary" not in html
    assert "{record" not in html
    assert "{_html_text" not in html
    assert "{question_title" not in html
    # The embedded JSON payload must remain valid after template rendering.
    assert isinstance(_payload(html), dict)


def test_dashboard_inlines_assets_and_supports_theming(report_module, tmp_path):
    summary = _summary(report_module, tmp_path)
    dashboard = tmp_path / "dashboard.html"

    report_module.write_dashboard(dashboard, summary, _records())
    html = dashboard.read_text(encoding="utf-8")

    # Assets are inlined, so the dashboard works without a bundler or extra files.
    assert "--paper:" in html
    assert ".panel-head" in html
    assert "zhihu-report-theme" in html
    assert 'id="theme-toggle"' in html
    assert "prefers-color-scheme" in html
    # The embedded dashboard reports its height to the Markdown iframe host.
    assert "zhihu-dashboard-height" in html
    assert "postMessage" in html
    # Only ECharts may be loaded externally, and it is vendored when available.
    assert html.count("<link") == 0
    assert html.count("<script src=") <= 1
    assert "echarts" in html


def test_write_dashboard_escapes_untrusted_title_and_payload(report_module, tmp_path):
    payload = '<script>alert("xss")</script>'
    summary = _summary(report_module, tmp_path, title=payload)
    records = _records()
    records[0]["title"] = payload
    dashboard = tmp_path / "dashboard.html"

    report_module.write_dashboard(dashboard, summary, records)
    html = dashboard.read_text(encoding="utf-8")

    # The raw payload must never appear as markup or inside the JSON script tag.
    assert payload not in html
    payload_region = re.search(
        r'<script id="report-data" type="application/json">(.*?)</script>',
        html,
        re.S,
    ).group(1)
    assert "</script>" not in payload_region
    assert "<\\/script>" in payload_region
    assert "&lt;script&gt;" in html  # escaped in the visible heading/title
    # Only the four intended script elements (theme pre-paint, ECharts, JSON payload, runtime).
    assert html.count("</script>") == 4
    assert _payload(html)["authorNetwork"] == {"nodes": [], "links": [], "categories": []}


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------


REPORT_SECTIONS = [
    "## 核心结论",
    "## 数据概览",
    "## 词云与高频词",
    "## 主题聚类 (LDA)",
    "## 作者关联网络",
    "## 情感分析",
    "## ECharts 可视化",
    "## 高频作者",
    "## 代表性回答",
    "## 数据文件",
]


def test_report_contains_all_sections(path_bundle):
    report = (path_bundle / "report.md").read_text(encoding="utf-8")

    assert report.startswith("# ")
    for section in REPORT_SECTIONS:
        assert section in report, f"missing report section: {section}"
    assert "<iframe" in report
    assert "analysis/dashboard.html" in report
    assert "![词云图](analysis/wordcloud.png)" in report
    assert "- [结构化摘要 summary.json](analysis/summary.json)" in report
    assert "- [答案数据 answers.jsonl](analysis/answers.jsonl)" in report


def test_report_table_row_counts_match_summary(path_bundle):
    report = (path_bundle / "report.md").read_text(encoding="utf-8")
    summary = json.loads(
        (path_bundle / "analysis" / "summary.json").read_text(encoding="utf-8")
    )

    assert len(_table_rows(report, "| 指标 | 数值 |")) == 13
    assert len(_table_rows(report, "| 关键词 | 频次 |")) == min(
        20, len(summary["top_keywords"])
    )
    assert len(_table_rows(report, "| 主题 | 关键词 | 覆盖回答数 | 占比 |")) == len(
        summary["lda_topics"]
    )
    assert len(_table_rows(report, "| 作者 A | 作者 B | 关联强度 | 关联依据 |")) == len(
        summary["author_network_highlights"]
    )
    assert len(_table_rows(report, "| 区间 | 回答数 |")) == 6
    assert len(_table_rows(report, "| 作者 | 回答数 |")) == min(
        10, len(summary["top_authors"])
    )
    assert len(
        _table_rows(
            report, "| 标题 | 作者 | 日期 | 赞同数 | 字数 | 情感分 | 原文 |"
        )
    ) == len(summary["top_answers"])


def _parse_markdown_links(text: str) -> list[tuple[str, str]]:
    """Parse CommonMark inline links, honouring <angle bracket> destinations."""
    pattern = re.compile(r"\[([^\]]+)\]\((?:<([^>]+)>|([^\s)]+))\)")
    return [
        (match.group(1), match.group(2) or match.group(3))
        for match in pattern.finditer(text)
    ]


def test_report_links_with_spaces_and_chinese_resolve(report_module, tmp_path):
    output = tmp_path / "bundle"
    output.mkdir()
    (output / "analysis").mkdir()
    (output / "analysis" / "summary.json").write_text("{}", encoding="utf-8")
    (output / "analysis" / "answers.jsonl").write_text("", encoding="utf-8")
    target = (
        output
        / "raw"
        / "entries"
        / "[2026-01-01] 中文 标题 (answer-1)"
        / "index.md"
    )
    target.parent.mkdir(parents=True)
    target.write_text("# 回答\n\n正文\n", encoding="utf-8")
    records = [
        {
            "author": "作者甲",
            "text": "正文",
            "char_count": 2,
            "token_count": 1,
            "sentiment": 0.5,
            "upvotes": 1,
            "date": "2026-01-01",
            "title": "中文 标题",
            "url": "https://example.com",
            "markdown_path": str(target),
        }
    ]
    summary = _summary(report_module, tmp_path, records=records)
    report_path = output / "report.md"

    report_module.write_report(report_path, summary, output)
    report = report_path.read_text(encoding="utf-8")

    representative_links = [
        destination
        for label, destination in _parse_markdown_links(report)
        if label == "index.md"
    ]
    assert representative_links, "representative answer link was not emitted"
    for destination in representative_links:
        # _md_target percent-encodes spaces/CJK, so the parsed URI needs unquoting
        # before it can be checked against the filesystem.
        assert "%20" in destination
        assert (output / unquote(destination)).resolve().is_file()


def test_report_data_file_links_resolve(report_module, tmp_path):
    output = tmp_path / "bundle"
    (output / "analysis").mkdir(parents=True)
    (output / "analysis" / "summary.json").write_text("{}", encoding="utf-8")
    (output / "analysis" / "answers.jsonl").write_text("", encoding="utf-8")
    target = output / "index.md"
    target.write_text("# 回答\n", encoding="utf-8")
    records = [
        {
            "author": "作者",
            "text": "正文",
            "char_count": 2,
            "token_count": 1,
            "sentiment": 0.5,
            "upvotes": 1,
            "date": "2026-01-01",
            "title": "标题",
            "url": "https://example.com",
            "markdown_path": str(target),
        }
    ]
    summary = _summary(report_module, tmp_path, records=records)
    report_path = output / "report.md"

    report_module.write_report(report_path, summary, output)
    report = report_path.read_text(encoding="utf-8")

    links = dict(_parse_markdown_links(report))
    assert (output / links["结构化摘要 summary.json"]).is_file()
    assert (output / links["答案数据 answers.jsonl"]).is_file()


def test_report_escapes_pipes_in_tables(report_module, tmp_path):
    output = tmp_path / "bundle"
    output.mkdir()
    records = [
        {
            "author": "作者|甲",
            "text": "正文",
            "char_count": 2,
            "token_count": 1,
            "sentiment": 0.5,
            "upvotes": 1,
            "date": "2026-01-01",
            "title": "标题|带竖线",
            "url": "https://example.com",
            "markdown_path": "/tmp/index.md",
        }
    ]
    summary = _summary(report_module, tmp_path, records=records)

    report_module.write_report(output / "report.md", summary, output)
    report = (output / "report.md").read_text(encoding="utf-8")

    assert "标题\\|带竖线" in report
    assert "作者\\|甲" in report
    assert "| 标题|带竖线 |" not in report


# ---------------------------------------------------------------------------
# Auxiliary writers
# ---------------------------------------------------------------------------


def test_write_jsonl_roundtrip(report_module, tmp_path):
    path = tmp_path / "records.jsonl"
    records = [{"author": "甲", "text": "中文"}, {"author": "乙", "text": "内容"}]

    report_module.write_jsonl(path, records)

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert [json.loads(line) for line in lines] == records


def test_create_wordcloud_writes_png(report_module, analysis_deps, font_path, tmp_path):
    path = tmp_path / "wordcloud.png"

    report_module.create_wordcloud(
        Counter({"乒乓球": 5, "央视": 3, "德甲": 2}),
        path,
        analysis_deps["WordCloud"],
        font_path,
    )

    assert path.is_file()
    assert path.stat().st_size > 0
    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_create_wordcloud_rejects_empty_frequencies(
    report_module, analysis_deps, font_path, tmp_path
):
    with pytest.raises(SystemExit):
        report_module.create_wordcloud(
            Counter(), tmp_path / "wordcloud.png", analysis_deps["WordCloud"], font_path
        )


def test_empty_keywords_still_generate_a_report(run_cli, tmp_path):
    dataset = tmp_path / "empty.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "id": "1",
                "title": "停顿词样本",
                "author": "作者",
                "url": "https://www.zhihu.com/answer/1",
                "date": "2026-01-01",
                "type": "answer",
                "upvotes": 1,
                "markdown_path": "",
                "text": "的 了 和 是 在 就 都 而",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "bundle"
    result = run_cli(dataset, "--output-dir", output)

    assert result.returncode == 0, result.stderr
    assert (output / "report.md").is_file()
    assert (output / "analysis" / "dashboard.html").is_file()
    assert (output / "analysis" / "wordcloud.png").read_bytes()[:8] == (
        b"\x89PNG\r\n\x1a\n"
    )


def test_write_outputs_creates_complete_bundle(
    report_module, analysis_deps, font_path, tmp_path
):
    output = tmp_path / "bundle"
    records = _records()
    summary = _summary(report_module, tmp_path)
    frequencies = Counter({"乒乓球": 3, "央视": 2, "德甲": 1})

    report_module.write_outputs(
        output, summary, records, frequencies, analysis_deps["WordCloud"], font_path
    )

    for relative in (
        "report.md",
        "analysis/summary.json",
        "analysis/answers.jsonl",
        "analysis/corpus.txt",
        "analysis/wordcloud.png",
        "analysis/dashboard.html",
    ):
        assert (output / relative).is_file(), f"missing {relative}"
        assert (output / relative).stat().st_size > 0
    corpus = (output / "analysis" / "corpus.txt").read_text(encoding="utf-8")
    assert "乒乓球 央视" in corpus
    assert "德甲 体育" in corpus
    stored_summary = json.loads(
        (output / "analysis" / "summary.json").read_text(encoding="utf-8")
    )
    assert stored_summary["answer_count"] == 3
