"""Regression tests for the robustness fixes reported by the code audit."""

from __future__ import annotations

import sys

import pytest

from core import api_client
from core.converter import ZhihuConverter
from core.cookie_manager import cookie_manager
from core.scraper import ZhihuCreatorDownloader
from core.utils import sanitize_filename


def test_extract_text_keeps_body_horizontal_rules(report_module):
    body = "第一段内容。\n\n---\n\n第二段内容。"
    assert report_module.extract_text_from_markdown(body) == "第一段内容。 第二段内容。"


def test_extract_text_strips_generated_header_only(report_module):
    content = (
        "# 测试问题\n\n"
        "> **Author / 作者**: 作者  \n"
        "> **Source / 来源**: [https://e](https://e)  \n"
        "> **Date / 日期**: 2026-01-01\n\n"
        "---\n\n"
        "第一段内容。\n\n---\n\n第二段内容。"
    )
    text = report_module.extract_text_from_markdown(content)
    assert "第一段内容" in text
    assert "第二段内容" in text
    assert "测试问题" not in text


def test_extract_text_handles_none(report_module):
    assert report_module.extract_text_from_markdown(None) == ""


def test_normalize_records_handles_missing_and_messy_fields(report_module):
    records = [
        {"title": None, "author": 123, "text": None, "upvotes": "1.2万"},
        {"title": "T", "author": None, "text": "正文", "upvotes": "1,234"},
    ]
    normalized = report_module.normalize_records(records)

    assert normalized[0]["author"] == "123"
    assert normalized[0]["title"] == ""
    assert normalized[0]["upvotes"] == 12000
    assert normalized[0]["markdown_path"] == ""
    assert normalized[0]["text"] == ""
    assert normalized[1]["author"] == "Unknown"
    assert normalized[1]["upvotes"] == 1234


def test_normalize_records_rejects_non_object(report_module):
    with pytest.raises(SystemExit):
        report_module.normalize_records(["not-an-object"])


def test_sanitize_filename_limits_utf8_bytes():
    name = sanitize_filename("中" * 100, max_length=50)
    assert len(name.encode("utf-8")) <= 50
    assert name


def test_long_chinese_folder_names_stay_under_255_bytes(report_module):
    prepared = report_module.PreparedSource(
        mode="url",
        source="https://www.zhihu.com/question/1",
        question_id="1",
        question_url="https://www.zhihu.com/question/1",
        question_title="很长的中文标题" * 20,
    )
    folder = report_module.default_output_dir(prepared)
    assert len(folder.name.encode("utf-8")) <= 255

    folder_name = report_module.build_output_folder_name(
        "2026-01-01", "长" * 200, "作" * 200, "answer-1234567890"
    )
    assert len(folder_name.encode("utf-8")) <= 255


def test_katex_repeat_expansion_is_capped():
    expanded = ZhihuConverter._fix_katex_array("*{100000}{c}")
    assert len(expanded) == 200


def test_creator_normalizers_tolerate_null_author():
    downloader = object.__new__(ZhihuCreatorDownloader)
    downloader.url_token = "tester"

    answer = downloader._normalize_creator_answer(
        {"id": 1, "author": None, "question": None}
    )
    assert answer["author"] == "未知作者"
    assert answer["title"] == "未知问题"

    article = downloader._normalize_creator_article({"id": 2, "author": None})
    assert article["author"] == "未知作者"
    assert article["title"] == "未知专栏标题"


def test_image_map_matches_size_and_host_variants():
    html = (
        '<p><img src="https://pica.zhimg.com/50/v2-abc_720w.jpg?source=2" alt=""></p>'
    )
    converter = ZhihuConverter(
        img_map={"https://picx.zhimg.com/v2-abc_r.jpg?source=1": "images/abc.jpg"}
    )
    markdown = converter.convert(html)

    assert "images/abc.jpg" in markdown
    assert "pica.zhimg.com" not in markdown


def test_fetch_overview_keeps_unknown_totals(report_module, monkeypatch):
    class FakeClient:
        def get_question_answers_page(self, *args, **kwargs):
            return {"data": [{"question": {"title": "测试问题"}}], "paging": {}}

    monkeypatch.setattr(report_module, "ZhihuAPIClient", lambda: FakeClient())
    prepared = report_module.PreparedSource(
        mode="url",
        source="https://www.zhihu.com/question/1",
        question_id="1",
        question_url="https://www.zhihu.com/question/1",
    )
    result = report_module.fetch_question_overview(prepared)

    assert result.question_title == "测试问题"
    assert result.total_answers is None


def test_fetch_api_retries_403_once(monkeypatch):
    client = api_client.ZhihuAPIClient()

    class Response:
        status_code = 403
        text = ""

    calls = []

    class Session:
        def get(self, *args, **kwargs):
            calls.append(1)
            return Response()

    client.session = Session()
    client._get_signature = lambda path: {}
    client._init_session = lambda: None
    monkeypatch.setattr(api_client.cookie_manager, "rotate_session", lambda: None)

    with pytest.raises(Exception, match="403"):
        client.fetch_api("/api/v4/test")

    assert len(calls) == 2


def test_fetch_api_reports_server_errors(monkeypatch):
    client = api_client.ZhihuAPIClient()

    class Response:
        status_code = 503
        text = ""

    class Session:
        def get(self, *args, **kwargs):
            return Response()

    client.session = Session()
    client._get_signature = lambda path: {}

    with pytest.raises(Exception, match="503"):
        client.fetch_api("/api/v4/test")


def test_cookie_session_requires_z_c0():
    assert cookie_manager._is_valid_session({"z_c0": "token"}) is True
    assert cookie_manager._is_valid_session({"d_c0": "device"}) is False


def test_detect_dns_servers_skips_windows(monkeypatch):
    monkeypatch.delenv("ZHIHU_DNS_SERVERS", raising=False)
    monkeypatch.delenv("ZHIHU_DISABLE_DNS_FIX", raising=False)
    monkeypatch.setattr(api_client.sys, "platform", "win32")
    assert api_client.detect_dns_servers() is None

    monkeypatch.setenv("ZHIHU_DNS_SERVERS", "1.1.1.1")
    assert api_client.detect_dns_servers() == "1.1.1.1"


def test_cli_rejects_invalid_numeric_flags(run_cli):
    for flag, value in (
        ("--answer-cap", "0"),
        ("--top-k", "0"),
        ("--lda-words", "0"),
        ("--lda-max-iter", "0"),
        ("--network-max-nodes", "-1"),
    ):
        result = run_cli("some-source", flag, value)
        assert result.returncode == 2, f"{flag} should be rejected"
        assert "argument" in result.stderr.lower()


def test_cli_rejects_missing_stopwords_file(run_cli, repo_root):
    result = run_cli(
        repo_root / "tests" / "fixtures" / "sample_raw",
        "--stopwords",
        "/tmp/definitely-missing-stopwords.txt",
    )
    assert result.returncode != 0
    assert "停用词" in (result.stdout + result.stderr)


def test_dashboard_js_escapes_tooltip_fields(report_module, tmp_path):
    summary = {
        "question_title": "测试",
        "generated_at": "2026-01-01T00:00:00",
        "answer_count": 1,
        "author_count": 1,
        "total_characters": 10,
        "average_characters": 10,
        "median_characters": 10,
        "max_characters": 10,
        "min_characters": 10,
        "average_sentiment": 0.5,
        "sentiment_stddev": 0,
        "sentiment_labels": {"positive": 0, "neutral": 1, "negative": 0},
        "sentiment_buckets": {"0.4-0.6": 1},
        "top_keywords": [{"word": "<img onerror=alert(1)>", "count": 1}],
        "lda_topics": [],
        "author_network": {"nodes": [], "links": [], "categories": []},
        "author_network_highlights": [],
        "timeline": [{"date": "2026-01-01", "count": 1}],
        "top_authors": [{"author": "<b>x</b>", "count": 1}],
        "top_answers": [],
    }
    dashboard = tmp_path / "dashboard.html"
    report_module.write_dashboard(dashboard, summary, [])
    html = dashboard.read_text(encoding="utf-8")

    assert "function escapeHtml(value)" in html
    assert html.count("escapeHtml(") >= 8
    # The payload may contain the raw string (it is inert JSON), but it must not
    # appear as markup or JS source outside the JSON block.
    payload_region = html.split('<script id="report-data" type="application/json">', 1)[1].split(
        "</script>", 1
    )[0]
    outside_payload = html.replace(payload_region, "")
    assert "<img onerror=alert(1)>" not in outside_payload
