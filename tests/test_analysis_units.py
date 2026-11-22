"""Unit tests for tokenisation, sentiment, LDA, author network and summary."""

from __future__ import annotations

import statistics
from collections import Counter
from pathlib import Path

import pytest

SHARED_TEXT = "乒乓球 央视 樊振东 德甲 体育 中国 德国 国际"


def _record(author: str, text: str, **overrides):
    record = {
        "author": author,
        "text": text,
        "char_count": len(text),
        "token_count": 0,
        "upvotes": 0,
        "date": "2026-01-01",
        "title": "T",
        "url": "https://example.com",
        "markdown_path": "/tmp/index.md",
    }
    record.update(overrides)
    return record


# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------


def test_tokenize_text_applies_stopwords_and_min_length(report_module, analysis_deps):
    jieba = analysis_deps["jieba"]

    tokens = report_module.tokenize_text(
        "乒乓球 的 樊振东 a 好 央视 2026",
        jieba,
        {"的", "樊振东"},
        min_word_length=2,
    )

    assert "乒乓球" in tokens
    assert "央视" in tokens
    assert "的" not in tokens
    assert "樊振东" not in tokens
    assert "a" not in tokens  # too short
    assert "好" not in tokens  # too short
    assert "2026" not in tokens  # pure digits are dropped


def test_tokenize_text_min_word_length_one_keeps_single_char_words(
    report_module, analysis_deps
):
    jieba = analysis_deps["jieba"]

    tokens = report_module.tokenize_text(
        "好 的 乒乓球", jieba, {"的"}, min_word_length=1
    )

    assert "好" in tokens
    assert "的" not in tokens


def test_tokenize_text_drops_punctuation_only_tokens(report_module, analysis_deps):
    jieba = analysis_deps["jieba"]

    tokens = report_module.tokenize_text("！！！ ... ——", jieba, set(), 1)

    assert tokens == []


def test_tokenize_records_counts_and_annotates_records(report_module, analysis_deps):
    jieba = analysis_deps["jieba"]
    records = [
        {"text": "乒乓球 乒乓球 央视"},
        {"text": ""},
        {"text": "央视 德甲"},
    ]

    frequencies = report_module.tokenize_records(records, jieba, set(), 2)

    assert frequencies["乒乓球"] == 2
    assert frequencies["央视"] == 2
    assert frequencies["德甲"] == 1
    assert records[0]["char_count"] == len("乒乓球 乒乓球 央视")
    assert records[0]["token_count"] == 3
    assert records[1]["char_count"] == 0
    assert records[1]["token_count"] == 0
    assert records[2]["token_count"] == 2


# ---------------------------------------------------------------------------
# Sentiment
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("score", "expected"),
    [
        (None, "unknown"),
        (0.0, "0.0-0.2"),
        (0.1999, "0.0-0.2"),
        (0.2, "0.2-0.4"),
        (0.3999, "0.2-0.4"),
        (0.4, "0.4-0.6"),
        (0.5, "0.4-0.6"),
        (0.5999, "0.4-0.6"),
        (0.6, "0.6-0.8"),
        (0.7999, "0.6-0.8"),
        (0.8, "0.8-1.0"),
        (1.0, "0.8-1.0"),
        (1.2, "0.8-1.0"),
    ],
)
def test_sentiment_bucket_boundaries(report_module, score, expected):
    assert report_module.sentiment_bucket(score) == expected


@pytest.mark.parametrize(
    ("score", "expected"),
    [
        (None, "unknown"),
        (0.0, "negative"),
        (0.2, "negative"),
        (0.3999, "negative"),
        (0.4, "neutral"),
        (0.5, "neutral"),
        (0.6, "neutral"),
        (0.6001, "positive"),
        (0.8, "positive"),
        (1.0, "positive"),
    ],
)
def test_sentiment_label_boundaries(report_module, score, expected):
    assert report_module.sentiment_label(score) == expected


def test_score_sentiment_uses_snownlp_and_handles_empty_text(report_module):
    class FakeSnowNLP:
        def __init__(self, text):
            self.text = text

        @property
        def sentiments(self):
            return 0.75 if self.text.strip() else 0.0

    records = [{"text": "正面内容"}, {"text": ""}]

    report_module.score_sentiment(records, FakeSnowNLP)

    assert records[0]["sentiment"] == 0.75
    assert records[1]["sentiment"] is None


def test_score_sentiment_swallows_snownlp_errors(report_module):
    class ExplodingSnowNLP:
        def __init__(self, text):
            raise ValueError("boom")

    records = [{"text": "任意内容"}]

    report_module.score_sentiment(records, ExplodingSnowNLP)

    assert records[0]["sentiment"] is None


# ---------------------------------------------------------------------------
# LDA topic modelling
# ---------------------------------------------------------------------------


LDA_VOCAB = [
    "乒乓球",
    "央视",
    "樊振东",
    "德甲",
    "体育",
    "中国",
    "德国",
    "国际",
    "比赛",
    "运动员",
    "训练",
    "球迷",
]


def _lda_records(count: int) -> list[dict]:
    offsets = (0, 1, 2, 3, 5, 7)
    records = []
    for index in range(count):
        words = [LDA_VOCAB[(index + offset) % len(LDA_VOCAB)] for offset in offsets]
        records.append({"text": " ".join(words * 3), "author": f"作者{index}"})
    return records


def test_run_lda_topics_disabled(report_module, analysis_deps):
    topics, distribution, note = report_module.run_lda_topics(
        records=[],
        jieba=analysis_deps["jieba"],
        stopwords=set(),
        min_word_length=2,
        topic_count=0,
        top_words=5,
        max_iter=5,
        LDA=analysis_deps["LDA"],
        CountVectorizer=analysis_deps["CountVectorizer"],
    )

    assert topics == []
    assert distribution == []
    assert "LDA" in note
    assert "关闭" in note


def test_run_lda_topics_insufficient_documents(report_module, analysis_deps):
    topics, distribution, note = report_module.run_lda_topics(
        records=_lda_records(3),
        jieba=analysis_deps["jieba"],
        stopwords=set(),
        min_word_length=2,
        topic_count=2,
        top_words=5,
        max_iter=5,
        LDA=analysis_deps["LDA"],
        CountVectorizer=analysis_deps["CountVectorizer"],
    )

    assert topics == []
    assert distribution == []
    assert "至少需要" in note
    assert "LDA" in note


def test_run_lda_topics_insufficient_vocabulary(report_module, analysis_deps):
    # ``alpha`` appears in every document and is pruned by max_df=0.95, leaving
    # only two usable terms, which must trigger the vocabulary guard.
    records = [{"text": "alpha beta gamma"}] + [{"text": "alpha"} for _ in range(7)]

    topics, distribution, note = report_module.run_lda_topics(
        records=records,
        jieba=analysis_deps["jieba"],
        stopwords=set(),
        min_word_length=2,
        topic_count=2,
        top_words=5,
        max_iter=5,
        LDA=analysis_deps["LDA"],
        CountVectorizer=analysis_deps["CountVectorizer"],
    )

    assert topics == []
    assert distribution == []
    assert "词汇" in note


def test_run_lda_topics_degenerate_vocabulary_returns_note(
    report_module, analysis_deps
):
    records = [{"text": "乒乓球 乒乓球"} for _ in range(8)]

    topics, distribution, note = report_module.run_lda_topics(
        records=records,
        jieba=analysis_deps["jieba"],
        stopwords=set(),
        min_word_length=2,
        topic_count=2,
        top_words=5,
        max_iter=5,
        LDA=analysis_deps["LDA"],
        CountVectorizer=analysis_deps["CountVectorizer"],
    )

    assert topics == []
    assert distribution == []
    assert "词汇" in note


def test_run_lda_topics_multiple_documents(report_module, analysis_deps):
    records = _lda_records(20)

    topics, distribution, note = report_module.run_lda_topics(
        records=records,
        jieba=analysis_deps["jieba"],
        stopwords=set(),
        min_word_length=2,
        topic_count=3,
        top_words=4,
        max_iter=10,
        LDA=analysis_deps["LDA"],
        CountVectorizer=analysis_deps["CountVectorizer"],
    )

    assert note is None
    assert len(topics) == 3
    assert len(distribution) == 3
    assert sum(topic["doc_count"] for topic in topics) == 20
    assert sum(topic["doc_share"] for topic in topics) == pytest.approx(1.0, abs=1e-3)
    assert all(topic["top_words"] for topic in topics)
    assert all(
        word in LDA_VOCAB for topic in topics for word in topic["top_words"]
    )
    assert all(set(topic) == {"topic_id", "top_words", "doc_count", "doc_share"} for topic in topics)
    assert all(set(item) == {"topic", "count", "keywords"} for item in distribution)
    assert all(1 <= record["topic_id"] <= 3 for record in records)
    assert all(0.0 < record["topic_score"] <= 1.0 for record in records)


def test_run_lda_topics_clamps_topic_count_to_available_data(
    report_module, analysis_deps
):
    # 20 documents keep the ``topic_count * 2`` document gate satisfied, while
    # the 8-word vocabulary forces the requested 10 topics down to 8.
    vocab = LDA_VOCAB[:8]
    offsets = (0, 1, 2, 3, 5, 7)
    records = [
        {
            "text": " ".join(
                [vocab[(index + offset) % len(vocab)] for offset in offsets] * 3
            )
        }
        for index in range(20)
    ]

    topics, _distribution, note = report_module.run_lda_topics(
        records=records,
        jieba=analysis_deps["jieba"],
        stopwords=set(),
        min_word_length=2,
        topic_count=10,
        top_words=3,
        max_iter=5,
        LDA=analysis_deps["LDA"],
        CountVectorizer=analysis_deps["CountVectorizer"],
    )

    assert note is None
    assert len(topics) == len(vocab)
    assert len(topics) == len({topic["topic_id"] for topic in topics})
    assert all(1 <= record["topic_id"] <= len(vocab) for record in records)


# ---------------------------------------------------------------------------
# Author network
# ---------------------------------------------------------------------------


def _network(report_module, analysis_deps, records, *, max_nodes=18, max_edges=28):
    return report_module.build_author_network(
        records=records,
        jieba=analysis_deps["jieba"],
        stopwords=set(),
        min_word_length=2,
        lda_topics=None,
        max_nodes=max_nodes,
        max_edges=max_edges,
    )


def test_author_network_disabled(report_module, analysis_deps):
    records = [
        _record("甲", SHARED_TEXT + " alpha", upvotes=10),
        _record("乙", SHARED_TEXT + " beta", upvotes=5),
        _record("丙", SHARED_TEXT + " gamma", upvotes=1),
    ]

    network, highlights, note = _network(
        report_module, analysis_deps, records, max_nodes=0
    )

    assert network == {"nodes": [], "links": [], "categories": []}
    assert highlights == []
    assert "作者" in note
    assert "关闭" in note


def test_author_network_empty_records(report_module, analysis_deps):
    network, highlights, note = _network(report_module, analysis_deps, [])

    assert network["nodes"] == []
    assert network["links"] == []
    assert highlights == []
    assert "作者" in note
    assert "3" in note


def test_author_network_single_author(report_module, analysis_deps):
    records = [
        _record("甲", SHARED_TEXT + " alpha", upvotes=10),
        _record("甲", SHARED_TEXT + " beta", upvotes=5),
        _record("甲", SHARED_TEXT + " gamma", upvotes=1),
    ]

    network, highlights, note = _network(report_module, analysis_deps, records)

    assert network["nodes"] == []
    assert network["links"] == []
    assert highlights == []
    assert "作者" in note
    assert "3" in note


def test_author_network_builds_nodes_and_links(report_module, analysis_deps):
    records = [
        _record("甲", SHARED_TEXT + " alpha", upvotes=10),
        _record("乙", SHARED_TEXT + " beta", upvotes=5),
        _record("丙", SHARED_TEXT + " gamma", upvotes=1),
    ]

    network, highlights, note = _network(report_module, analysis_deps, records)

    node_ids = {node["id"] for node in network["nodes"]}
    assert node_ids == {"甲", "乙", "丙"}
    assert len(network["links"]) == 3
    assert network["categories"]
    for link in network["links"]:
        assert link["source"] in node_ids
        assert link["target"] in node_ids
        assert link["value"] >= 0.2
        assert link["shared_keywords"]
        assert link["relation"]
        assert link["lineStyle"]["width"] >= 1.2
    for node in network["nodes"]:
        assert 0 <= node["category"] < len(network["categories"])
        assert node["symbolSize"] > 0
        assert node["answers"] >= 1
    assert 1 <= len(highlights) <= 6
    assert "推断" in note


def test_author_network_skips_weak_links(report_module, analysis_deps):
    records = [
        _record("甲", "苹果 香蕉 葡萄", upvotes=10),
        _record("乙", "汽车 轮胎 发动机", upvotes=5),
        _record("丙", "足球 篮球 排球", upvotes=1),
    ]

    network, highlights, note = _network(report_module, analysis_deps, records)

    assert network["nodes"] == []
    assert network["links"] == []
    assert highlights == []
    assert "作者" in note
    assert "关联" in note


def test_author_network_respects_max_nodes(report_module, analysis_deps):
    records = [
        _record("甲", SHARED_TEXT + " alpha", upvotes=40),
        _record("乙", SHARED_TEXT + " beta", upvotes=30),
        _record("丙", SHARED_TEXT + " gamma", upvotes=20),
        _record("丁", SHARED_TEXT + " delta", upvotes=10),
    ]

    network, _highlights, _note = _network(
        report_module, analysis_deps, records, max_nodes=3
    )

    node_ids = {node["id"] for node in network["nodes"]}
    assert node_ids == {"甲", "乙", "丙"}
    assert "丁" not in node_ids
    assert len(network["links"]) <= 28


def test_author_network_respects_max_edges(report_module, analysis_deps):
    records = [
        _record("甲", SHARED_TEXT + " alpha", upvotes=40),
        _record("乙", SHARED_TEXT + " beta", upvotes=30),
        _record("丙", SHARED_TEXT + " gamma", upvotes=20),
        _record("丁", SHARED_TEXT + " delta", upvotes=10),
    ]

    network, _highlights, _note = _network(
        report_module, analysis_deps, records, max_edges=2
    )

    node_ids = {node["id"] for node in network["nodes"]}
    assert len(network["links"]) == 2
    for link in network["links"]:
        assert link["source"] in node_ids
        assert link["target"] in node_ids


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def test_build_summary_computes_all_metrics(report_module, tmp_path):
    prepared = report_module.PreparedSource(
        mode="path",
        source=str(tmp_path),
        path=tmp_path,
        question_title="测试问题",
    )
    records = [
        _record(
            "甲",
            "正文",
            char_count=10,
            token_count=2,
            sentiment=0.1,
            upvotes=5,
            date="2026-01-02",
            title="A",
        ),
        _record(
            "乙",
            "正文",
            char_count=20,
            token_count=4,
            sentiment=0.8,
            upvotes=3,
            date="2026-01-01",
            title="B",
        ),
        _record(
            "甲",
            "正文",
            char_count=30,
            token_count=6,
            sentiment=None,
            upvotes=None,
            date="",
            title="C",
        ),
    ]
    frequencies = Counter({"词": 5, "语": 2})

    summary = report_module.build_summary(
        records, frequencies, prepared, top_k=1
    )

    assert summary["answer_count"] == 3
    assert summary["author_count"] == 2
    assert summary["total_characters"] == 60
    assert summary["average_characters"] == 20.0
    assert summary["median_characters"] == 20
    assert summary["average_sentiment"] == pytest.approx(0.45)
    assert summary["sentiment_stddev"] == pytest.approx(
        round(statistics.pstdev([0.1, 0.8]), 4)
    )
    assert summary["sentiment_labels"] == {
        "negative": 1,
        "positive": 1,
        "unknown": 1,
    }
    assert summary["sentiment_buckets"] == {
        "0.0-0.2": 1,
        "0.8-1.0": 1,
        "unknown": 1,
    }
    assert summary["top_keywords"] == [{"word": "词", "count": 5}]
    assert summary["date_range"] == {"start": "2026-01-01", "end": "2026-01-02"}
    assert summary["timeline"] == [
        {"date": "2026-01-01", "count": 1},
        {"date": "2026-01-02", "count": 1},
        {"date": "unknown", "count": 1},
    ]
    assert summary["top_authors"][0] == {"author": "甲", "count": 2}
    assert [item["title"] for item in summary["top_answers"]] == ["A", "B"]
    assert summary["lda_topics"] == []
    assert summary["author_network"] == {
        "nodes": [],
        "links": [],
        "categories": [],
    }


def test_build_summary_handles_records_without_upvotes(report_module, tmp_path):
    prepared = report_module.PreparedSource(mode="path", source="src", path=tmp_path)
    records = [
        _record("甲", "短", char_count=1, token_count=1, upvotes=None),
        _record("乙", "更长的正文", char_count=5, token_count=2, upvotes=None),
    ]

    summary = report_module.build_summary(
        records, Counter(), prepared, top_k=5
    )

    assert summary["average_sentiment"] is None
    assert summary["sentiment_stddev"] == 0
    assert summary["sentiment_labels"] == {"unknown": 2}
    assert [item["author"] for item in summary["top_answers"]] == ["乙", "甲"]
    assert summary["date_range"] == {"start": "2026-01-01", "end": "2026-01-01"}
