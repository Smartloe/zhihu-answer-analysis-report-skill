"""Live network smoke test.

Disabled by default. Enable with::

    RUN_NETWORK_TESTS=1 python3 -m pytest tests/test_network.py -q

The test intentionally fetches a single answer without images so it is as light
as the real API allows.
"""

from __future__ import annotations

import json

import pytest

QUESTION_URL = "https://www.zhihu.com/question/2010761296434458950"


@pytest.mark.network
def test_url_mode_fetches_single_answer(run_cli, tmp_path):
    output = tmp_path / "network-bundle"

    result = run_cli(
        QUESTION_URL,
        "--answer-cap",
        "1",
        "--no-images",
        "--output-dir",
        output,
        timeout=600,
    )

    assert result.returncode == 0, (
        f"URL mode failed\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    summary = json.loads(
        (output / "analysis" / "summary.json").read_text(encoding="utf-8")
    )
    assert summary["answer_count"] >= 1
    assert summary["question_id"] == "2010761296434458950"
    assert (output / "report.md").is_file()
    assert (output / "analysis" / "dashboard.html").is_file()
    assert list((output / "raw" / "entries").rglob("index.md"))
