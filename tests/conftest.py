"""Shared fixtures for the zhihu answer report smoke test suite.

The suite intentionally treats ``scripts/zhihu_answer_report.py`` as a black
box module: it is loaded by path so the tests do not mutate the source tree and
do not depend on ``scripts`` being an importable package.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "zhihu_answer_report.py"
SAMPLE_ROOT = REPO_ROOT / "data" / "reports" / "standalone-smoke-test"
SAMPLE_RAW = SAMPLE_ROOT / "raw"
SAMPLE_JSONL = SAMPLE_ROOT / "analysis" / "answers.jsonl"


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "network: tests that talk to the live Zhihu API (disabled by default)",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if os.environ.get("RUN_NETWORK_TESTS") == "1":
        return

    skip_marker = pytest.mark.skip(
        reason="network test disabled; set RUN_NETWORK_TESTS=1 to enable it"
    )
    for item in items:
        if item.get_closest_marker("network"):
            item.add_marker(skip_marker)


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def script_path() -> Path:
    return SCRIPT_PATH


@pytest.fixture(scope="session")
def report_module():
    """Load the CLI script as a module without touching ``sys.path`` hooks twice."""
    spec = importlib.util.spec_from_file_location(
        "zhihu_answer_report_under_test", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def analysis_deps(report_module):
    """Return the heavy analysis dependencies resolved exactly like the CLI."""
    jieba, SnowNLP, WordCloud, LDA, CountVectorizer = (
        report_module.load_analysis_dependencies()
    )
    return {
        "jieba": jieba,
        "SnowNLP": SnowNLP,
        "WordCloud": WordCloud,
        "LDA": LDA,
        "CountVectorizer": CountVectorizer,
    }


@pytest.fixture(scope="session")
def font_path(report_module) -> str:
    return report_module.resolve_wordcloud_font(None)


@pytest.fixture(scope="session")
def stopwords(report_module) -> set[str]:
    return report_module.load_stopwords(None)


@pytest.fixture(scope="session")
def sample_raw() -> Path:
    if not SAMPLE_RAW.is_dir():
        pytest.skip(f"sample raw directory is missing: {SAMPLE_RAW}")
    return SAMPLE_RAW


@pytest.fixture(scope="session")
def sample_jsonl() -> Path:
    if not SAMPLE_JSONL.is_file():
        pytest.skip(f"sample answers.jsonl is missing: {SAMPLE_JSONL}")
    return SAMPLE_JSONL


@pytest.fixture(scope="session")
def run_cli():
    """Run the CLI in a subprocess from the repository root."""

    def _run(*args: object, timeout: int = 600) -> subprocess.CompletedProcess[str]:
        command = [sys.executable, str(SCRIPT_PATH), *[str(arg) for arg in args]]
        return subprocess.run(
            command,
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=timeout,
        )

    return _run


def _assert_run_ok(result: subprocess.CompletedProcess[str]) -> None:
    assert result.returncode == 0, (
        f"CLI failed with exit code {result.returncode}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )


@pytest.fixture(scope="session")
def path_bundle(tmp_path_factory, run_cli, sample_raw) -> Path:
    """Full default pipeline over the bundled raw/ Markdown scrape."""
    output = tmp_path_factory.mktemp("path-mode-bundle")
    result = run_cli(sample_raw, "--output-dir", output)
    _assert_run_ok(result)
    return output


@pytest.fixture(scope="session")
def jsonl_bundle(tmp_path_factory, run_cli, sample_jsonl) -> Path:
    """Full default pipeline over an existing answers.jsonl dataset."""
    output = tmp_path_factory.mktemp("jsonl-mode-bundle")
    result = run_cli(sample_jsonl, "--output-dir", output)
    _assert_run_ok(result)
    return output


@pytest.fixture(scope="session")
def flag_bundle(tmp_path_factory, run_cli, sample_raw, font_path) -> dict[str, object]:
    """Pipeline run with every optional CLI flag that changes analysis output."""
    root = tmp_path_factory.mktemp("flags-mode")
    extra_stopwords = root / "extra-stopwords.txt"
    extra_stopwords.write_text("樊振东\n", encoding="utf-8")
    output = root / "bundle"
    result = run_cli(
        sample_raw,
        "--output-dir",
        output,
        "--lda-topics",
        "0",
        "--network-max-nodes",
        "0",
        "--stopwords",
        extra_stopwords,
        "--font-path",
        font_path,
    )
    _assert_run_ok(result)
    return {
        "output": output,
        "stopwords": extra_stopwords,
        "font_path": font_path,
        "result": result,
    }


def load_summary(bundle: Path) -> dict:
    return json.loads((bundle / "analysis" / "summary.json").read_text(encoding="utf-8"))
