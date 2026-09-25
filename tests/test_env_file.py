"""Reading .env: convenience that must never override what the caller actually exported."""

import os

import pytest

from rootlogic.cli import load_env_file


@pytest.fixture(autouse=True)
def restore_environment():
    """``load_env_file`` writes to ``os.environ`` for real, which is the point of it. Without
    this, a variable set from a fixture file outlived its test: ROOTLOGIC_HOME leaked, and
    every later test that ran the CLI wrote its reports into the directory this test named."""
    before = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(before)


def write(tmp_path, text):
    f = tmp_path / ".env"
    f.write_text(text)
    return f


def test_a_dotenv_fills_in_what_the_shell_is_missing(tmp_path, monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    named = load_env_file(write(tmp_path, "TAVILY_API_KEY=tvly-from-file\n"))
    assert os.environ["TAVILY_API_KEY"] == "tvly-from-file"
    assert named == ["TAVILY_API_KEY"]


def test_an_exported_value_always_wins(tmp_path, monkeypatch):
    """`TAVILY_API_KEY=... rootlogic research ...` has to keep meaning what it says: a file
    that silently beat an explicit export would be a nasty surprise to debug."""
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-from-shell")
    load_env_file(write(tmp_path, "TAVILY_API_KEY=tvly-from-file\n"))
    assert os.environ["TAVILY_API_KEY"] == "tvly-from-shell"


def test_no_file_is_not_an_event(tmp_path):
    assert load_env_file(tmp_path / "absent.env") == []


def test_a_broken_file_does_not_end_the_run(tmp_path, monkeypatch):
    """The variables may well be exported anyway; a malformed convenience file is not a
    reason to refuse to research anything."""
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    assert load_env_file(write(tmp_path, "\x00\x00 not = a { valid } file\n")) == []


def test_a_commented_out_variable_is_not_reported_as_set(tmp_path, monkeypatch):
    monkeypatch.delenv("ROOTLOGIC_HOME", raising=False)
    named = load_env_file(write(tmp_path, "# ROOTLOGIC_HOME=.rootlogic\nTAVILY_API_KEY=x\n"))
    assert named == ["TAVILY_API_KEY"]


def test_rootlogic_home_from_the_file_is_honoured(tmp_path, monkeypatch):
    """It was not: HOME was a module constant read at import, so a .env loaded in main() said
    one directory while the run used another. Caught by running the CLI in a scratch dir."""
    from rootlogic.cli import data_home

    monkeypatch.delenv("ROOTLOGIC_HOME", raising=False)
    assert data_home().name == ".rootlogic"

    load_env_file(write(tmp_path, "ROOTLOGIC_HOME=.rootlogic-elsewhere\n"))
    assert data_home().name == ".rootlogic-elsewhere"
