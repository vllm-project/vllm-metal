# SPDX-License-Identifier: Apache-2.0
import json
import urllib.error
from unittest.mock import Mock

import parity
import pytest


@pytest.fixture
def event(monkeypatch, tmp_path):
    payload = {
        "issue": {"number": 42, "pull_request": {}},
        "comment": {
            "body": "/ci parity",
            "user": {"login": "reviewer"},
            "author_association": "MEMBER",
        },
    }
    event_path = tmp_path / "event.json"
    event_path.write_text(json.dumps(payload))
    for key, value in {
        "GITHUB_REPOSITORY": parity.REPOSITORY,
        "GITHUB_EVENT_NAME": "issue_comment",
        "GITHUB_EVENT_PATH": str(event_path),
        "GITHUB_SHA": "main-sha",
        "GITHUB_SERVER_URL": "https://github.com",
        "GITHUB_RUN_ID": "123",
        "GITHUB_STEP_SUMMARY": str(tmp_path / "summary.md"),
        "PARITY_TESTED_SHA": "pr-sha",
    }.items():
        monkeypatch.setenv(key, value)


def test_read_only_user_cannot_launch_pr_code(event, monkeypatch):
    api = Mock(return_value={"permission": "read"})
    monkeypatch.setattr(parity, "github_api", api)
    assert parity.start() is None
    api.assert_called_once_with("collaborators/reviewer/permission")


def test_captured_fork_sha_and_original_check_are_used(event, monkeypatch):
    pr = {
        "state": "open",
        "head": {"sha": "pr-sha", "repo": {"full_name": "contributor/fork"}},
    }
    api = Mock(side_effect=[{"permission": "maintain"}, pr, {"id": 99}, pr, {}])
    monkeypatch.setattr(parity, "github_api", api)
    target = parity.start()
    assert target["repository"] == "contributor/fork"
    assert target["sha"] == "pr-sha"
    assert api.call_args.kwargs["data"]["head_sha"] == "pr-sha"

    pr["head"]["sha"] = "newer-pr-sha"
    parity.finish(target["check_id"], "failure")
    assert api.call_args.args == ("check-runs/99",)
    assert api.call_args.kwargs["data"]["conclusion"] == "failure"


@pytest.mark.parametrize(
    ("head", "expected"),
    [
        ({"head": {"sha": "a" * 40}}, "Up to date at report time."),
        ({"head": {"sha": "b" * 40}}, "Outdated"),
        (urllib.error.URLError("unavailable"), "Freshness unknown"),
    ],
)
def test_report_preserves_result_and_shows_freshness(
    event, monkeypatch, tmp_path, head, expected
):
    monkeypatch.setenv("PARITY_TESTED_SHA", "a" * 40)
    api = Mock(side_effect=[head, {}])
    monkeypatch.setattr(parity, "github_api", api)
    parity.finish("99", "success")
    assert api.call_args_list[0].args == ("pulls/42",)
    assert api.call_args.args == ("check-runs/99",)
    data = api.call_args.kwargs["data"]
    assert data["conclusion"] == "success"
    summary = data["output"]["summary"]
    assert f"Tested commit: `{'a' * 40}`" in summary
    assert expected in summary
    if isinstance(head, dict):
        assert f"Current PR head: `{head['head']['sha']}`" in summary
    assert summary in (tmp_path / "summary.md").read_text()


def test_nightly_report_does_not_fetch_pr_head(event, monkeypatch):
    monkeypatch.setenv("GITHUB_EVENT_NAME", "schedule")
    monkeypatch.setenv("PARITY_TESTED_SHA", "main-sha")
    api = Mock()
    monkeypatch.setattr(parity, "github_api", api)
    parity.finish("99", "success")
    assert api.call_count == 1
    assert api.call_args.args == ("check-runs/99",)
    assert (
        "Tested commit: `main-sha`" in api.call_args.kwargs["data"]["output"]["summary"]
    )


def test_nightly_uses_event_sha_and_does_not_run_in_forks(event, monkeypatch):
    api = Mock()
    monkeypatch.setattr(parity, "github_api", api)
    monkeypatch.setenv("GITHUB_EVENT_NAME", "schedule")
    assert parity.resolve_target() == {
        "repository": parity.REPOSITORY,
        "sha": "main-sha",
        "key": "nightly",
    }
    monkeypatch.setenv("GITHUB_REPOSITORY", "contributor/fork")
    assert parity.start() is None
    api.assert_not_called()
