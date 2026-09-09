# SPDX-License-Identifier: Apache-2.0
import json
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
    api = Mock(side_effect=[{"permission": "maintain"}, pr, {"id": 99}, {}])
    monkeypatch.setattr(parity, "github_api", api)
    target = parity.start()
    assert target["repository"] == "contributor/fork"
    assert target["sha"] == "pr-sha"
    assert api.call_args.kwargs["data"]["head_sha"] == "pr-sha"

    pr["head"]["sha"] = "newer-pr-sha"
    parity.finish(target["check_id"], "failure")
    assert api.call_args.args == ("check-runs/99",)
    assert api.call_args.kwargs["data"]["conclusion"] == "failure"


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
