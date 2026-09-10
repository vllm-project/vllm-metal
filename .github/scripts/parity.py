# SPDX-License-Identifier: Apache-2.0
"""Authorize parity runs and report checks; never execute PR code here."""

import argparse
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

REPOSITORY = "vllm-project/vllm-metal"


def workflow_url() -> str:
    return (
        f"{os.environ['GITHUB_SERVER_URL']}/{REPOSITORY}"
        f"/actions/runs/{os.environ['GITHUB_RUN_ID']}"
    )


def github_api(path: str, *, method: str = "GET", data: dict | None = None) -> dict:
    request = urllib.request.Request(
        f"{os.environ['GITHUB_API_URL']}/repos/{REPOSITORY}/{path}",
        data=json.dumps(data).encode() if data is not None else None,
        headers={
            "Authorization": f"Bearer {os.environ['GH_TOKEN']}",
            "Accept": "application/vnd.github+json",
            "Content-Type": "application/json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
        method=method,
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def resolve_target() -> dict | None:
    if os.environ["GITHUB_REPOSITORY"] != REPOSITORY:
        return None
    event_name = os.environ["GITHUB_EVENT_NAME"]
    if event_name == "schedule":
        return {
            "repository": REPOSITORY,
            "sha": os.environ["GITHUB_SHA"],
            "key": "nightly",
        }
    event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())
    if (
        event_name != "issue_comment"
        or event.get("issue", {}).get("pull_request") is None
        or event.get("comment", {}).get("body") != "/ci parity"
    ):
        return None
    username = urllib.parse.quote(event["comment"]["user"]["login"], safe="")
    try:
        permission = github_api(f"collaborators/{username}/permission")["permission"]
    except urllib.error.HTTPError as error:
        if error.code in (403, 404):
            return None
        raise
    if permission not in {"admin", "maintain", "write"}:
        return None
    number = int(event["issue"]["number"])
    pr = github_api(f"pulls/{number}")
    if pr["state"] != "open" or not pr["head"]["repo"]:
        return None
    return {
        "repository": pr["head"]["repo"]["full_name"],
        "sha": pr["head"]["sha"],
        "key": f"pr-{number}",
    }


def start() -> dict | None:
    target = resolve_target()
    if target is None:
        return None
    check = github_api(
        "check-runs",
        method="POST",
        data={
            "name": "Parity",
            "head_sha": target["sha"],
            "status": "in_progress",
            "details_url": workflow_url(),
            "output": {
                "title": "Parity matrix requested",
                "summary": f"Testing {target['repository']}@{target['sha']}. "
                f"[View workflow run]({workflow_url()}) for results and logs.",
            },
        },
    )
    return {**target, "check_id": str(check["id"])}


def finish(check_id: str, result: str) -> None:
    conclusion = result if result in {"success", "cancelled"} else "failure"
    tested_sha = os.environ["PARITY_TESTED_SHA"]
    summary = f"Tested commit: `{tested_sha}`.\n\n"
    if os.environ["GITHUB_EVENT_NAME"] == "issue_comment":
        event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())
        number = int(event["issue"]["number"])
        try:
            head_sha = github_api(f"pulls/{number}")["head"]["sha"]
            freshness = (
                "Up to date at report time."
                if head_sha == tested_sha
                else "Outdated — the current PR head was not tested by this run. "
                "Comment `/ci parity` to test the current head."
            )
            summary += f"Current PR head: `{head_sha}`.\n\n{freshness}\n\n"
        except (OSError, ValueError, KeyError) as error:
            print(f"::warning::Could not read current PR head: {error}")
            summary += "Freshness unknown: could not read the current PR head.\n\n"
    summary += (
        f"[View workflow run]({workflow_url()}) for per-model "
        "summaries of EXACT, TOP_K_MATCH, and FAIL. Download the parity "
        "artifacts for per-batch logs and environment versions."
    )
    github_api(
        f"check-runs/{int(check_id)}",
        method="PATCH",
        data={
            "status": "completed",
            "conclusion": conclusion,
            "output": {
                "title": f"Parity matrix: {conclusion}",
                "summary": summary,
            },
        },
    )
    with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as output:
        print(f"### Parity matrix: {conclusion}\n\n{summary}", file=output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("resolve", "report"))
    args = parser.parse_args()
    if args.action == "report":
        finish(os.environ["PARITY_CHECK_ID"], os.environ["PARITY_RESULT"])
        return
    target = start()
    with open(os.environ["GITHUB_OUTPUT"], "a") as output:
        print(f"run={str(target is not None).lower()}", file=output)
        for key, value in (target or {}).items():
            print(f"{key}={value}", file=output)
    if target is None:
        print("::notice::Parity was not authorized or the PR is no longer open.")


if __name__ == "__main__":
    main()
