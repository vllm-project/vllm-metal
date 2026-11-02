# SPDX-License-Identifier: Apache-2.0
"""Authorize parity runs, summarize results and report checks."""

import argparse
import json
import os
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
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


def summarize(directory: Path, *, model: str, revision: str, sha: str) -> str:
    """Render saved results in the unprivileged model job, including missing cases."""
    lines = [
        f"### {model}",
        f"\nCommit: `{sha}`  \nCheckpoint revision: `{revision}`\n",
        "Greedy output parity, 40 prompts, 20 output tokens, checkpoint precision. "
        "TOP_K_MATCH is mutual top-5 membership at the first divergence; "
        "comparisons overlap and are not additive.\n",
        "| Comparison | B | EXACT | TOP_K_MATCH | FAIL | Result |",
        "|---|---:|---:|---:|---:|---|",
    ]

    def row(label, batch, report):
        if report is None:
            lines.append(f"| {label} | {batch} | — | — | — | Missing |")
            return
        counts = report["counts"]
        status = "PASS" if report["passed"] else "FAIL"
        lines.append(
            f"| {label} | {batch} | {counts['EXACT']} | {counts['TOP_K_MATCH']} "
            f"| {counts['FAIL']} | {status} |"
        )

    for batch in (1, 2):
        path = directory / f"batch-{batch}.json"
        report = json.loads(path.read_text())["comparison"] if path.exists() else None
        row("Native → HTTP target", batch, report)
    if model == "Qwen/Qwen3-0.6B":
        path = directory / "speculative/summary.json"
        comparisons = (
            json.loads(path.read_text())["comparisons"] if path.exists() else {}
        )
        for name, label in (
            ("native-vs-target", "Native → engine target"),
            ("native-vs-draft_model", "Native → same-model draft"),
            ("target-vs-draft_model", "Engine target → same-model draft"),
        ):
            for batch in (1, 2):
                row(label, batch, comparisons.get(f"{name}-b{batch}"))
        lines.extend(
            [
                "\nFP32 engine checks, 3 draft positions, cold and cached prefixes. "
                "Same-model: 8 prompts × 65 tokens, ≥98% acceptance. "
                "Controlled: 2 prompts × 37 tokens, rejection at each position "
                "followed by full acceptance; exact recovery and counters required. "
                "Controlled rates describe the fixture, not model quality.\n",
                "| Check | B | Prefix | Draft rounds | Accepted / proposed | Acceptance | EAL | Position 1 / 2 / 3 (%) | Result |",
                "|---|---:|---|---:|---:|---:|---:|---|---|",
            ]
        )
        records = {}
        for path in sorted((directory / "acceptance").glob("*/metrics.json")):
            if path.parent.is_symlink():
                continue  # pytest's "current" aliases refer to the same case.
            result = json.loads(path.read_text())
            key = result["kind"], result["batch_size"]
            if key in records:
                raise ValueError(f"Duplicate acceptance results: {key}")
            records[key] = result["phases"]
        for kind in ("same-model", "controlled"):
            for batch in (1, 2):
                for phase in ("cold", "cached"):
                    record = records.get((kind, batch), {}).get(phase)
                    prefix = f"| {kind} | {batch} | {phase} |"
                    if record is None:
                        lines.append(prefix + " — | — | — | — | — | Missing |")
                        continue
                    before, after = record["before"], record["after"]
                    rounds, proposed, accepted = (
                        after[name] - before[name]
                        for name in (
                            "num_drafts",
                            "num_draft_tokens",
                            "num_accepted_tokens",
                        )
                    )
                    acceptance = f"{accepted / proposed:.2%}" if proposed else "—"
                    eal = f"{1 + accepted / rounds:.2f}" if rounds else "—"
                    positions = " / ".join(
                        f"{100 * (end - begin) / rounds:.1f}" if rounds else "—"
                        for begin, end in zip(
                            before["per_position"], after["per_position"], strict=True
                        )
                    )
                    status = "PASS" if record["passed"] else "FAIL"
                    lines.append(
                        f"{prefix} {rounds:g} | {accepted:g} / {proposed:g} | "
                        f"{acceptance} | {eal} | {positions} | {status} |"
                    )
        lines.append(
            "\nEAL = 1 + accepted tokens / draft rounds. Position rates use draft rounds as the denominator."
        )
        junit = directory / "acceptance.xml"
        if junit.exists():
            suites = ET.parse(junit).getroot().findall("testsuite")
            totals = {
                name: sum(int(suite.attrib[name]) for suite in suites)
                for name in ("tests", "failures", "errors", "skipped")
            }
            lines.append(
                "\nAcceptance suite: "
                + ", ".join(f"{value} {name}" for name, value in totals.items())
                + "."
            )
        else:
            lines.append(
                "\nAcceptance suite: missing JUnit result; completion not confirmed."
            )
    environment = directory / "environment.txt"
    if environment.exists():
        lines.append(
            f"\n<details><summary>Environment</summary>\n\n```text\n{environment.read_text().rstrip()}\n```\n</details>"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("resolve", "report", "summary"))
    args = parser.parse_args()
    if args.action == "summary":
        print(
            summarize(
                Path(os.environ["RUNNER_TEMP"]) / "parity",
                model=os.environ["MODEL_ID"],
                revision=os.environ["MODEL_REVISION"],
                sha=os.environ["TESTED_SHA"],
            )
        )
        return
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
