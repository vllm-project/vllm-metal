// SPDX-License-Identifier: Apache-2.0
const HUMAN_ITEMS = [
  'I (a human) have written or reviewed the code in this PR to the best of my ability.',
  'I (a human) understand the code changes and can explain their behavior and design choices.',
];
const MARKER = '<!-- vllm-metal-pr-readiness -->';

function confirmed(body) {
  const checked = new Set();
  let fence = null;
  const text = (body ?? '').replace(/<!--[\s\S]*?(?:-->|$)/g, '');
  for (const line of text.split(/\r?\n/)) {
    const delimiter = line.match(/^ {0,3}(`{3,}|~{3,})(.*)$/);
    if (delimiter) {
      if (!fence) {
        fence = delimiter[1];
      } else if (
        delimiter[1][0] === fence[0]
        && delimiter[1].length >= fence.length
        && delimiter[2].trim() === ''
      ) {
        fence = null;
      }
      continue;
    }
    if (!fence) {
      const item = line.match(/^ {0,3}[-*+] \[[xX]\] (.*?)\s*$/);
      if (item) checked.add(item[1]);
    }
  }
  return HUMAN_ITEMS.every((item) => checked.has(item));
}

module.exports = async function enforce({ github, context }) {
  const target = { ...context.repo, pull_number: context.issue.number };
  const { data: initial } = await github.rest.pulls.get(target);
  if (initial.state !== 'open' || initial.draft) return;

  const open = await github.paginate(github.rest.pulls.list, {
    ...context.repo,
    state: 'open',
    per_page: 100,
  });
  const { data: pr } = await github.rest.pulls.get(target);
  if (pr.state !== 'open' || pr.draft) return;

  const others = open.filter((other) =>
    other.number !== pr.number
    && other.user.id === pr.user.id
    && other.state === 'open'
    && !other.draft
  );
  const reasons = [];
  if (!confirmed(pr.body)) {
    reasons.push('Manually check both human checklist items in the PR description.');
  }
  if (others.length >= 2) {
    const links = others.map((other) => `[#${other.number}](${other.html_url})`);
    reasons.push(
      `You already have ${others.length} PRs ready for review: ${links.join(', ')}. `
      + 'Convert one to draft, or wait for one to merge or close.',
    );
  }
  if (!reasons.length) return;

  await github.graphql(
    `mutation ($id: ID!) {
      convertPullRequestToDraft(input: {pullRequestId: $id}) {
        pullRequest { isDraft }
      }
    }`,
    { id: pr.node_id },
  );

  const issue = { ...context.repo, issue_number: pr.number };
  const comments = await github.paginate(github.rest.issues.listComments, {
    ...issue,
    per_page: 100,
  });
  const previous = comments.find((comment) =>
    comment.user.login === 'github-actions[bot]'
    && comment.body?.startsWith(MARKER)
  );
  const body = [
    MARKER,
    'This PR was returned to draft:',
    ...reasons.map((reason) => `- ${reason}`),
    '',
    'Draft discussion remains welcome. Mark this PR ready when these requirements are met.',
  ].join('\n');
  if (previous) {
    if (previous.body !== body) {
      await github.rest.issues.updateComment({
        ...context.repo, comment_id: previous.id, body,
      });
    }
  } else {
    await github.rest.issues.createComment({ ...issue, body });
  }
};
