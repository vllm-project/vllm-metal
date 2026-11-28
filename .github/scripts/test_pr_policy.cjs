// SPDX-License-Identifier: Apache-2.0
const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const { join } = require('node:path');
const { test } = require('node:test');
const enforce = require('./pr_policy.cjs');

const template = readFileSync(join(__dirname, '../PULL_REQUEST_TEMPLATE.md'), 'utf8');
const confirmed = template.replaceAll('- [ ] I (a human)', '- [x] I (a human)');
const humanLines = confirmed.split('\n').filter((line) =>
  line.startsWith('- [x] I (a human)')
).join('\n');

function pr(number, extra = {}) {
  return {
    number, node_id: String(number), user: { id: 1 },
    state: 'open', draft: false, body: confirmed,
    html_url: `https://github.com/example/repo/pull/${number}`,
    ...extra,
  };
}

function fixture(pulls) {
  const comments = [];
  const converted = [];
  const github = {
    rest: {
      pulls: {
        get: async ({ pull_number }) => ({
          data: { ...pulls.find((pull) => pull.number === pull_number) },
        }),
        list() {},
      },
      issues: {
        listComments() {},
        createComment: async (data) => comments.push({
          ...data, id: comments.length + 1,
          user: { login: 'github-actions[bot]' },
        }),
        updateComment: async ({ comment_id, body }) => {
          comments.find((comment) => comment.id === comment_id).body = body;
        },
      },
    },
    paginate: async (method) => {
      if (method === github.rest.pulls.list) return pulls.map((pull) => ({ ...pull }));
      assert.equal(method, github.rest.issues.listComments);
      return comments;
    },
    graphql: async (_, { id }) => {
      pulls.find((pull) => pull.node_id === id).draft = true;
      converted.push(id);
    },
  };
  return {
    github, comments, converted,
    run: (number) => enforce({
      github, context: { repo: { owner: 'example', repo: 'repo' }, issue: { number } },
    }),
  };
}

test('two ready PRs are allowed; drafts and other authors do not consume slots', async () => {
  const f = fixture([pr(1), pr(2), pr(3, { draft: true }), pr(4, { user: { id: 2 } })]);
  await f.run(1);
  await f.run(2);
  assert.deepEqual(f.converted, []);
});

test('a third ready PR returns to draft without changing the existing two', async () => {
  const pulls = [pr(1), pr(2), pr(3)];
  const f = fixture(pulls);
  await f.run(3);
  assert.deepEqual(f.converted, ['3']);
  assert.deepEqual(pulls.map((pull) => pull.draft), [false, false, true]);
  assert.match(f.comments[0].body, /#1.*#2/);
});

test('unchecked, missing, quoted, or hidden confirmations do not qualify', async () => {
  for (const body of [
    template, '', null,
    confirmed.replace('- [x] I (a human) have written', '- [ ] I (a human) have written'),
    confirmed.replace('- [x] I (a human) understand', '- [ ] I (a human) understand'),
    `<!--\n${humanLines}\n-->`,
    `\`\`\`markdown\n${confirmed}\n\`\`\``,
    confirmed.split('\n').map((line) => '> ' + line).join('\n'),
  ]) {
    const f = fixture([pr(1, { body })]);
    await f.run(1);
    assert.deepEqual(f.converted, ['1']);
    assert.match(f.comments[0].body, /Manually check both/);
  }
});

test('a closed PR or a PR returned to draft while waiting is left alone', async () => {
  for (const change of [{ state: 'closed' }, { draft: true }]) {
    const current = pr(1);
    const f = fixture([current, pr(2), pr(3)]);
    const paginate = f.github.paginate;
    f.github.paginate = async (...args) => {
      Object.assign(current, change);
      return paginate(...args);
    };
    await f.run(1);
    assert.deepEqual(f.converted, []);
  }
});

test('serialized arrivals preserve the existing PR and stay within two ready slots', async () => {
  const pulls = [pr(1), pr(2), pr(3)];
  const f = fixture(pulls);
  await f.run(2);
  await f.run(3);
  assert.equal(pulls[0].draft, false);
  assert.equal(pulls.filter((pull) => !pull.draft).length, 2);
});

test('repeated attempts update one explanation instead of posting again', async () => {
  const current = pr(3, { body: template });
  const f = fixture([pr(1), pr(2), current]);
  await f.run(3);
  current.draft = false;
  current.body = confirmed;
  await f.run(3);
  assert.equal(f.comments.length, 1);
  assert.doesNotMatch(f.comments[0].body, /Manually check both/);
});
