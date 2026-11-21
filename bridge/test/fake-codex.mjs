#!/usr/bin/env node
// A stand-in for `codex app-server` in tests: answers initialize, and thread/resume as FAKE_RESUME says
// ("gone": an error; "other": another thread id; otherwise the requested thread). No provider is contacted.
import { createInterface } from 'node:readline';
const lines = createInterface({ input: process.stdin });
const send = message => process.stdout.write(JSON.stringify(message) + '\n');
lines.on('line', line => {
  const m = JSON.parse(line);
  if (m.method === 'initialize') send({ id: m.id, result: {} });
  else if (m.method === 'thread/resume') {
    if (process.env.FAKE_RESUME === 'gone') send({ id: m.id, error: { code: -32600, message: `no rollout found for thread id ${m.params.threadId}` } });
    else send({ id: m.id, result: { thread: { id: process.env.FAKE_RESUME === 'other' ? 'thread-new' : m.params.threadId } } });
  } else if (m.method === 'thread/start') send({ id: m.id, result: { thread: { id: 'thread-new' } } });
  else if (m.method === 'turn/start') send({ id: m.id, result: { turn: { id: 'turn-1' } } });
});
