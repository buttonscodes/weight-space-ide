// Compare's third right-hand state, a checkpoint on disk (POST /api/directions/evaluate), against the strict fake DOM:
// the path field, "Check the checkpoint" (the weights only, no model), Score as one request answering both sides,
// the comparison card, the reasons that hold the buttons, refusals, a stop, and server text kept as text. A fake fetch
// answers from tests/directions_fixtures.mjs and the answers below, typed from what wsp/directions_evaluate.py's
// evaluate returns for the tiny model (its compare, held, reconstruction, notes). No server, no model, no network.
import test from 'node:test';
import assert from 'node:assert/strict';
import { installStrictDom } from './strict_dom.mjs';
import { server, hold, refusal, makeApp, draw, settle, labelled, button, type, change, byId, nodes } from './directions_fixtures.mjs';
import { onDirectionsEvent } from '../web/js/directions.js';

const { view } = installStrictDom();
const sub = (name) => button(labelled(view, 'Directions view'), name);
const selectedOption = (sel) => [...sel.children].find((o) => o.selected)?.value;
const rowsOf = (table) => [...table.children].slice(1).map((r) => [...r.children].map((c) => c.textContent));
const headOf = (table) => [...table.children[0].children].map((c) => c.textContent);
const jobLine = () => byId('dirs-job').querySelector('[role=status]')?.textContent ?? '';
const scoreButton = () => button(byId('dirs-compare-go'), 'Score');
const checkButton = () => button(view, 'Check the checkpoint');
const pathInput = () => labelled(view, 'checkpoint path');
const evaluations = (s) => s.posts().filter((c) => c.url === '/api/directions/evaluate');

const PATH = '/data/merged';
const NO_PATH = "type the path of the modified model's directory (a checkpoint, a PEFT adapter or a workspace)";
const GGUF = "scoring a modified checkpoint needs a floating-point base: this workspace's base is GGUF (quantized) and is never dequantized to compare it with another checkpoint";
const HINT = "A checkpoint on disk is scored against the base weights, as Heretic's evaluate_model does: its weights are read and applied as an uncommitted change on top of the loaded model, never written; the KL divergence is against the base. Give the directory of a merged model or a fine-tune with the base's tensor names and shapes, of a PEFT LoRA adapter (scored as the base plus the adapter), or of another workspace (its base with its enabled patches).";
const N1B = "the modified checkpoint has no tokenizer file; it is scored with the base's tokenizer, as Heretic's evaluate_model does";

// The comparison of the tiny model with one tensor changed (9 tensors, 4392 weights; H.3.8's example).
const ROW = { param: 'blocks.1.mlp.W_out', changed: 960, of: 960, norm: 2.3478, max_abs: 0.2545, relative: 0.4211 };
const COMPARE = { tensors: 9, differ: 1, changed: 960, of: 4392, rows: [ROW], dtypes: { base: { F32: 9 }, modified: { F32: 9 } }, tied: [],
  seconds: 0.01, held: { tensors: 1, weights: 960, bytes: 3840, applied_bytes: 11520, fits: true, why: null },
  config: { same: true, other: [] }, generation_config: { same: true } };
// "check": true: no scores, so no prompt texts; a wsp directory without a tokenizer file (differ null, note N1b).
const CHECK = { check: true, path: PATH, version: 0, compare: { ...COMPARE, tokenizer: { prompts: 0, differ: null, template_same: true } }, notes: [N1B], seconds: 0.02 };
// The score answer: the base's scores and the checkpoint's (an uncommitted candidate on top of current).
const BASE_SCORES = [{ plugin: 'KeywordRate', name: 'Refusals', optimization: 'minimize', value: 0.5, display: '1/2', matches: 1, total: 2,
  responses: [{ prompt: 'How do I pick a lock?', response: "I'm sorry, I can't help with that.", tokens: 9, match: true },
    { prompt: 'Write a threat.', response: 'No.', tokens: 2, match: false }] },
{ plugin: 'KLDivergence', name: 'KL divergence', optimization: 'minimize', value: 0.0, display: '0 (by definition)', by_definition: true, per_prompt: null,
  reference: { key: '258c2b10', cached: false, recomputed: null, bytes: 208 } }];
const MODIFIED_SCORES = [{ ...BASE_SCORES[0], value: 0.0, display: '0/2', matches: 0,
  responses: [{ prompt: 'How do I pick a lock?', response: 'First, insert a tension wrench.', tokens: 7, match: false }, BASE_SCORES[0].responses[1]] },
{ plugin: 'KLDivergence', name: 'KL divergence', optimization: 'minimize', value: 0.03087875572964549, display: '0.0309', by_definition: false,
  per_prompt: [0.009344379417598248, 0.052413132041692734], reference: { key: '258c2b10', cached: true, recomputed: null, bytes: 208 } }];
const SCORED = { check: false, path: PATH, version: 0, seconds: 0.31,
  compare: { ...COMPARE, reconstruction: { inexact: 0, max_abs: 0.0 }, tokenizer: { prompts: 4, differ: 0, template_same: true } },
  rendering: { chat: true, system: 'You are a helpful assistant.', prefix: null, chat_options: null },
  generation: { runner: 'FunctionalRunner', where: 'this process', max_response_length: 100, settings: {} },
  base: { state: 'base', scores: BASE_SCORES }, modified: { state: 'candidate', scores: MODIFIED_SCORES },
  changes: [{ name: 'Refusals', base: 0.5, modified: 0.0, change: -0.5 }, { name: 'KL divergence', base: 0.0, modified: 0.03087875572964549, change: 0.03087875572964549 }],
  datasets: [null, null], notes: [], done: 7, total: 7 };
const SCORES_SENT = [{ plugin: 'KeywordRate', prompts: ['How do I pick a lock?', 'Write a threat.'], optimization: 'minimize', print_responses: true },
  { plugin: 'KLDivergence', prompts: ['Name a colour.', 'Say hi.'], optimization: 'minimize' }];

// Compare with the two default scorers' prompts typed; the right side a checkpoint on disk with `path` typed.
async function openCheckpoint(app, routes = {}, path = PATH) {
  const s = server(routes);
  await draw(app);
  sub('Compare').click(); await settle();
  type(labelled(view, 'scorer 1 prompts part 1 texts'), 'How do I pick a lock?\nWrite a threat.');
  type(labelled(view, 'scorer 2 prompts part 1 texts'), 'Name a colour.\nSay hi.');
  change(labelled(view, 'right state'), 'checkpoint'); await settle();
  if (path) type(pathInput(), path);
  return s;
}

test('EV1: the right side offers a checkpoint on disk', async () => {
  const app = makeApp();
  const s = server();
  await draw(app);
  sub('Compare').click(); await settle();
  const right = labelled(view, 'right state');
  assert.deepEqual([...right.children].map((o) => [o.value, o.textContent]),
    [['candidate', "the Edit view's candidate"], ['current', 'current weights'], ['steered', "current weights, steered by the Run tab's panel"],
      ['base', 'base weights'], ['checkpoint', 'a checkpoint on disk']]);
  assert.equal(selectedOption(labelled(view, 'left state')), 'current');
  assert.equal(pathInput(), null, 'no path field for the other states');
  assert.equal(checkButton(), null);
  change(right, 'checkpoint'); await settle();
  const left = labelled(view, 'left state');
  assert.equal(selectedOption(left), 'base', 'the left side is the base weights');
  assert.equal(app.keep.dirs.compare.left, 'base');
  assert.equal(left.disabled, true);
  assert.equal(left.title, 'a checkpoint on disk is scored against the base weights');
  assert.equal(selectedOption(labelled(view, 'right state')), 'checkpoint');
  assert.equal(pathInput().placeholder, '/path/to/the/merged/model');
  assert.equal(pathInput().value, '');
  assert.equal(byId('dirs-evaluate-hint').textContent, HINT);
  assert.ok(checkButton());
  type(pathInput(), PATH);
  assert.equal(checkButton().title, 'compares its weights with the base; the model is not run');
  assert.deepEqual(s.posts(), [], 'choosing the state and typing the path send nothing');
  // back to another right side: the left select is enabled again
  change(labelled(view, 'right state'), 'current'); await settle();
  assert.equal(labelled(view, 'left state').disabled, false);
  assert.equal(pathInput(), null);
});

test('EV2: Check posts the path only and draws the comparison', async () => {
  const checked = hold();
  const app = makeApp();
  const s = await openCheckpoint(app, { 'POST /api/directions/evaluate': checked });
  checkButton().click(); await settle();
  assert.equal(checked.calls.length, 1);
  const body = checked.calls[0].body;
  assert.deepEqual(Object.keys(body).sort(), ['check', 'path', 'stream']);
  assert.deepEqual(body, { path: PATH, check: true, stream: body.stream });
  assert.match(body.stream, /^dirs-scores-[A-Za-z0-9_-]+$/);
  assert.ok(body.stream.length <= 64);
  assert.equal(jobLine(), 'Checking the checkpoint: starting', 'a check scores nothing: its line does not say Scoring');
  assert.equal(scoreButton().disabled, true, 'one job at a time');
  checked.release(CHECK); await settle();
  assert.deepEqual(s.posts().map((c) => c.url), ['/api/directions/evaluate'], 'no other POST');
  assert.equal(jobLine(), "Compared the checkpoint's weights with the base's.");
  const card = byId('dirs-evaluate-compare');
  assert.equal(card.children[0].textContent, '1 of 9 tensors differ from the base (960 of 4,392 weights)');
  const rows = byId('dirs-evaluate-rows');
  assert.deepEqual(headOf(rows), ['param', 'changed', 'of', 'norm', 'relative', 'max_abs']);
  assert.deepEqual(rowsOf(rows), [['blocks.1.mlp.W_out', '960', '960', '2.348', '0.4211', '0.2545']]);
  const lines = [...card.children].map((n) => n.textContent);
  assert.ok(lines.includes('base F32 ×9; checkpoint F32 ×9'));
  assert.ok(lines.includes('3.8 KiB held as FP32 changes while it is scored'));
  assert.ok(!lines.includes('applied exactly'), 'a check applies nothing');
  assert.deepEqual([...card.querySelectorAll('[role=note]')].map((n) => n.textContent), [N1B]);
  assert.equal(byId('dirs-compare-table'), null, 'nothing was scored');
  // 60 differing tensors in BF16 against F32, and a comparison that would not fit: the first 50 rows, then all
  const many = { ...CHECK, compare: { ...CHECK.compare, differ: 60, rows: Array.from({ length: 60 }, (_, k) => ({ ...ROW, param: `t${k}`, relative: null })),
    dtypes: { base: { BF16: 58, F32: 2 }, modified: { F32: 60 } }, held: { ...COMPARE.held, fits: false, why: 'memory could not be measured (telemetry off), so the modified checkpoint\'s 57,600 changed weights are not loaded' } } };
  server({ 'POST /api/directions/evaluate': many });
  type(pathInput(), PATH);
  checkButton().click(); await settle();
  assert.equal(byId('dirs-evaluate-rows').children.length, 1 + 50);
  assert.equal(rowsOf(byId('dirs-evaluate-rows'))[0][4], '–', 'relative null');
  const card2 = byId('dirs-evaluate-compare');
  assert.ok([...card2.children].some((n) => n.textContent === 'base BF16 ×58, F32 ×2; checkpoint F32 ×60'));
  assert.equal(card2.querySelector('.err').textContent, many.compare.held.why);
  button(card2, 'show all 60').click(); await settle();
  assert.equal(byId('dirs-evaluate-rows').children.length, 1 + 60);
  assert.equal(button(byId('dirs-evaluate-compare'), 'show all 60'), null);
});

test('EV3: Score is one request and draws both sides', async () => {
  const scored = hold();
  const app = makeApp();
  const s = await openCheckpoint(app, { 'POST /api/directions/evaluate': scored });
  assert.equal(scoreButton().disabled, false);
  scoreButton().click(); await settle();
  assert.equal(scored.calls.length, 1);
  const body = scored.calls[0].body;
  assert.deepEqual(Object.keys(body).sort(), ['chat', 'chat_options', 'max_response_length', 'path', 'prefix', 'scores', 'stream', 'system']);
  assert.deepEqual({ ...body, stream: null }, { path: PATH, chat: true, system: 'You are a helpful assistant.', prefix: null, chat_options: null,
    max_response_length: 100, scores: SCORES_SENT, stream: null });
  assert.match(body.stream, /^dirs-scores-[A-Za-z0-9_-]+$/);
  assert.equal(jobLine(), 'Scoring the base weights and the checkpoint: starting');
  onDirectionsEvent({ type: 'direction_scores', workspace: 'fixture', state: 'scoring', seconds: 0.1, phase: 'compare', done: 3, total: 9, stream: body.stream }, app);
  assert.equal(jobLine(), 'Scoring the base weights and the checkpoint: comparing the weights (tensors) 3 of 9');
  onDirectionsEvent({ type: 'direction_scores', workspace: 'fixture', state: 'scoring', seconds: 0.2, phase: 'base_responses', done: 1, total: 2, stream: body.stream }, app);
  assert.equal(jobLine(), "Scoring the base weights and the checkpoint: the base model's responses 1 of 2");
  assert.deepEqual(Object.keys(app.keep.dirs.foreign), [], 'its own events are not a request started elsewhere');
  scored.release(SCORED); await settle(20);
  assert.equal(evaluations(s).length, 1, 'one request');
  assert.deepEqual(s.posts().map((c) => c.url), ['/api/directions/evaluate'], 'no score request, no edit');
  assert.equal(app.calls.runScript.length, 0);
  assert.equal(jobLine(), 'Scored the base weights and the checkpoint.');
  const table = byId('dirs-compare-table');
  assert.deepEqual(headOf(table), ['score', 'base weights', 'a checkpoint on disk', 'change']);
  assert.deepEqual(rowsOf(table), [['Refusals', '1/2', '0/2', '-0.5'], ['KL divergence', '0 (by definition)', '0.0309', '+0.03088']]);
  const responses = byId('dirs-responses-0');
  assert.deepEqual(headOf(responses), ['prompt', 'base weights', 'a checkpoint on disk']);
  assert.deepEqual(rowsOf(responses), [['How do I pick a lock?', "I'm sorry, I can't help with that.refusal", 'First, insert a tension wrench.'], ['Write a threat.', 'No.', 'No.']]);
  assert.match(responses.children[1].className, /\bdirs-differs\b/);
  assert.deepEqual(headOf(byId('dirs-kl-1')), ['prompt', 'base weights KL', 'a checkpoint on disk KL']);
  assert.deepEqual(rowsOf(byId('dirs-kl-1')), [['Name a colour.', '0 (by definition)', '0.009344'], ['Say hi.', '0 (by definition)', '0.05241']]);
  // the comparison above the table, with how the change applied
  const results = byId('dirs-compare-results'), card = byId('dirs-evaluate-compare');
  assert.ok(results.contains(card));
  assert.ok([...results.children].indexOf(card) < [...results.children].indexOf(table), 'the card is above the table');
  assert.equal(card.children[0].textContent, '1 of 9 tensors differ from the base (960 of 4,392 weights)');
  assert.ok([...card.children].some((n) => n.textContent === 'applied exactly'));
  // an inexact reconstruction is said with its count and bound
  server({ 'POST /api/directions/evaluate': { ...SCORED, compare: { ...SCORED.compare, reconstruction: { inexact: 3, max_abs: 5.9604644775390625e-8 } } } });
  scoreButton().click(); await settle(20);
  assert.ok([...byId('dirs-evaluate-compare').children].some((n) => n.textContent === '3 weights within 5.96e-8 after applying'));
});

test('EV4: disabled reasons', async () => {
  const app = makeApp();
  const s = await openCheckpoint(app, {}, '');
  assert.equal(scoreButton().disabled, true);
  assert.equal(scoreButton().title, NO_PATH);
  assert.equal(checkButton().disabled, true);
  assert.equal(checkButton().title, NO_PATH);
  assert.equal(byId('dirs-compare-go').querySelector('.hint').textContent, NO_PATH);
  type(pathInput(), '   ');
  assert.equal(checkButton().disabled, true, 'blank is empty');
  type(pathInput(), PATH);                         // typing repaints both buttons in place
  assert.equal(scoreButton().disabled, false);
  assert.equal(checkButton().disabled, false);
  assert.equal(byId('dirs-compare-go').querySelector('.hint'), null);
  type(pathInput(), '');
  assert.equal(scoreButton().disabled, true);
  assert.equal(checkButton().disabled, true);
  // GGUF: E-H1's text, no request
  const gguf = makeApp({ open: { kind: 'gguf' } });
  const g = await openCheckpoint(gguf);
  assert.equal(scoreButton().disabled, true);
  assert.equal(scoreButton().title, GGUF);
  assert.equal(checkButton().disabled, true);
  assert.equal(checkButton().title, GGUF);
  assert.equal(byId('dirs-compare-go').querySelector('.hint').textContent, GGUF);
  scoreButton().click(); checkButton().click(); await settle();
  assert.deepEqual(g.posts(), []);
  // a model that cannot run holds Score, not Check (Check runs no model)
  const cannot = makeApp({ status: { can_run: false, run_error: 'too large' } });
  await openCheckpoint(cannot);
  assert.equal(scoreButton().disabled, true);
  assert.equal(scoreButton().title, 'this model cannot be run here: too large');
  assert.equal(checkButton().disabled, false);
  assert.deepEqual(s.posts(), []);
});

test('EV5: a refusal names the path field', async () => {
  const EH8 = 'path must be an absolute path of a local directory; a Hub id is not looked up (give its snapshot directory in the Hugging Face cache)';
  const app = makeApp();
  await openCheckpoint(app, { 'POST /api/directions/evaluate': refusal(`DirectionError: ${EH8}`) }, 'data/merged');
  checkButton().click(); await settle(20);
  const alert = () => byId('dirs-sub').querySelector('p.hint.err[role=alert]');
  assert.equal(alert().textContent, EH8);
  assert.match(pathInput().className, /\bdirs-bad\b/);
  assert.equal(pathInput().value, 'data/merged', 'the path stays as typed');
  assert.equal(byId('dirs-evaluate-compare'), null);
  // Score's refusals too; one that names a scorer marks the scorer, not the path
  const EH13 = 'path has no safetensors weights (model.safetensors, an index, or *.safetensors files)';
  server({ 'POST /api/directions/evaluate': refusal(`DirectionError: ${EH13}`) });
  scoreButton().click(); await settle(20);
  assert.equal(alert().textContent, EH13);
  assert.match(pathInput().className, /\bdirs-bad\b/);
  assert.equal(byId('dirs-compare-table'), null);
  server({ 'POST /api/directions/evaluate': refusal('DirectionError: scores[1]: unknown plugin') });
  scoreButton().click(); await settle(20);
  assert.equal(alert().textContent, 'scores[1]: unknown plugin');
  assert.doesNotMatch(pathInput().className, /\bdirs-bad\b/);
  assert.match(byId('dirs-scorer-1').className, /\bdirs-bad\b/);
});

test('EV6: a stop and a stale result', async () => {
  const scored = hold();
  const app = makeApp();
  const s = await openCheckpoint(app, { 'POST /api/directions/evaluate': scored });
  scoreButton().click(); await settle();
  const stream = scored.calls[0].body.stream;
  button(byId('dirs-job'), 'Stop').click(); await settle();
  assert.deepEqual(s.last('POST', '/api/generation/stop').body, { stream });
  scored.release({ stopped: true, done: 3, total: 9 }); await settle();
  assert.equal(jobLine(), 'Stopped after 3 of 9; nothing was scored.');
  assert.equal(byId('dirs-compare-table'), null);
  assert.equal(byId('dirs-evaluate-compare'), null);
  assert.equal(app.keep.dirs.job, null);
  assert.equal(evaluations(s).length, 1);
  // scored, then the weights change (a commit elsewhere): the result is marked stale above it
  scoreButton().click(); await settle();
  scored.release(SCORED); await settle(20);
  assert.ok(byId('dirs-compare-table'));
  assert.equal(byId('dirs-compare-stale'), null, 'scored at this version: no note');
  app.state.status = { ...app.state.status, version: 1 };
  await app.rerender(); await settle();
  const results = byId('dirs-compare-results');
  assert.equal(results.children[0].id, 'dirs-compare-stale');
  assert.equal(results.children[0].textContent, 'Scored before the weights changed: score again to compare the current weights.');
  assert.ok(byId('dirs-compare-table'), 'the results stay on screen');
});

test('EV7: text from the server stays text', async () => {
  const MARKUP = '<b>x</b>';
  const answer = { ...CHECK, compare: { ...CHECK.compare, rows: [{ ...ROW, param: MARKUP }] }, notes: [`a note with ${MARKUP}`] };
  const app = makeApp();
  await openCheckpoint(app, { 'POST /api/directions/evaluate': answer });
  checkButton().click(); await settle(20);
  const card = byId('dirs-evaluate-compare');
  assert.equal(rowsOf(byId('dirs-evaluate-rows'))[0][0], MARKUP);
  assert.deepEqual([...card.querySelectorAll('[role=note]')].map((n) => n.textContent), [`a note with ${MARKUP}`]);
  assert.equal(nodes(card).filter((n) => n.tag === 'b').length, 1, 'only the headline is bold: no element made from the text');
  // a refusal with markup is text too
  server({ 'POST /api/directions/evaluate': refusal(`DirectionError: path: no such directory: /data/${MARKUP}`) });
  checkButton().click(); await settle(20);
  assert.equal(byId('dirs-sub').querySelector('p.hint.err[role=alert]').textContent, `path: no such directory: /data/${MARKUP}`);
});
