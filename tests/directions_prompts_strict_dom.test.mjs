// The Directions tab's prompt sets against the strict fake DOM: parts of typed prompts and dataset specifications,
// the value sent, the field a refusal names, Check (nothing fetched) and Download (a stoppable job), and a measurement's
// stored prompts as parts again. Expectations from the tab's build spec (5.4, 9.6, 9.1.6, 13.2) and docs/DIRECTIONS.md
// ("Prompt sets from datasets", "POST /api/directions/datasets"); made-up prompts only. No server, model or network.
import test from 'node:test';
import assert from 'node:assert/strict';
import { installStrictDom } from './strict_dom.mjs';
import { server, hold, refusal, makeApp, draw, settle, labelled, button, type, change, byId, nodes } from './directions_fixtures.mjs';
import { newSet, parseTexts, setValue, partOfItem, errorTarget, setFromMeasurement, checkAfterRefusal } from '../web/js/directions_prompts.js';
import { onDirectionsEvent } from '../web/js/directions.js';

const { view } = installStrictDom();
const SPEC = { dataset: 'mlabonne/harmless_alpaca', split: 'train[:400]', column: 'text' };
const texts = (text) => ({ kind: 'texts', text });
const dataset = (spec) => ({ kind: 'dataset', spec: { dataset: '', commit: '', config: '', split: '', column: '', prefix: '', suffix: '', system_prompt: '', ...spec } });
const CACHED = { cached: true, count: 400, prompts_sha256: 'b13d6c60adef58f41897e5ae50245cb171fd0145768300ffe5857b382e080c46',
  datasets: { version: 1, count: 400, prompts_sha256: 'b13d', parts: [] }, preview: ['Name three fruits.', 'Write a haiku.', { user: 'Explain rain.', system: 'Be brief.' }],
  fetched: [], fetched_bytes: 0, seconds: 0.001 };
// docs "POST /api/directions/datasets": what a fetch would get, with no file list in the workspace
const NOT_CACHED = { cached: false, would_fetch: [{ field: 'prompts', repository: 'mlabonne/harmless_alpaca', revision: 'main', commit: null, files: null,
  requests: ["the repository's file list at main (one Hub API request)", 'README.md', 'the data files of split train (names and sizes known after the file list)'] }],
fetched: [], fetched_bytes: 0, seconds: 0.002 };
const goodSet = () => byId('dirs-set-good');
const partBox = (set, n) => set.querySelector(`[data-part="${n}"]`);

test('P1: one box of typed prompts: its non-empty lines, each kept as typed', () => {
  const r = setValue({ parts: [texts('a\n\n b \nc')] });
  assert.deepEqual(r.value, ['a', ' b ', 'c']);
  assert.ok(r.value.every((x) => typeof x === 'string'));
  assert.deepEqual(r.problems, []);
  assert.deepEqual(setValue({ parts: [texts('x\r\ny')] }).value, ['x', 'y']);
});

test('P2: a box starting with [ is a JSON list of texts and {user, system}; anything else is refused there', () => {
  assert.deepEqual(setValue({ parts: [texts('[ "x", {"user": "y", "system": "s"} ]')] }).value, ['x', { user: 'y', system: 's' }]);
  assert.deepEqual(parseTexts('[1]', 1), { items: [], problem: 'part 1: starts with [ but is not a JSON list of prompts (item 1 is neither text nor {user, system})' });
  assert.match(parseTexts('[oops', 1).problem, /^part 1: starts with \[ but is not a JSON list of prompts \(.+\)$/);
  assert.deepEqual(setValue({ parts: [texts('[oops')] }).value, []);
  assert.equal(setValue({ parts: [texts('[oops')] }).problems.length, 1);
});

test('P3: mixed parts: the typed prompts, then the specification with only its filled fields; index maps items to parts', () => {
  const set = { parts: [texts('a\nb'), dataset({ dataset: 'mlabonne/harmless_alpaca', split: 'train[:400]', column: 'text' })] };
  const r = setValue(set);
  assert.deepEqual(r.value, ['a', 'b', SPEC]);
  assert.deepEqual(r.index, [0, 0, 1]);
  assert.deepEqual(Object.keys(r.value[2]), ['dataset', 'split', 'column'], 'no empty keys');
});

test('P4: errorTarget: the set, its item and the part that item comes from', () => {
  const set = { parts: [texts('a\nb'), dataset(SPEC)] };
  const t = errorTarget('DirectionError: good[2].split: no split named "tran"; the splits are train, test');
  assert.deepEqual(t, { field: 'good', item: 2 });
  assert.equal(partOfItem(set, t.item), 1, 'part 2 (from 0: 1)');
  assert.deepEqual(errorTarget('good must list 1..1024 prompts'), { field: 'good', item: null });
  assert.deepEqual(errorTarget('scorers[1].prompts[0].dataset: no file or directory at /data/x'), { field: 'scorers[1].prompts', item: 0 });
  assert.deepEqual(errorTarget('bad prompt 3 must be text of 1..20000 characters'), { field: 'bad', item: 2 });
  assert.deepEqual(errorTarget('DirectionError: weight_distributions.attn.o_proj.min_weight_distance: must be above 0'), { field: 'attn.o_proj.min_weight_distance', item: null });
  assert.deepEqual(errorTarget('scores[0]: unknown plugin'), { field: 'scores[0]', item: null });
  assert.deepEqual(errorTarget('modifiers[0].lora_rank: must be a whole number'), { field: 'modifiers[0]', item: null });
  assert.deepEqual(errorTarget('winsorization_quantile must be a number from 0 to 1', ['name', 'winsorization_quantile']), { field: 'winsorization_quantile', item: null });
  assert.equal(errorTarget('the model cannot be run here'), null);
});

test('P5: Check sends exactly {prompts, system} and shows the count, the hash and three previews', async () => {
  const s = server({ 'POST /api/directions/datasets': CACHED });
  const app = makeApp();
  await draw(app);
  type(labelled(view, 'good prompts part 1 texts'), 'Name three fruits.\nWrite a haiku.');
  button(goodSet(), "Heretic's default").click(); await settle();
  button(goodSet(), 'Check').click(); await settle();
  const call = s.last('POST', '/api/directions/datasets');
  assert.deepEqual(Object.keys(call.body), ['prompts', 'system']);
  assert.deepEqual(call.body, { prompts: ['Name three fruits.', 'Write a haiku.', SPEC], system: 'You are a helpful assistant.' });
  const result = goodSet().querySelector('.dirs-check');
  assert.match(result.textContent, /^400 prompts · sha256 b13d6c60adef(?!5)/);
  assert.deepEqual([...result.querySelector('ol').children].map((li) => li.textContent), ['Name three fruits.', 'Write a haiku.', 'Explain rain.system: Be brief.']);
  assert.match(goodSet().textContent, /2 prompts typed \+ 398 from datasets, of 1\.\.1024/);
  // chat off: no system message is sent
  change(labelled(view, 'as a chat message'), false); await settle();
  button(goodSet(), 'Check').click(); await settle();
  assert.equal(s.last('POST', '/api/directions/datasets').body.system, null);
});

test('P6: not cached: what a download would fetch; Download is a stoppable job with its progress', async () => {
  const fetchRoute = hold();
  const s = server({ 'POST /api/directions/datasets': (body) => (body.download ? fetchRoute : NOT_CACHED) });
  const app = makeApp();
  await draw(app);
  button(goodSet(), "Heretic's default").click(); await settle();
  assert.match(goodSet().textContent, /0 prompts typed \+ part 2 not checked/);
  button(goodSet(), 'Check').click(); await settle();
  const result = goodSet().querySelector('.dirs-check');
  assert.match(result.textContent, /mlabonne\/harmless_alpaca at main \(latest commit unknown\)/);
  assert.match(result.textContent, /file names and sizes known after the file list/);
  for (const line of NOT_CACHED.would_fetch[0].requests) assert.ok(nodes(result).some((n) => n.tag === 'li' && n.textContent === line), line);
  button(result, 'Download').click(); await settle();
  const sent = fetchRoute.calls[0].body;
  assert.equal(sent.download, true);
  assert.match(sent.stream, /^dirs-fetch-[A-Za-z0-9_-]+$/);
  assert.deepEqual(sent.prompts, [SPEC]);
  assert.equal(sent.system, 'You are a helpful assistant.');
  assert.equal(byId('dirs-job').querySelector('[role=status]').textContent, 'Downloading: starting.');
  onDirectionsEvent({ type: 'direction_datasets', workspace: 'fixture', state: 'fetching', seconds: 0.4, phase: 'file', file: 'data/train-00000-of-00001.parquet', done_bytes: 1024, total_bytes: 4096, stream: sent.stream }, app);
  assert.equal(byId('dirs-job').querySelector('[role=status]').textContent, 'Downloading data/train-00000-of-00001.parquet: 1.0 KiB of 4.0 KiB');
  onDirectionsEvent({ type: 'direction_datasets', workspace: 'fixture', state: 'fetching', seconds: 0.1, phase: 'list', file: null, done_bytes: 0, total_bytes: 0, stream: sent.stream }, app);
  assert.equal(byId('dirs-job').querySelector('[role=status]').textContent, 'Downloading the file list: 0 bytes of 0 bytes');
  assert.equal(button(goodSet(), 'Check').disabled, true, 'no other request while the download runs');
  button(byId('dirs-job'), 'Stop').click(); await settle();
  assert.deepEqual(s.last('POST', '/api/generation/stop').body, { stream: sent.stream });
  fetchRoute.release({ stopped: true, done_bytes: 388, total_bytes: 972182, fetched: [{ repository: 'mlabonne/harmless_alpaca', commit: '02c6a92cfcf11bb0c387334f8146d149d65b587f', path: 'README.md', bytes: 388 }] });
  await settle();
  assert.equal(byId('dirs-job').querySelector('[role=status]').textContent, 'Download stopped after 388 bytes of 949.4 KiB; the 1 file(s) fetched stay in the cache.');
  assert.equal(app.keep.dirs.job, null);
});

test('P6: known files are listed with their sizes and the Download button sums them; a finished download shows the set', async () => {
  const files = [{ path: 'README.md', bytes: 388 }, { path: 'data/train-00000-of-00001.parquet', bytes: 971794 }];
  const s = server({ 'POST /api/directions/datasets': (body) => (body.download ? { ...CACHED, fetched: files.map((f) => ({ repository: 'mlabonne/harmless_alpaca', commit: '02c6a92cfcf11bb0c387334f8146d149d65b587f', ...f })), fetched_bytes: 972182 }
    : { ...NOT_CACHED, would_fetch: [{ ...NOT_CACHED.would_fetch[0], commit: '02c6a92cfcf11bb0c387334f8146d149d65b587f', files, requests: files.map((f) => f.path) }] }) });
  const app = makeApp();
  await draw(app);
  button(goodSet(), "Heretic's default").click(); await settle();
  button(goodSet(), 'Check').click(); await settle();
  const result = goodSet().querySelector('.dirs-check');
  assert.match(result.textContent, /at main \(02c6a92cfcf1\)/);
  assert.ok(nodes(result).some((n) => n.tag === 'li' && n.textContent === 'data/train-00000-of-00001.parquet (949.0 KiB)'));
  button(result, 'Download (949.4 KiB)').click(); await settle();
  assert.equal(s.last('POST', '/api/directions/datasets').body.download, true);
  assert.equal(byId('dirs-job').querySelector('[role=status]').textContent, 'Downloaded 2 file(s), 949.4 KiB.');
  assert.match(goodSet().querySelector('.dirs-check').textContent, /^400 prompts · sha256 b13d6c60adef · fetched 2 file\(s\), 949\.4 KiB/);
});

test('P7: pyarrow missing: the refusal is shown under the dataset part and names requirements-datasets.txt', async () => {
  server({ 'POST /api/directions/datasets': refusal('DirectionError: prompts[1]: reading parquet files needs pyarrow, which is not installed here: python -m pip install -r requirements-datasets.txt') });
  const app = makeApp();
  await draw(app);
  type(labelled(view, 'good prompts part 1 texts'), 'one typed prompt');
  button(goodSet(), "Heretic's default").click(); await settle();
  button(goodSet(), 'Check').click(); await settle();
  const part = partBox(goodSet(), 2);
  const alert = [...part.querySelectorAll('[role=alert]')].find((n) => n.textContent.includes('pyarrow'));
  assert.equal(alert.textContent, 'prompts[1]: reading parquet files needs pyarrow, which is not installed here: python -m pip install -r requirements-datasets.txt');
  assert.match(part.className, /\bdirs-bad\b/);
  assert.doesNotMatch(partBox(goodSet(), 1).textContent, /pyarrow/);
});

test("P8: Heretic's default adds exactly Heretic's specification and sends nothing", async () => {
  const s = server();
  const app = makeApp();
  await draw(app);
  const before = s.calls.length;
  button(goodSet(), "Heretic's default").click(); await settle();
  const set = app.keep.dirs.sets.good;
  assert.deepEqual(setValue(set).value.at(-1), { dataset: 'mlabonne/harmless_alpaca', split: 'train[:400]', column: 'text' });
  assert.equal(s.calls.length, before, 'no request');
  assert.equal(labelled(view, 'good prompts part 2 dataset').value, 'mlabonne/harmless_alpaca');
  assert.equal(labelled(view, 'good prompts part 2 split').value, 'train[:400]');
  button(byId('dirs-set-bad'), "Heretic's default").click(); await settle();
  assert.deepEqual(setValue(app.keep.dirs.sets.bad).value.at(-1), { dataset: 'mlabonne/harmful_behaviors', split: 'train[:400]', column: 'text' });
  // the fields are Heretic's eight, each its own input; an edited field is sent
  type(labelled(view, 'good prompts part 2 commit'), '02c6a92cfcf11bb0c387334f8146d149d65b587f');
  assert.deepEqual(setValue(set).value.at(-1), { ...SPEC, commit: '02c6a92cfcf11bb0c387334f8146d149d65b587f' });
  assert.deepEqual(setValue(set).value.at(-1).dataset, 'mlabonne/harmless_alpaca');
});

test('P9: setFromMeasurement: one box without a record (JSON when an object is there); a record gives its parts back', () => {
  assert.deepEqual(setFromMeasurement(['a', 'b'], null), { parts: [texts('a\nb')], check: null });
  const withObject = setFromMeasurement(['a', { system: 'S', user: 'u' }], null);
  assert.equal(withObject.parts.length, 1);
  assert.deepEqual(JSON.parse(withObject.parts[0].text), ['a', { system: 'S', user: 'u' }]);
  assert.deepEqual(setValue(withObject).value, ['a', { system: 'S', user: 'u' }], 'reads back as the same set');
  assert.deepEqual(JSON.parse(setFromMeasurement(['two\nlines'], null).parts[0].text), ['two\nlines']);
  const prompts = Array.from({ length: 400 }, (_, i) => `prompt ${i + 1}`);
  const spec = { dataset: 'mlabonne/harmless_alpaca', commit: '02c6a92cfcf11bb0c387334f8146d149d65b587f', config: null, split: 'train[:398]', column: 'text', prefix: '', suffix: '', system_prompt: null };
  const set = setFromMeasurement(prompts, { version: 1, count: 400, prompts_sha256: 'x',
    parts: [{ kind: 'inline', first: 1, count: 2 }, { kind: 'hub', first: 3, count: 398, specification: spec }] });
  assert.deepEqual(set.parts[0], texts('prompt 1\nprompt 2'));
  assert.equal(set.parts[1].kind, 'dataset');
  assert.deepEqual(setValue(set).value, ['prompt 1', 'prompt 2', { dataset: spec.dataset, commit: spec.commit, split: spec.split, column: 'text' }]);
});

test('P10: a refusal naming the datasets route for good[0] Checks the good set at once (no download) and shows the entries', async () => {
  const s = server({ 'POST /api/directions/datasets': NOT_CACHED });
  const app = makeApp();
  await draw(app);
  const st = app.keep.dirs;
  st.sets.good = { parts: [dataset(SPEC)], check: null };
  const text = 'DirectionError: good[0]: mlabonne/harmless_alpaca at 02c6a92cfcf1 is not in the local Hugging Face cache (<cache>), and nothing is downloaded without a request; fetch it with POST /api/directions/datasets (this prompt set and "download": true), which fetches the repository\'s file list, README.md and the files of split train';
  assert.equal(await checkAfterRefusal(app, st, text, st.sets, () => app.rerender()), true);
  await settle();
  const call = s.last('POST', '/api/directions/datasets');
  assert.deepEqual(call.body, { prompts: [SPEC], system: 'You are a helpful assistant.' });
  assert.ok(!('download' in call.body));
  assert.match(goodSet().querySelector('.dirs-check').textContent, /mlabonne\/harmless_alpaca at main \(latest commit unknown\)/);
  assert.ok(button(goodSet(), 'Download'));
  assert.equal(await checkAfterRefusal(app, st, 'DirectionError: good prompt 1 is empty', st.sets), false, 'other refusals check nothing');
});

test('a cache filled partly by another tool: the Check is refused with the route named, and a plain Download is offered', async () => {
  const fetched = hold();
  const s = server({ 'POST /api/directions/datasets': (body) => (body.download ? fetched
    : refusal('DirectionError: prompts[0]: the cached files of split train of mlabonne/harmless_alpaca at 02c6a92cfcf1 hold 10 rows and its dataset card says 400: files are missing from the cache (it was filled by another tool); fetch them with POST /api/directions/datasets ("download": true)')) });
  const app = makeApp();
  await draw(app);
  app.keep.dirs.sets.good = { parts: [dataset(SPEC)], check: null };
  await app.rerender();
  button(goodSet(), 'Check').click(); await settle();
  const part = partBox(goodSet(), 1);
  assert.match(part.textContent, /files are missing from the cache/);
  const download = button(part, 'Download');
  assert.ok(download, 'a plain Download button');
  download.click(); await settle();
  const sent = fetched.calls[0].body;
  assert.equal(sent.download, true);
  assert.deepEqual(sent.prompts, [SPEC]);
  assert.match(sent.stream, /^dirs-fetch-/);
  fetched.release(CACHED); await settle();
  assert.match(goodSet().querySelector('.dirs-check').textContent, /^400 prompts/);
  assert.doesNotMatch(goodSet().textContent, /files are missing/);
  assert.equal(s.posts().filter((c) => c.url === '/api/directions/datasets').length, 2);
});

test('a download refused offline: the message under the dataset part, and the set as it was', async () => {
  const OFFLINE = "DirectionError: prompts[1]: download was asked for, but HF_HUB_OFFLINE is set in the IDE's environment, so nothing can be downloaded";
  const fetched = hold();
  const s = server({ 'POST /api/directions/datasets': (body) => (body.download ? fetched : NOT_CACHED) });
  const app = makeApp();
  await draw(app);
  type(labelled(view, 'good prompts part 1 texts'), 'Name three fruits.');
  button(goodSet(), "Heretic's default").click(); await settle();
  const set = app.keep.dirs.sets.good, before = JSON.parse(JSON.stringify(set.parts));
  button(goodSet(), 'Check').click(); await settle();
  button(goodSet().querySelector('.dirs-check'), 'Download').click(); await settle();
  assert.equal(fetched.calls[0].body.download, true);
  fetched.refuse(OFFLINE); await settle();
  const alert = [...partBox(goodSet(), 2).querySelectorAll('[role=alert]')].find((n) => n.textContent.includes('HF_HUB_OFFLINE'));
  assert.ok(alert, 'shown under the dataset part');
  assert.equal(alert.textContent, "prompts[1]: download was asked for, but HF_HUB_OFFLINE is set in the IDE's environment, so nothing can be downloaded");
  assert.match(partBox(goodSet(), 2).className, /\bdirs-bad\b/);
  assert.doesNotMatch(partBox(goodSet(), 1).textContent, /HF_HUB_OFFLINE/);
  assert.equal(app.keep.dirs.sets.good, set, 'the same set');
  assert.deepEqual(set.parts, before, 'its parts as they were');
  assert.equal(labelled(view, 'good prompts part 1 texts').value, 'Name three fruits.');
  assert.equal(labelled(view, 'good prompts part 2 dataset').value, 'mlabonne/harmless_alpaca');
  assert.equal(app.keep.dirs.job, null, 'the job ended');
  assert.equal(button(goodSet(), 'Check').disabled, false);
  assert.equal(s.posts().filter((c) => c.url === '/api/directions/datasets').length, 2, 'nothing sent again by itself');
});

test('parts: + prompts, + dataset, remove; a JSON problem is shown under its box as it is typed', async () => {
  server();
  const app = makeApp();
  await draw(app);
  button(goodSet(), '+ dataset').click(); await settle();
  button(goodSet(), '+ prompts').click(); await settle();
  assert.deepEqual(app.keep.dirs.sets.good.parts.map((p) => p.kind), ['texts', 'dataset', 'texts']);
  type(labelled(view, 'good prompts part 3 texts'), '[1, 2');
  assert.match(partBox(goodSet(), 3).textContent, /part 3: starts with \[ but is not a JSON list of prompts/);
  button(partBox(goodSet(), 2), 'remove').click(); await settle();
  assert.deepEqual(app.keep.dirs.sets.good.parts.map((p) => p.kind), ['texts', 'texts']);
  assert.equal(newSet().parts.length, 1);
});

test('"into the form": a measurement\'s stored prompts become the good and bad sets (a GET)', async () => {
  const s = server({ 'GET /api/directions/refusal/prompts': { good: ['abc', 'hello', 'fox'], bad: ['xyz', 'qqq'] } });
  const app = makeApp();
  await draw(app);
  button(byId('dirs-list'), 'into the form').click(); await settle();
  assert.equal(labelled(view, 'good prompts part 1 texts').value, 'abc\nhello\nfox');
  assert.equal(labelled(view, 'bad prompts part 1 texts').value, 'xyz\nqqq');
  assert.deepEqual(s.posts(), []);
});
