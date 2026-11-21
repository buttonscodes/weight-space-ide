// The Directions tab's Output sub-view against the strict fake DOM: the workspace's patches exported as a PEFT LoRA
// adapter (POST /api/export_adapter), a patch's reproduction record with its checks (GET /api/patches/{pid}/record),
// switching the patch off, and the re-run of the edit from its record (POST /api/patches/{pid}/reproduce), also with
// the measurement made again from the record. A fake fetch stands in for the server; no server, no model, no network.
// Expectations come from the export build spec (5.5-5.7, 6.4-6.9, 7, 9, 10.4) and its addendum (F, H); made-up ids,
// digests and paths.
import test from 'node:test';
import assert from 'node:assert/strict';
import { installStrictDom } from './strict_dom.mjs';
import { LIST, PATCHES, EXPORT_WRITTEN, EXPORT_PREVIEW, RECORD_ANSWER, RECORD_ANSWER_OFF, RECORD_ANSWER_REMEASURE, REMEASURE_BODY, REPRODUCE,
  server, hold, refusal, makeApp, draw, settle, labelled, button, type, change, byId, nodes } from './directions_fixtures.mjs';
import { patchOrigin, onPatchesLoaded } from '../web/js/directions.js';
import { verdictText } from '../web/js/directions_output.js';

const { view } = installStrictDom({ shell: true });
const sub = (name) => button(labelled(view, 'Directions view'), name);
const go = (text) => button(byId('dirs-export-go'), text);
const rowsOf = (table) => [...table.children].slice(1).map((r) => [...r.children].map((c) => c.textContent));
const FRAME = new Set(['/api/directions', '/api/directions/searches', '/api/probe/chat']);
const HF = { open: { kind: 'hf' } };
const A1 = "a PEFT adapter is for a transformers checkpoint; this is one of this project's own models: use the existing export (GET /api/export)";
const A2 = "this workspace's base is a quantized GGUF file: a PEFT adapter applies to a floating-point transformers checkpoint, and these patches were computed against the quantized weights, so the adapter's base would be the dequantized model, which the IDE never builds. Export the WSP companion bundle (GET /api/export), or apply the edit to the floating-point checkpoint the GGUF was made from";
const A8 = "AdapterError: an adapter named 'a1' exists in this workspace; pass replace, or choose another name";
const NO_RECORD = 'No reproduction record: this patch was not made by ablate_direction, ablate_ara, direction or direction_means, or it was made before records existed.';

const clone = (v) => JSON.parse(JSON.stringify(v));
// The tab drawn with these routes and the Output view shown; the patches are the shell's list (app.js renderPatches
// keeps it in app.keep.patchList), not read by the view.
async function openOutput(app, routes = {}, { patches = PATCHES.patches } = {}) {
  const s = server(routes);
  app.keep.patchList = clone(patches);
  await draw(app);
  sub('Output').click(); await settle();
  return s;
}
// app.weightsChanged as the shell's: the patches read again (here: `next`), then onPatchesLoaded.
function shellReads(app, next) {
  app.weightsChanged = async () => { app.calls.weightsChanged++; app.keep.patchList = clone(next()); onPatchesLoaded(app); };
}
const exportPosts = (s) => s.posts().filter((c) => c.url === '/api/export_adapter').map((c) => c.body);
const reproducePosts = (s) => s.posts().filter((c) => c.url === '/api/patches/p1/reproduce').map((c) => c.body);
const record = () => byId('dirs-record-p1');
async function showRecord() { labelled(view, 'record p1').click(); await settle(); }

// ------------------------------------------------------------------------------------------------ U1 to U5: the export
test("U1: opening Output sends no request: no GET, no POST; the patches shown are the shell's list; with no list yet the hint, then onPatchesLoaded draws them", async () => {
  const app = makeApp(HF);
  const s = server();
  app.keep.patchList = clone(PATCHES.patches);
  await draw(app);
  const before = s.calls.length;
  sub('Output').click(); await settle();
  assert.equal(s.calls.length, before, s.calls.slice(before).map((c) => c.url).join(', '));
  assert.ok(!s.gets().some((u) => !FRAME.has(u)), 'no GET of its own');
  assert.deepEqual(s.posts(), []);
  const boxes = nodes(byId('dirs-output-patches')).filter((n) => n.tag === 'input' && n.type === 'checkbox');
  assert.deepEqual(boxes.map((b) => [b.getAttribute('aria-label'), b.checked]), [['export p3', false], ['export p2', true], ['export p1', true]]);
  assert.ok(labelled(view, 'record p1') && labelled(view, 'record p3'), 'a Show record button per patch');
  assert.equal(labelled(view, 'record p2').textContent, 'Show record');
  assert.equal(record().children.length, 0, 'no record is shown before a click');
  assert.ok(byId('dirs-output-patches').textContent.includes('blocks[0].mlp.W_out *= 0.5'), 'the message');
  assert.ok(byId('dirs-output-patches').textContent.includes('agent'), 'the author');
  // drawn again (another sub-view and back): still no request
  sub('Measure').click(); await settle();
  sub('Output').click(); await settle();
  assert.equal(s.calls.length, before);
  // no list yet: the hint; the shell's read then draws the patches; on another sub-view onPatchesLoaded redraws nothing
  const fresh = makeApp(HF);
  const s2 = server();
  await draw(fresh);
  sub('Output').click(); await settle();
  assert.equal(byId('dirs-output-reading').textContent, 'Reading the patches…');
  assert.equal(byId('dirs-output-patches'), null);
  fresh.keep.patchList = clone(PATCHES.patches);
  onPatchesLoaded(fresh); await settle();
  assert.equal(byId('dirs-output-reading'), null);
  assert.equal(labelled(view, 'export p1').checked, true);
  sub('Measure').click(); await settle();
  const redraws = fresh.calls.rerender;
  onPatchesLoaded(fresh); await settle();
  assert.equal(fresh.calls.rerender, redraws, 'another sub-view is not redrawn');
  assert.ok(!s2.gets().some((u) => !FRAME.has(u)) && s2.posts().length === 0);
});

test('U2: a wsp or a gguf workspace: Preview and Export disabled with the A1 / A2 text, and no request', async () => {
  for (const [kind, text] of [['wsp', A1], ['gguf', A2]]) {
    const s = await openOutput(makeApp({ open: { kind } }));
    for (const label of ['Preview (dry run)', 'Export']) {
      assert.equal(go(label).disabled, true, `${kind} ${label}`);
      assert.equal(go(label).title, text);
      go(label).click(); await settle();
    }
    assert.equal(byId('dirs-export-go').querySelector('.hint').textContent, text);
    assert.deepEqual(s.posts(), [], kind);
  }
});

test('U3: the body: name, patches null for exactly the enabled ones, else the ticked ids in stack order; Preview is a dry run, Export is not', async () => {
  const s = await openOutput(makeApp(HF), { 'POST /api/export_adapter': (body) => (body.dry_run ? EXPORT_PREVIEW : EXPORT_WRITTEN) });
  type(labelled(view, 'adapter name'), 'a1');
  go('Preview (dry run)').click(); await settle();
  assert.deepEqual(exportPosts(s), [{ name: 'a1', patches: null, layout: 'auto', base_model_name_or_path: null, replace: false, dry_run: true }]);
  // p1 unticked and p3 (switched off) ticked: shown newest first, sent in stack order
  change(labelled(view, 'export p1'), false); await settle();
  change(labelled(view, 'export p3'), true); await settle();
  go('Preview (dry run)').click(); await settle();
  assert.deepEqual(exportPosts(s)[1].patches, ['p2', 'p3']);
  change(labelled(view, 'layout'), 'per_module');
  type(labelled(view, 'base model'), 'tiny-random/qwen2.5');
  go('Export').click(); await settle();
  assert.deepEqual(exportPosts(s)[2], { name: 'a1', patches: ['p2', 'p3'], layout: 'per_module', base_model_name_or_path: 'tiny-random/qwen2.5', replace: false, dry_run: false });
  // nothing ticked: both disabled, and no request
  change(labelled(view, 'export p2'), false); await settle();
  change(labelled(view, 'export p3'), false); await settle();
  assert.equal(go('Export').disabled, true);
  assert.equal(go('Export').title, 'tick at least one patch');
  assert.equal(exportPosts(s).length, 3);
});

test('U4: an "exists" refusal shows the replace box; ticked, replace is sent; another name hides it again', async () => {
  const s = await openOutput(makeApp(HF), { 'POST /api/export_adapter': (body) => (body.dry_run ? EXPORT_PREVIEW : body.replace ? EXPORT_WRITTEN : refusal(A8)) });
  type(labelled(view, 'adapter name'), 'a1');
  assert.equal(labelled(view, 'replace the adapter'), null, 'no box before the refusal');
  go('Export').click(); await settle();
  assert.equal(byId('dirs-export').querySelector('[role=alert]').textContent, "an adapter named 'a1' exists in this workspace; pass replace, or choose another name");
  const box = labelled(view, 'replace the adapter');
  assert.ok(box, 'the box appears');
  assert.equal(box.checked, false);
  go('Preview (dry run)').click(); await settle();
  assert.equal(exportPosts(s)[1].replace, false, 'not ticked: not sent');
  change(labelled(view, 'replace the adapter'), true);
  go('Export').click(); await settle();
  assert.deepEqual(exportPosts(s)[2], { name: 'a1', patches: null, layout: 'auto', base_model_name_or_path: null, replace: true, dry_run: false });
  assert.equal(labelled(view, 'replace the adapter'), null, 'written: the box is gone');
  // a refusal for 'a1', then the name changed: no box, replace false
  s.table['POST /api/export_adapter'] = (body) => (body.replace ? EXPORT_WRITTEN : refusal(A8));
  go('Export').click(); await settle();
  change(labelled(view, 'replace the adapter'), true);
  const name = labelled(view, 'adapter name');
  type(name, 'a2'); change(name); await settle();
  assert.equal(labelled(view, 'replace the adapter'), null);
  go('Export').click(); await settle();
  assert.deepEqual([exportPosts(s).at(-1).name, exportPosts(s).at(-1).replace], ['a2', false]);
});

test('U5: the result card: where it was written, three files with 12-hex hashes, the modules, the notes; a preview says nothing was written', async () => {
  await openOutput(makeApp(HF), { 'POST /api/export_adapter': (body) => (body.dry_run ? EXPORT_PREVIEW : EXPORT_WRITTEN) });
  type(labelled(view, 'adapter name'), 'a1');
  go('Export').click(); await settle();
  const card = byId('dirs-export-result');
  assert.equal(card.querySelector('b').textContent, 'Adapter written to <workspace>/export/adapters/a1');
  assert.deepEqual(rowsOf(byId('dirs-export-files')), [['adapter_config.json', '712 bytes', '0a1b2c3d4e5f'],
    ['adapter_model.safetensors', '24.8 KiB', '9f8e7d6c5b4a'], ['wsp_adapter.json', '1.4 KiB', '1234abcd5678']]);
  assert.deepEqual(rowsOf(byId('dirs-export-modules')), [['model.layers.1.self_attn.o_proj', '3', 'exact', 'p1:lowrank×3']]);
  assert.deepEqual([...card.querySelectorAll('[role=note]')].map((n) => n.textContent), [EXPORT_WRITTEN.notes[0]]);
  go('Preview (dry run)').click(); await settle();
  assert.equal(byId('dirs-export-result').querySelector('b').textContent, 'Preview: nothing written');
  assert.equal(byId('dirs-export-files'), null, 'a dry run has no files');
  assert.deepEqual(rowsOf(byId('dirs-export-modules'))[0].slice(0, 3), ['model.layers.1.self_attn.o_proj', '3', 'exact']);
});

// ------------------------------------------------------------------------------------------------ U6 to U10: records and the re-run
const NONE = { record: null, stale: false, checks: null, environment: null, remeasure: null, remeasure_unavailable: null, heretic: null };
const checkRows = (r) => [...r.querySelector('table.dirs-checks').children].slice(1).filter((tr) => tr.children.length === 4);

test('U6: Show record: one GET; the calls, hashes and made on; the failing check marked; a high severity a warn pill; Heretic: the reasons, or the link', async () => {
  const s = await openOutput(makeApp(HF), { 'GET /api/patches/p1/record': RECORD_ANSWER, 'GET /api/patches/p3/record': NONE });
  await showRecord();
  assert.deepEqual(s.gets().filter((u) => u.startsWith('/api/patches/')), ['/api/patches/p1/record']);
  assert.deepEqual(s.posts(), []);
  const r = record();
  assert.deepEqual(rowsOf(r.querySelector('table.dirs-calls')).map((x) => x.slice(0, 3)), [['ablate_direction', 'refusal', 'ablation']]);
  assert.ok(r.querySelector('table.dirs-calls details pre').textContent.includes('"row_normalization": "full"'), 'the parameters as JSON');
  assert.equal(r.querySelector('.dirs-measurements').textContent,
    'refusal: meta aa11bb22cc33, means dd44ee55ff66, module inputs and outputs none, prompts {"good":"c505","bad":"1c61"}');
  assert.ok(r.textContent.includes('made on: the base · pieces fedcba987654'));
  assert.deepEqual(checkRows(r).map((tr) => [...tr.children].map((c) => c.textContent).concat(tr.className)),
    [['base', '{"kind":"wsp"}', '{"kind":"wsp"}', 'refuse', ''], ['made_on', 'the base', 'p1', 'refuse', 'dirs-bad'],
      ['measurement:refusal', checkRows(r)[2].children[1].textContent, checkRows(r)[2].children[1].textContent, 'refuse', '']]);
  assert.ok(checkRows(r)[2].children[1].textContent.includes('"meta_sha256":"aa11bb22cc33"'), 'a digest cut to 12');
  assert.ok(r.querySelector('table.dirs-checks').textContent.includes(RECORD_ANSWER.checks[1].message), "the check's message");
  const env = [...r.querySelector('table.dirs-environment').children].slice(1);
  assert.deepEqual(env.map((tr) => [tr.children[0].textContent, tr.children[3].querySelector('span').className, tr.className]),
    [['torch', 'pill warn', 'dirs-differs'], ['python', 'pill', ''], ['runner', 'pill', '']]);
  assert.equal(env[2].children[2].textContent, 'not loaded yet', 'the runner of a model not loaded is not a difference');
  assert.equal(env[0].children[2].textContent, '2.9.0');
  assert.equal(r.querySelector('a'), null, 'not runnable: no link');
  assert.equal(r.querySelector('.dirs-heretic').textContent, 'Heretic cannot run this record:' + RECORD_ANSWER.heretic.reasons[0]);
  // runnable: the link to the Heretic file, and no reasons
  s.table['GET /api/patches/p1/record'] = { ...RECORD_ANSWER, heretic: { runnable: true, reasons: [] } };
  await showRecord();
  const link = record().querySelector('a[href="/api/patches/p1/record/heretic"][download="reproduce.json"]');
  assert.ok(link, 'the link');
  assert.equal(link.textContent, 'Heretic reproduce.json');
  assert.equal(record().querySelector('.dirs-heretic'), null);
  // a patch without a record: section 7's text
  labelled(view, 'record p3').click(); await settle();
  assert.equal(byId('dirs-record-p3').textContent, NO_RECORD);
  assert.deepEqual(s.posts(), [], 'reading records sends nothing else');
});

test('U6b: the Heretic file without a Hub repository: needs_model shows two fields and the link with the typed values encoded; runnable the plain link; otherwise the reasons only', async () => {
  const reason = 'no Hub repository is known for this checkpoint: pass model (and model_commit)';
  const answerWith = (heretic) => ({ ...RECORD_ANSWER, heretic });
  const s = await openOutput(makeApp(HF), { 'GET /api/patches/p1/record': answerWith({ runnable: false, reasons: [reason], needs_model: true }) });
  await showRecord();
  const links = () => [...record().querySelectorAll('a')];
  assert.deepEqual(links(), [], 'no model id typed: no link');
  assert.equal(record().querySelector('.dirs-heretic').textContent, 'Heretic cannot run this record:' + reason);
  assert.ok(record().textContent.includes("the repository this checkpoint came from, for Heretic's settings.model"), 'the hint');
  type(labelled(record(), 'Hub model id'), 'a b/c?d');
  assert.equal(links().length, 1);
  assert.equal(links()[0].getAttribute('href'), '/api/patches/p1/record/heretic?model=a%20b%2Fc%3Fd');
  assert.equal(links()[0].getAttribute('download'), 'reproduce.json');
  assert.equal(links()[0].textContent, 'Heretic reproduce.json');
  type(labelled(record(), 'Hub commit'), '0123abc&x=1');
  assert.equal(links()[0].getAttribute('href'), '/api/patches/p1/record/heretic?model=a%20b%2Fc%3Fd&model_commit=0123abc%26x%3D1');
  type(labelled(record(), 'Hub model id'), '');
  assert.deepEqual(links(), [], 'the model field emptied: no link');
  type(labelled(record(), 'Hub model id'), '   ');
  assert.deepEqual(links(), [], 'spaces only: no link');
  // the typed values are kept when the record is drawn again
  type(labelled(record(), 'Hub model id'), 'tiny-random/qwen2.5');
  await showRecord();
  assert.equal(labelled(record(), 'Hub model id').value, 'tiny-random/qwen2.5');
  assert.equal(links()[0].getAttribute('href'), '/api/patches/p1/record/heretic?model=tiny-random%2Fqwen2.5&model_commit=0123abc%26x%3D1');
  // runnable: the plain link and no fields
  s.table['GET /api/patches/p1/record'] = answerWith({ runnable: true, reasons: [], needs_model: false });
  await showRecord();
  assert.deepEqual(links().map((a) => [a.getAttribute('href'), a.getAttribute('download')]), [['/api/patches/p1/record/heretic', 'reproduce.json']]);
  assert.equal(labelled(record(), 'Hub model id'), null);
  // neither (needs_model false, or absent): the reasons, no fields, no link
  for (const heretic of [{ runnable: false, reasons: [reason], needs_model: false }, { runnable: false, reasons: [reason] }]) {
    s.table['GET /api/patches/p1/record'] = answerWith(heretic);
    await showRecord();
    assert.deepEqual(links(), []);
    assert.equal(labelled(record(), 'Hub model id'), null);
    assert.equal(record().querySelector('.dirs-heretic').textContent, 'Heretic cannot run this record:' + reason);
  }
  assert.deepEqual(s.posts(), []);
  assert.ok(!s.gets().some((u) => u.includes('/record/heretic')), 'the file is fetched only by the link');
});

test('U7: Switch patch p1 off: the patch route, then weightsChanged and the record read again; no button for a patch that is off', async () => {
  let n = 0;
  const afterOff = PATCHES.patches.map((p) => (p.id === 'p1' ? { ...p, enabled: false } : p));
  const off = { ok: true, missing: false, status: { version: 1 }, patches: afterOff };
  const app = makeApp(HF);
  shellReads(app, () => afterOff);
  const s = await openOutput(app, { 'GET /api/patches/p1/record': () => (n++ ? RECORD_ANSWER_OFF : RECORD_ANSWER), 'POST /api/patch': off,
    'GET /api/patches/p3/record': RECORD_ANSWER });
  await showRecord();
  assert.deepEqual(s.posts(), []);
  button(record(), 'Switch patch p1 off').click(); await settle();
  assert.deepEqual(s.posts().map((c) => [c.url, c.body]), [['/api/patch', { action: 'set', id: 'p1', enabled: false }]]);
  assert.equal(app.calls.weightsChanged, 1);
  const urls = s.calls.map((c) => `${c.method} ${c.url}`);
  assert.ok(urls.lastIndexOf('GET /api/patches/p1/record') > urls.indexOf('POST /api/patch'), urls.join(', '));
  assert.equal(n, 2, 'the record read again');
  assert.equal(button(record(), 'Switch patch p1 off'), null, 'its made_on check passes now');
  assert.equal(labelled(view, 'export p1').checked, false, "p1 is off in the shell's list read again: not ticked by default any more");
  assert.ok(!s.gets().includes('/api/patches'), 'the view reads no list of its own');
  // p3 is off already: its failing made_on check has no switch-off button
  labelled(view, 'record p3').click(); await settle();
  assert.equal(checkRows(byId('dirs-record-p3'))[1].className, 'dirs-bad');
  assert.equal(button(byId('dirs-record-p3'), 'Switch patch p3 off'), null);
});

const DIFFERENT = { verdict: 'different', digest: '2'.repeat(64), differences: [{ tensor: 'blocks.0.attn.W_O', relative: 0.0123, max_abs: 0.01 }] };
test('U8: Re-run: the dry-run body; each verdict as text; different shows the box; Commit sends dry_run false and the box; a commit calls weightsChanged', async () => {
  const app = makeApp(HF);
  const held = hold();
  let answer = held;
  const s = await openOutput(app, { 'GET /api/patches/p1/record': RECORD_ANSWER_OFF, 'POST /api/patches/p1/reproduce': () => answer });
  await showRecord();
  button(record(), 'Re-run (dry run)').click(); await settle();
  assert.deepEqual(reproducePosts(s), [{ measurement: null, dry_run: true, message: null, enforce_scope: false, accept_different: false }]);
  assert.equal(byId('dirs-job').querySelector('[role=status]').textContent, 'Re-running patch p1 (dry run); this cannot be stopped.');
  assert.equal(button(byId('dirs-job'), 'Stop'), null, 'a re-run has no stop point');
  assert.equal(button(record(), 'Re-run (dry run)').disabled, true, 'one job at a time');
  held.release(REPRODUCE()); await settle();
  const status = () => record().querySelector('.dirs-rerun-result [role=status]').textContent;
  assert.equal(status(), 'The re-run made the same edit (pieces fedcba987654).');
  assert.equal(byId('dirs-job').textContent, 'The re-run made the same edit (pieces fedcba987654).');
  assert.equal(labelled(record(), 'accept a different edit'), null);
  assert.ok(button(record(), 'Commit the re-run'));
  answer = REPRODUCE({ verdict: 'within_rounding', digest: '1'.repeat(64), bound: 1e-5,
    differences: [{ tensor: 'blocks.0.attn.W_O', relative: 3.2e-6, max_abs: 1e-7 }, { tensor: 'blocks.1.attn.W_O', relative: 1.5e-6, max_abs: 1e-7 }] });
  button(record(), 'Re-run (dry run)').click(); await settle();
  assert.equal(status(), 'The re-run differs from the recorded edit by rounding only (largest relative difference 3.2e-6, bound 1.0e-5).');
  answer = REPRODUCE(DIFFERENT);
  button(record(), 'Re-run (dry run)').click(); await settle();
  assert.equal(status(), 'The re-run made a different edit (largest relative difference 0.012).');
  assert.equal(labelled(record(), 'accept a different edit').checked, false, 'the box, unticked');
  // a tensor without a ratio (the server sends null, never a non-finite number) is named instead
  const lone = (only_in) => verdictText({ verdict: 'different', differences: [{ tensor: 'a', relative: 0.5, max_abs: 1, only_in: null }, { tensor: 'b', relative: null, max_abs: 1, only_in }] });
  assert.equal(lone('recorded'), 'The re-run made a different edit (b is changed by the recorded edit only).');
  assert.equal(lone('rerun'), 'The re-run made a different edit (b is changed by the re-run only).');
  assert.equal(lone(null), 'The re-run made a different edit (b has a recorded change of zero).');
  // Commit with the box unticked: accept_different false, and the refusal shown
  const refused = 'the re-run made a different edit (largest relative difference 0.0123 in blocks.0.attn.W_O). The library versions are those it was made with, so this is not expected; report it. Pass accept_different to commit it anyway';
  answer = { ...REPRODUCE(DIFFERENT), ok: false, error: refused, dry_run: false };
  button(record(), 'Commit the re-run').click(); await settle();
  assert.deepEqual(reproducePosts(s).at(-1), { measurement: null, dry_run: false, message: null, enforce_scope: false, accept_different: false });
  assert.equal(record().querySelector('.dirs-rerun-result [role=alert]').textContent, `Not committed: ${refused}`);
  assert.equal(app.calls.weightsChanged, 0);
  // ticked: accept_different true; committed: weightsChanged, the report, the patches read again
  answer = REPRODUCE({ ...DIFFERENT, committed: true, dry_run: false, patch: 'p4' });
  change(labelled(record(), 'accept a different edit'), true);
  shellReads(app, () => [...PATCHES.patches, { ...PATCHES.patches[0], id: 'p4', message: 're-run of patch p1',
    source: { kind: 'wscript', script: 'x', origin: { kind: 'reproduction', of: 'p1', verdict: 'different' } } }]);
  button(record(), 'Commit the re-run').click(); await settle();
  assert.deepEqual(reproducePosts(s).at(-1), { measurement: null, dry_run: false, message: null, enforce_scope: false, accept_different: true });
  assert.equal(app.calls.weightsChanged, 1);
  assert.equal(record().querySelector('.dirs-rerun-result .card b').textContent, 'Committed patch p4');
  assert.equal(byId('dirs-job').textContent, 'Committed the re-run of patch p1 as patch p4.');
  assert.equal(button(record(), 'Commit the re-run'), null);
  assert.equal(labelled(record(), 'accept a different edit'), null);
  assert.ok(labelled(view, 'export p4'), "the new patch is listed once the shell read the list again");
  assert.ok(byId('dirs-output-patches').textContent.includes('re-run of patch p1: different'), 'with its origin');
  assert.ok(!s.gets().includes('/api/patches'), 'the view reads no list of its own');
  // an identical dry run committed: accept_different stays false (no box was shown)
  answer = REPRODUCE();
  button(record(), 'Re-run (dry run)').click(); await settle();
  answer = REPRODUCE({ committed: true, dry_run: false, patch: 'p5', record_error: 'OSError: disk full' });
  button(record(), 'Commit the re-run').click(); await settle();
  assert.equal(reproducePosts(s).at(-1).accept_different, false);
  assert.equal(record().querySelector('.dirs-rerun-result .pill.warn').textContent, 'Committed, but its reproduction record could not be written: OSError: disk full');
  // a refused re-run (a check): its text
  answer = refusal(`RecordError: ${RECORD_ANSWER.checks[1].message}`);
  button(record(), 'Re-run (dry run)').click(); await settle();
  assert.equal(record().querySelector('.dirs-rerun [role=alert]').textContent, RECORD_ANSWER.checks[1].message);
  assert.equal(button(record(), 'Commit the re-run'), null, 'nothing to commit');
});

test('U9: Measure again: the remeasure GET, the measure request with that body and a stream, then the re-run with that measurement and the means line', async () => {
  const app = makeApp(HF);
  const measure = hold();
  const s = await openOutput(app, { 'GET /api/patches/p1/record': RECORD_ANSWER_OFF, 'GET /api/patches/p1/record?remeasure=true': RECORD_ANSWER_REMEASURE,
    'POST /api/directions/measure': measure, 'POST /api/patches/p1/reproduce': REPRODUCE({ means: { equal: true, relative: { good: 0, bad: 0 } } }) });
  await showRecord();
  assert.ok(!s.gets().some((u) => u.includes('remeasure')), 'the remeasure body is read on the click only');
  button(record(), 'Measure again from the record').click(); await settle();
  assert.ok(s.gets().includes('/api/patches/p1/record?remeasure=true'));
  assert.equal(measure.calls.length, 1);
  const { stream, ...sent } = measure.calls[0].body;
  assert.deepEqual(sent, REMEASURE_BODY);
  assert.match(stream, /^dirs-measure-/);
  assert.equal(app.keep.dirs.job.stream, stream, "the job slot's stream");
  assert.equal(byId('dirs-job').querySelector('[role=status]').textContent, "Measuring 'refusal-rp1': starting; this loads the model if it is not loaded.");
  assert.ok(button(byId('dirs-job'), 'Stop'), 'the measurement can be stopped');
  assert.deepEqual(reproducePosts(s), [], 'nothing is re-run before the measurement is made');
  const lists = s.gets().filter((u) => u === '/api/directions').length;
  measure.release({ name: 'refusal-rp1', seconds: 1.5, n_layers: 2, d_model: 24 });
  await settle(20);
  assert.deepEqual(reproducePosts(s), [{ measurement: 'refusal-rp1', dry_run: true, message: null, enforce_scope: false, accept_different: false }]);
  assert.ok(s.gets().filter((u) => u === '/api/directions').length > lists, 'the measurements are listed again');
  const means = () => record().querySelector('.dirs-rerun-result .dirs-means').textContent;
  assert.equal(means(), "The new measurement's means equal the recorded ones.");
  // the means differ; Commit keeps the measurement
  s.table['POST /api/directions/measure'] = { name: 'refusal-rp1', seconds: 1.5 };
  s.table['POST /api/patches/p1/reproduce'] = REPRODUCE({ means: { equal: false, relative: { good: 2.5e-7, bad: 4e-7 } } });
  button(record(), 'Measure again from the record').click(); await settle(20);
  assert.equal(means(), "The new measurement's means differ by at most 4.0e-7 (relative).");
  button(record(), 'Commit the re-run').click(); await settle();
  assert.deepEqual(reproducePosts(s).at(-1), { measurement: 'refusal-rp1', dry_run: false, message: null, enforce_scope: false, accept_different: false });
  // the means snapshot missing: the answer's notes under the verdict, and no fixed text; without notes, the fixed text
  s.table['POST /api/patches/p1/reproduce'] = REPRODUCE({ means: null, notes: ['the recorded means are not in the workspace'] });
  button(record(), 'Measure again from the record').click(); await settle(20);
  const result = record().querySelector('.dirs-rerun-result');
  assert.deepEqual([...result.querySelectorAll('[role=note]')].map((n) => n.textContent), ['the recorded means are not in the workspace']);
  assert.equal(result.children[1].getAttribute('role'), 'note', 'right under the verdict');
  assert.equal(result.querySelector('.dirs-means'), null);
  s.table['POST /api/patches/p1/reproduce'] = REPRODUCE({ means: null, notes: [] });
  button(record(), 'Measure again from the record').click(); await settle(20);
  assert.equal(means(), 'The recorded means are not in the workspace.');
  // a stopped measurement: no re-run
  const before = reproducePosts(s).length;
  s.table['POST /api/directions/measure'] = { stopped: true, done: 1, total: 5 };
  button(record(), 'Measure again from the record').click(); await settle(20);
  assert.equal(reproducePosts(s).length, before);
  assert.equal(byId('dirs-job').textContent, 'Stopped after 1 of 5 prompts; nothing was saved.');
  // the prompts snapshot missing: the reason, and no measure request
  const reason = "the prompts of measurement 'refusal' are not in the workspace (snapshot 888888888888 is missing)";
  s.table['GET /api/patches/p1/record?remeasure=true'] = { ...RECORD_ANSWER_OFF, remeasure: [{ measurement: 'refusal', body: null }], remeasure_unavailable: [reason] };
  const measured = s.posts().filter((c) => c.url === '/api/directions/measure').length;
  button(record(), 'Measure again from the record').click(); await settle();
  assert.equal(record().querySelector('.dirs-rerun [role=alert]').textContent, reason);
  assert.equal(s.posts().filter((c) => c.url === '/api/directions/measure').length, measured);
  // a record that cannot be rebuilt has no such button
  s.table['GET /api/patches/p1/record'] = { ...RECORD_ANSWER_OFF, record: { ...RECORD_ANSWER_OFF.record, rebuildable: false } };
  await showRecord();
  assert.equal(button(record(), 'Measure again from the record'), null);
  assert.ok(button(record(), 'Re-run (dry run)'));
});

test('U10: patchOrigin of a re-run: "re-run of patch <of>: <verdict>"', () => {
  const line = patchOrigin({ source: { kind: 'wscript', script: 'x', origin: { kind: 'reproduction', of: 'p1', verdict: 'identical' } } });
  assert.equal(line.textContent, 're-run of patch p1: identical');
  assert.equal(patchOrigin({ source: { kind: 'wscript', script: 'x', origin: { kind: 'reproduction', of: 'p2', verdict: 'different' } } }).textContent, 're-run of patch p2: different');
});

test('X2 (Output): a patch message, a note, a module name, a Heretic reason, a check message and a refusal are shown as typed', async () => {
  const EVIL = '<img src=x onerror=alert(1)>';
  const exactly = (root) => nodes(root).filter((n) => n.textContent === EVIL && [...n.childNodes].every((c) => c.nodeType === 3));
  const patches = PATCHES.patches.map((p) => (p.id === 'p1' ? { ...p, message: EVIL, author: EVIL } : p));
  const written = { ...EXPORT_WRITTEN, notes: [EVIL], modules: [{ ...EXPORT_WRITTEN.modules[0], module: EVIL }] };
  const rec = { ...RECORD_ANSWER, heretic: { runnable: false, reasons: [EVIL] }, checks: RECORD_ANSWER.checks.map((c) => (c.key === 'made_on' ? { ...c, message: EVIL } : c)),
    environment: [{ name: EVIL, then: EVIL, now: '1', severity: 'low', same: false }] };
  await openOutput(makeApp(HF), { 'POST /api/export_adapter': written, 'GET /api/patches/p1/record': rec,
    'POST /api/patches/p1/reproduce': refusal(`RecordError: ${EVIL}`) }, { patches });
  assert.ok(exactly(byId('dirs-output-patches')).filter((n) => n.tag === 'span').length === 2, 'the message and the author');
  type(labelled(view, 'adapter name'), 'a1');
  go('Export').click(); await settle();
  assert.ok(exactly(byId('dirs-export-modules')).some((n) => n.tag === 'td'), 'the module name');
  assert.ok(exactly(byId('dirs-export-result')).some((n) => n.getAttribute('role') === 'note'), 'the note');
  await showRecord();
  assert.ok(exactly(record()).some((n) => n.tag === 'li'), 'the Heretic reason');
  assert.ok(exactly(record().querySelector('table.dirs-checks')).some((n) => n.tag === 'td'), "the check's message");
  assert.equal(exactly(record().querySelector('table.dirs-environment')).filter((n) => n.tag === 'td').length, 2, 'an environment row');
  button(record(), 'Re-run (dry run)').click(); await settle();
  assert.ok(exactly(record()).some((n) => n.getAttribute('role') === 'alert'), 'the refusal');
  assert.ok(!nodes(view).some((n) => n.tag === 'img'), 'no img element was made');
});
