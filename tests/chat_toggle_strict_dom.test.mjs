// The Run tab's "Send:" toggle (one message to both models, or a separate message to each) against the strict fake DOM
// of tests/strict_dom.mjs, whose `children` is an element-only collection without array methods, as in a browser.
// This choice was easy to miss in the running IDE: it was a small select inside the settings
// row. It is now a two-way toggle on its own line just above the message box(es). No server, model or provider.
import test from 'node:test';
import assert from 'node:assert/strict';
import { runView } from '../web/js/run.js';
import { installStrictDom, mount, all, flush } from './strict_dom.mjs';

const { view } = installStrictDom();
const tick = () => new Promise((r) => setImmediate(r));
const settle = async () => { for (let k = 0; k < 6; k++) await tick(); };
const json = (data, ok = true) => ({ ok, statusText: 'Bad Request', headers: { get: () => 'application/json' }, json: async () => data });
const runtime = { generation: { default_output: 16, output_limit: 64, context_limit: 128 } };
const generation = { requested_output: 8, effective_output: 8, prompt_tokens: 2, context: 128 };
const TPL = { user: 'User: ', assistant: 'Assistant: ', system: '', sep: '\n' };
const LABEL = 'Send one message to both models, or a separate message to each';
const CHOICE = { both: 'One message to both models', separate: 'A separate message to each model' };
const BOX = { both: 'Message to both models', base: 'Message to the base model', current: 'Message to the edited model' };

function makeApp(run = {}) {
  return { sel: 'model', tab: 'Run', variant: 'current', keep: { run: { mode: 'chat', chat: [], tpl: TPL, ...run } }, selected: [],
    state: { open: { name: 'fixture', kind: 'wsp' }, status: { can_run: true, version: 0 } }, rerender() {}, select() {} };
}
// The chat request is held until the test answers it; everything else answers at once.
function server() {
  const held = { bodies: [] };
  global.fetch = async (url, o = {}) => {
    if (url === '/api/generation/stop') return json({ ok: true, found: true });
    if (url !== '/api/chat') return json(runtime);
    held.body = JSON.parse(o.body); held.bodies.push(held.body);
    return new Promise((resolve) => { held.finish = (data, ok = true) => resolve(json(data, ok)); });
  };
  return held;
}
// Draw the tab the way the shell does: built, then mounted in the document.
async function draw(app) { const root = document.createElement('div'); await runView(root, app); mount(view, root); return root; }
const labelled = (root, label) => all(root).find((n) => n.getAttribute('aria-label') === label) || null;
const toggle = (root) => labelled(root, LABEL);
const halves = (root) => [...toggle(root).children];
const half = (root, v) => halves(root).find((b) => b.textContent === CHOICE[v]);
const pressed = (root) => halves(root).filter((b) => b.getAttribute('aria-pressed') === 'true').map((b) => b.textContent);
const buttonsNamed = (root, t) => all(root).filter((n) => n.tag === 'button' && n.textContent === t);
const conversation = (root, name) => labelled(root, `${name} conversation`);
const order = (root, node) => all(root).indexOf(node);
const type = (inp, v) => { inp.value = v; inp.dispatchEvent({ type: 'input', target: inp }); };

test('the toggle is drawn in chat mode, on its own line above the box, and not in completion mode', async () => {
  server();
  const completion = await draw(makeApp({ mode: 'completion', prompt: 'a b', n: 4, temp: 0, result: null }));
  assert.equal(toggle(completion), null, 'no Send toggle in completion mode');
  assert.doesNotMatch(completion.textContent, /One message to both models|A separate message to each model/);

  const root = await draw(makeApp());
  const t = toggle(root);
  assert.ok(t, 'the Send toggle is drawn in chat mode');
  assert.equal(t.role, 'group');
  assert.equal(t.className, 'seg');
  assert.deepEqual(halves(root).map((b) => [b.tag, b.textContent, b.getAttribute('aria-pressed'), b.className, b.disabled]),
    [['button', CHOICE.both, 'true', 'on', false], ['button', CHOICE.separate, 'false', '', false]]);
  // Its line says what it is, holds nothing else, and is not the settings row (the one with max tokens and clear).
  const line = t.parentElement;
  assert.equal(line.children.length, 2);
  assert.equal(line.children[0].textContent, 'Send:');
  assert.ok(!all(line).some((n) => n.tag === 'input' || n.tag === 'select'), 'nothing but the toggle on its line');
  const settings = buttonsNamed(root, 'clear')[0].parentElement;
  assert.ok(!settings.contains(t), 'not in the settings row');
  assert.doesNotMatch(settings.textContent, /messages/i);
  assert.ok(!all(root).some((n) => n.tag === 'select' && /message/i.test(n.getAttribute('aria-label') || '')), 'the old select is gone');
  // Below the steering panel and the conversation, directly above the message box.
  const box = labelled(root, BOX.both);
  const steering = all(root).find((n) => n.className === 'card' && /steer/i.test(n.textContent));
  assert.ok(steering && order(root, steering) < order(root, t), 'below the steering panel');
  assert.ok(order(root, t) < order(root, box), 'above the message box');
  const between = all(root).slice(order(root, t) + all(t).length, order(root, box)).filter((n) => n.tag !== 'div');
  assert.deepEqual(between.map((n) => n.tag), [], 'nothing drawn between the toggle and the box');
});

test('pressing "A separate message to each model" shows two boxes, each under its model, and then two conversations', async () => {
  const app = makeApp({ chat: [{ role: 'user', content: 'hi' }, { role: 'assistant', content: 'E1' }], baseChat: [{ role: 'user', content: 'hi' }, { role: 'assistant', content: 'B1' }] });
  const held = server();
  let root = await draw(app);
  assert.ok(labelled(root, BOX.both) && !labelled(root, BOX.base));
  half(root, 'separate').click();
  assert.equal(app.keep.run.chatSplit, true);
  root = await draw(app);
  assert.deepEqual(pressed(root), [CHOICE.separate]);
  assert.equal(half(root, 'separate').className, 'on');
  assert.equal(labelled(root, BOX.both), null, 'the one box is gone');
  const base = labelled(root, BOX.base), edited = labelled(root, BOX.current);
  assert.ok(base && edited, 'a box for each model');
  assert.equal(buttonsNamed(root, 'Send').length, 2);
  // Side by side: the two boxes are in the two halves of one split, each headed by the model it talks to.
  const split = base.parentElement.parentElement.parentElement;
  assert.equal(split.className, 'split');
  assert.equal(split.children.length, 2);
  assert.ok(split.children[0].contains(base) && split.children[1].contains(edited));
  assert.equal(split.children[0].children[0].textContent, 'To the base model');
  assert.equal(split.children[1].children[0].textContent, 'To the edited model');
  assert.ok(order(root, toggle(root)) < order(root, split), 'the toggle stays above the boxes');
  // A message to the base model alone: the two conversations now differ and are shown as two.
  type(base, 'only base');
  buttonsNamed(split.children[0], 'Send')[0].click(); await tick();
  assert.deepEqual(held.body.variants, ['base']);
  held.finish({ chat_template: false, generation, current: '', base: 'B2', variants: { base: 'complete' }, generations: { base: generation } }); await settle();
  root = await draw(app);
  assert.ok(conversation(root, 'base') && conversation(root, 'edited'), 'two conversations');
  assert.match(conversation(root, 'base').textContent, /only base.*B2/);
  assert.doesNotMatch(conversation(root, 'edited').textContent, /only base/);
  assert.deepEqual(pressed(root), [CHOICE.separate]);
  // And back to one box (the boxes are empty, so nothing is refused).
  half(root, 'both').click();
  root = await draw(app);
  assert.deepEqual(pressed(root), [CHOICE.both]);
  assert.ok(labelled(root, BOX.both) && !labelled(root, BOX.base) && !labelled(root, BOX.current));
  assert.equal(buttonsNamed(root, 'Send').length, 1);
  assert.match(root.textContent, /One message still goes to both conversations\./);
});

test('pressing back with two different drafts is refused, and the toggle still shows "separate"', async () => {
  const app = makeApp();
  server();
  let root = await draw(app);
  half(root, 'separate').click();
  root = await draw(app);
  type(labelled(root, BOX.base), 'for base');
  type(labelled(root, BOX.current), 'for edited');
  half(root, 'both').click();
  assert.equal(app.keep.run.chatSplit, true, 'the switch is refused');
  root = await draw(app);
  assert.deepEqual(pressed(root), [CHOICE.separate], 'the toggle shows the mode in force');
  assert.equal(half(root, 'separate').className, 'on');
  assert.equal(half(root, 'both').className, '');
  const alert = all(root).find((n) => n.role === 'alert' && n.textContent);
  assert.equal(alert.textContent, 'The two boxes hold different messages. Send or clear one of them before switching to one message to both.');
  assert.equal(labelled(root, BOX.base).value, 'for base');
  assert.equal(labelled(root, BOX.current).value, 'for edited');
  // The same message in both boxes is one message: then the switch goes through and the message is in the one box.
  type(labelled(root, BOX.current), 'for base');
  half(root, 'both').click();
  root = await draw(app);
  assert.deepEqual(pressed(root), [CHOICE.both]);
  assert.equal(labelled(root, BOX.both).value, 'for base');
  assert.ok(!all(root).some((n) => n.role === 'alert' && n.textContent), 'the refusal is cleared');
  // And forward again: the one message goes into both column boxes.
  half(root, 'separate').click();
  root = await draw(app);
  assert.equal(labelled(root, BOX.base).value, 'for base');
  assert.equal(labelled(root, BOX.current).value, 'for base');
});

test('the toggle is disabled while a chat request is pending, and enabled again when it ends', async () => {
  const app = makeApp();
  const held = server();
  let root = await draw(app);
  type(labelled(root, BOX.both), 'hello');
  buttonsNamed(root, 'Send')[0].click(); await tick();
  assert.ok(held.body, 'the request is pending');
  root = await draw(app);
  assert.equal(halves(root).length, 2);
  assert.ok(halves(root).every((b) => b.disabled), 'both halves disabled');
  assert.deepEqual(pressed(root), [CHOICE.both], 'still showing the mode in force');
  half(root, 'separate').click();                                  // a browser ignores a click on a disabled button
  half(root, 'separate').dispatchEvent({ type: 'click' });         // and the page ignores one that gets through anyway
  assert.equal(!!app.keep.run.chatSplit, false);
  held.finish({ chat_template: false, generation, current: 'E', base: 'B', variants: { current: 'complete', base: 'complete' } }); await settle();
  root = await draw(app);
  assert.ok(halves(root).every((b) => !b.disabled), 'enabled after the reply');
  half(root, 'separate').click();
  assert.equal(app.keep.run.chatSplit, true);
  await flush();
});
