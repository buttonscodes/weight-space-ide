// Run: completion and chat, base and edited side by side.   Trace: what happens inside one forward pass.
import { h, api, fmt, heatmap, hbars, tip, esc, tabs, field, seq, legend, visible } from './util.js';
import { steerPanel, steerRequest, steerState, steerLabel, steeredLine, steeringStatus } from './steer.js';

function defaultPrompt(app) {
  const t = app.state.open.name;
  if (t.startsWith('facts')) return '<bos> Alice lives in';
  if (t.startsWith('shakespeare')) return 'ROMEO:\n';
  if (t === 'blank-reverse') return '<bos> a b c d |';
  if (t === 'blank-modadd') return '3 + 5 =';
  if (t.startsWith('tinystories')) return 'Once upon a time, there was a little girl named';
  return 'The capital of France is';
}

// Views that need forward passes: say why they are unavailable instead of failing.
function cannotRun(el, app) {
  if (app.state.status.can_run) return false;
  const o = app.state.open;
  const caps = o.capabilities;
  el.append(h('div', { class: 'card', style: { marginTop: '12px', maxWidth: '720px' } },
    h('b', {}, caps ? 'Generation is unavailable for this configuration.' : 'This model is open for viewing and editing only.'),
    h('p', {}, o.remote ? `Its weights (${(o.bytes / 1e9).toFixed(2)} GB) are still on the Hub and are being read on demand. Running it needs them on disk and enough memory for the whole model: use "Pull to run" in the top bar.`
      : (app.state.status.run_error || caps?.generation?.reason || 'It has no configuration that says how to run it.')),
    h('p', { class: 'muted' }, caps ? `${caps.representation}. Open Capabilities for the supported inspection, edit and export operations.`
      : 'The hierarchy, floating-point weight views, chunking and edits remain available without an inference runner.')));
  return true;
}

// The chat template options of the Run tab's thinking choice; Probe's "as a chat message" renders with the same.
export function thinkingOptions(thinking) {
  return !thinking || thinking === 'model default' ? {} : thinking === 'off' ? { enable_thinking: false } : { enable_thinking: true, reasoning_effort: thinking };
}

function nextBars(next, other) {
  const o = Object.fromEntries((other || []).map((x) => [x[0], x[1]]));
  return hbars((next || []).map((x) => ({ label: visible(x[0]), value: x[1], title: `${visible(JSON.stringify(x[0]))}  logit ${x[2]}` + (o[x[0]] != null ? `  (other model: ${fmt(o[x[0]])})` : '') })),
    { width: 300, signed: false, max: 1, labelW: 96, cls: 'bar plain', valueFmt: (v) => (100 * v).toFixed(1) + '%' });
}

function outputError(raw, minimum, maximum) {
  const n = typeof raw === 'string' && raw.trim() === '' ? NaN : Number(raw);
  return Number.isFinite(n) && Number.isSafeInteger(n) && n >= minimum && n <= maximum ? ''
    : `Enter a whole number from ${minimum} to ${maximum} output tokens under the active runtime limits. The saved output value is a default. The server fits your requested ceiling to the space remaining after the complete prompt/history.`;
}

function runtimeFigure(value, code, key = '') {
  if (value == null) return 'not available';
  if (typeof value === 'object') return Object.entries(value).map(([k,n]) => k.replaceAll('_',' ') + ': ' + runtimeFigure(n,code,k)).join('; ');
  if (typeof value !== 'number') return String(value);
  // Named fields first: a worker loss reports a process id, a return code and grace periods, not bytes.
  if (key.endsWith('pid') || key === 'returncode') return String(value);
  if (key.endsWith('seconds')) return value.toFixed(1)+' seconds';
  if (code.includes('expired') || code.includes('timeout')) return value.toFixed(1)+' seconds';
  if (code.includes('pressure')) return 'pressure level '+value;
  if (code.includes('threads')) return value+' threads';
  return value >= 2**30 ? (value/2**30).toFixed(2)+' GiB' : (value/2**20).toFixed(1)+' MiB';
}

function runtimeEvent(record, dismiss) {
  return h('div',{class:'card',role:'status'},h('b',{},`Runtime ${record.action}: ${record.code}`),
    h('p',{},record.message),
    record.measured != null ? h('p',{class:'small'},'Measured: '+runtimeFigure(record.measured,record.code)+'; limit: '+runtimeFigure(record.limit,record.code)) : null,
    h('p',{},(record.recommendations || []).join(' ')),
    dismiss ? h('button',{onclick:dismiss},'Dismiss runtime event') : null);
}

// ------------------------------------------------------------------------------------------------ live text
// While a Run or Chat request is pending the server publishes `generation` events for the request's
// stream id: how much of the text shown so far still stands (`keep`) and what to append (`text`).
// The edited model runs first, then the base model. The response replaces the live text.
// A completion and a chat turn can be pending together, so each mode has its own slot in `st.live`.
function startLive(st, mode, fields) {
  const id = globalThis.crypto?.randomUUID?.() || 'r' + Math.random().toString(36).slice(2) + Date.now().toString(36);
  const live = { id, current: { text: '', state: 'waiting' }, base: { text: '', state: 'waiting' }, paint: null, ...fields };
  (st.live || (st.live = {}))[mode] = live;
  return live;
}

function endLive(st, mode, live) {
  if (st.live?.[mode] === live) st.live[mode] = null;
}

function liveLabel(live, name) {
  const v = live[name];
  if (v.lost) return 'Live text is unavailable; the result appears when the request finishes.';
  if (v.state === 'complete') return 'complete';
  if (v.state === 'interrupted') return 'interrupted';
  if (v.state === 'failed') return 'stopped by an error';
  if (v.state === 'stopped') return 'stopped';
  if (live.stopping && v.state === 'started') return 'stopping…';  // a reply not yet started never will be
  if (v.state === 'started') return v.text ? 'generating…' : 'waiting for the first token…';
  // Before the steered reply starts: calibration and column-norm reads (steering events of this request).
  if (name === 'current' && live.steer && live.steering) return steeringStatus(live.steering) + (live.stopping ? ' (stopping after this step)' : '');
  return name === 'base' && live.variants?.[0] !== 'base' ? 'runs after the edited model' : 'starting…';
}

function liveNode(live, name) {
  const body = h('span', { style: { whiteSpace: 'pre-wrap' } }), status = h('div', { class: 'small muted', role: 'status' });
  const paint = () => { body.textContent = live[name].lost ? '' : live[name].text; status.textContent = liveLabel(live, name); };
  paint();
  return { body, status, paint };
}

// Stop: the server ends the reply being generated and skips the rest; the request then answers with
// `stopped`, and nothing of it is kept. The button stays "Stopping…" until that answer arrives. If the
// request finished first, its answer is an ordinary result and is shown as usual.
function stopControl(live, redraw) {
  const stop = async () => {
    if (live.stopping) return;
    live.stopping = true; live.stopError = '';
    live.paint?.(); redraw();
    try { await api('/api/generation/stop', { stream: live.id }); }
    catch (e) { live.stopping = false; live.stopError = 'Stop failed: ' + e.message.replace(/[.!?]+$/, '') + '. Try again.'; redraw(); }
  };
  return [h('button', { disabled: !!live.stopping, onclick: stop, style: { alignSelf: 'flex-end' } }, live.stopping ? 'Stopping…' : 'Stop'),
    live.stopError ? h('span', { role: 'alert', style: { color: 'var(--crit)', alignSelf: 'flex-end' } }, live.stopError) : null];
}

// The stopped chat message goes back into the box. Something typed there since it was sent is kept,
// after the stopped message, so nothing is lost silently.
export function restoredDraft(message, draft) {
  if (!draft || !draft.trim()) return message;
  if (draft.startsWith(message)) return draft;   // the box still holds the message, perhaps with more after it
  return message + ' ' + draft;
}

export function onGenerationEvent(ev, app) {
  const live = Object.values(app.keep.run?.live || {}).find((l) => l && l.id === ev.stream);
  if (!live || (ev.variant !== 'base' && ev.variant !== 'current')) return;
  const v = live[ev.variant];
  if (ev.state) v.state = ev.state;
  if (ev.lost) v.lost = true;
  if (typeof ev.text === 'string' && !v.lost) {
    if (!(ev.keep <= v.text.length)) v.lost = true;  // an earlier event never arrived: stop guessing
    else v.text = v.text.slice(0, ev.keep) + ev.text;
  }
  live.paint?.();
}

export async function runView(el, app, active = () => true) {
  const rt = await api('/api/runtime');
  if (!active()) return;
  const output = rt.generation?.default_output ?? 2048;
  const maxOutput = rt.generation?.output_limit ?? Number.MAX_SAFE_INTEGER;
  const guarded = app.state.open.kind === 'gguf' || app.state.open.capabilities?.format === 'gguf';
  const st = app.keep.run || (app.keep.run = { mode: 'completion', prompt: defaultPrompt(app), n: output, temp: 0, result: null, chat: [], tpl: { user: 'User: ', assistant: 'Assistant: ', system: '', sep: '\n' } });
  const again = () => { if (app.keep.run === st && app.tab === 'Run') app.rerender(); };
  el.append(h('div', { class: 'row' }, h('h2', {}, 'Run'), tabs(['completion', 'chat'], st.mode, (m) => { st.mode = m; again(); }),
    h('span', { class: 'muted' }, st.mode === 'chat' ? 'the base model and the edited model, each answering its own conversation' : 'the base model and the edited model on the same input')));
  if (guarded) {
    try {
      const limits = rt.limits || {};
      const b = rt.backend;
      el.append(h('div', {class:'card'}, h('b', {}, `Runtime: ${b?.selected || 'external harness'}`),
        h('p', {}, b ? `${b.reason}. ${b.blockers.join('; ')}` : rt.note),
        h('p', {}, rt.supervised ? `External memory supervision. Context ${limits.context || 'harness limit'}, output ${limits.new_tokens || 'harness limit'} tokens. ` +
          (rt.forecast ? `Forecast ${(rt.forecast.total / 2**30).toFixed(2)} GiB. ` : '') +
          (limits.resident_seconds ? `Idle timeout ${limits.resident_seconds}s after the last operation; active work is not expired.` : '') : 'Start with python -m wsp serve for managed supervision.'),
        rt.error || rt.reservation?.reason ? h('p', {style:{color:'var(--crit)'}}, rt.error || rt.reservation.reason) : null,
        h('p', {class:'muted'}, 'Run performs fresh preflight. No model is loaded by this tab. Use python -m wsp doctor for setup details.'),
        h('button', {onclick:async()=>{ await api('/api/runtime/unload',{}); await app.load(); }}, 'Unload / clear failed load')));
    } catch(e) { el.append(h('p',{style:{color:'var(--crit)'}},e.message)); }
  }
  if (rt.event) el.append(runtimeEvent(rt.event, async()=>{await api('/api/runtime/dismiss',{});again();}));
  for (const warning of rt.warnings || []) el.append(runtimeEvent(warning));
  if (cannotRun(el, app)) return;
  el.append(h('p', { class:'hint' }, `${rt.generation?.output_limit != null ? 'Active output limit '+maxOutput+' tokens; ' : ''}Saved response default ${output} tokens; ${rt.generation?.output_limit == null ? 'per-request output may use the remaining context' : 'the active output limit remains enforced'}${rt.generation?.context_limit ? `; prompt, complete history, template and response share ${rt.generation.context_limit} context tokens` : ''}. Saving a new response default applies it to the next request.`));
  const summary = rt.settings_summary;
  if (summary?.next_start) {
    // Each sentence follows wsp/settings.py: output applies when saved; context at the next model load (supervised) or start.
    const loaded = summary.loaded && summary.loaded.context !== summary.next_start.context
      ? ` The loaded model keeps context ${summary.loaded.context} until it is unloaded; the next load uses ${summary.next_start.context}.` : '';
    el.append(h('p', {class:'hint'}, `Profile: ${summary.profile || 'config defaults'}. Saved output ${summary.saved.new_tokens}, context ${summary.saved.context}; with existing launch overrides: output ${summary.next_start.new_tokens}, context ${summary.next_start.context}.${loaded}${summary.restart_required ? ' Saved changes are pending restart: this server has no memory supervisor to apply them to.' : ''}${summary.overridden.length ? ' Launch overrides mask: '+summary.overridden.join(', ')+'.' : ''}`));
  } else if (summary?.error) el.append(h('p',{class:'hint'},'Saved settings could not be read: '+summary.error));
  el.append(h('button',{onclick:()=>app.select(app.sel,'Settings')},'Open Settings'));

  if (st.mode === 'completion') {
    const ta = h('textarea', { rows: 3, value: st.prompt, oninput: (e) => { st.prompt = e.target.value; } });
    const validation = h('p',{role:'alert',style:{color:'var(--crit)'}},st.runValidation || '');
    const go = async () => {
      if (st.result?.busy || !active()) return;
      st.runValidation = outputError(st.n,0,maxOutput); validation.textContent = st.runValidation;
      if (st.runValidation) return;
      const steer = steerRequest(steerState(app));
      if (steer.problems.length) { st.runValidation = 'Steering: ' + steer.problems.join('; '); validation.textContent = st.runValidation; return; }
      st.result = { busy: true };
      const live = startLive(st, 'completion', { prompt: st.prompt, steer: steer.label });
      again();
      try {
        const r = await api('/api/run', { prompt: st.prompt, max_new_tokens: Number(st.n), temperature: st.temp, top_k: st.temp > 0 ? 40 : 0, seed: st.seed || 0, stream: live.id, ...(steer.fields || {}) });
        if (r.stopped) {
          // Nothing is kept. A prompt edited during the run stays as edited; an emptied box gets the run's prompt back.
          // A runtime event that came with the stop (the worker was also interrupted) stays visible.
          st.result = r.event ? { stopped: true, event: r.event } : { stopped: true };
          if (!st.prompt || !st.prompt.trim()) st.prompt = live.prompt;
        } else { st.result = r; st.version = app.state.status.version; }
      }
      catch (e) { st.result = { error: e.message, event:e.event }; }
      finally { endLive(st, 'completion', live); }
      again();
    };
    ta.addEventListener('keydown', (e) => { if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') go(); });
    el.append(h('div', { style: { margin: '8px 0' } }, ta), h('div', { class: 'row' },
      field('new tokens', h('input', { type: 'number', 'aria-label':'Completion output tokens', value: st.n, min: 0, max: maxOutput, step:1, style: { width: '70px' }, oninput: (e) => {
        st.n = e.target.value;
        if (st.runValidation) { st.runValidation = outputError(st.n,0,maxOutput); validation.textContent = st.runValidation; }
      } })),
      field('temperature', h('input', { type: 'number', value: st.temp, min: 0, max: 2, step: 0.1, style: { width: '70px' }, onchange: (e) => { st.temp = +e.target.value; } }), ),
      field('seed', h('input', { type: 'number', value: st.seed || 0, style: { width: '70px' }, onchange: (e) => { st.seed = +e.target.value; } })),
      h('button', { class: 'primary', disabled: st.result?.busy, onclick: go, style: { alignSelf: 'flex-end' } }, 'Run  (Ctrl+Enter)'),
      st.result?.busy && st.live?.completion ? stopControl(st.live.completion, again) : null,
      h('button', { style: { alignSelf: 'flex-end' }, onclick: () => { app.keep.trace = { ...(app.keep.trace || {}), prompt: st.prompt, result: null }; app.tab = 'Trace'; app.select(app.sel); } }, 'Trace this prompt →')));
    el.append(validation, steerPanel(app));   // the same panel and entries as in chat
    const r = st.result;
    if (!r) {
      el.append(h('p', { class:'hint' }, 'Choose a prompt and click Run. Opening this tab does not run the model.'));
      return;
    }
    if (r.busy) {
      const live = st.live?.completion;
      if (!live) { el.append(h('p', { class: 'muted' }, 'running…')); return; }
      const nodes = { base: liveNode(live, 'base'), current: liveNode(live, 'current') };
      const col = (title, n) => h('div', { class: 'card' }, h('h3', { style: { marginTop: 0 } }, title),
        h('div', { class: 'mono muted', style: { whiteSpace: 'pre-wrap' } }, live.prompt),
        h('div', { class: 'mono', style: { whiteSpace: 'pre-wrap', marginBottom: '10px', fontWeight: 'bold' } }, n.body), n.status, n === nodes.current ? steeredLine(live.steer) : null);
      el.append(h('p', { class: 'muted' }, 'running… the edited model first, then the base model'),
        h('div', { class: 'split', style: { marginTop: '6px' } }, col('base', nodes.base), col('edited', nodes.current)));
      live.paint = () => { nodes.base.paint(); nodes.current.paint(); };
      return;
    }
    if (r.event) el.append(runtimeEvent(r.event));
    if (r.stopped) { el.append(h('p', { class: 'hint', role: 'status' }, 'Run stopped; nothing was kept. Edit the prompt and run again.')); return; }
    if (r.error) { el.append(h('p', { style: { color: 'var(--crit)' } }, r.error)); return; }
    if (st.version !== app.state.status.version) {
      el.append(h('p', { class:'hint' }, 'Weights changed. Click Run to compare the updated model.'));
    }
    el.append(h('p', { class: 'small muted', style: { marginTop: '10px' } }, `${r.tokens.length} tokens: `, r.tokens.map((t) => h('span', { class: 'tok', style: { marginRight: '2px' } }, visible(t)))));
    if (r.generation && r.generation.effective_output < r.generation.requested_output) el.append(h('p',{class:'hint'},`Response ceiling reduced to ${r.generation.effective_output} token${r.generation.effective_output === 1 ? '' : 's'} to fit the remaining context; full prompt preserved.`));
    const same = r.base.status !== 'not_run' && r.current.status !== 'not_run' && !r.interrupted && r.base.text === r.current.text;
    // The edited reply ran with the steering the server reports; the base reply never is steered.
    const steered = r.steering ? steerLabel(r.steering.entries) : '';
    const col = (name, d, other) => h('div', { class: 'card' }, h('h3', { style: { marginTop: 0 } }, name),
      d === r.current ? steeredLine(steered) : steered ? h('div', { class: 'small muted' }, 'unsteered') : null,
      d.text != null ? h('div', { class: 'mono', style: { whiteSpace: 'pre-wrap', marginBottom: '10px' } }, h('span', { class: 'muted' }, r.prompt + r.join), h('b', {}, d.text)) : null,
      h('div', { class: 'small muted' }, d.status === 'not_run' ? 'Not run after interruption' : d.status === 'interrupted' ? 'Interrupted — partial output' : d === r.current && steered ? 'next token (steered)' : 'next token'), nextBars(d.next, other.next));
    el.append(h('div', { class: 'split', style: { marginTop: '6px' } }, col('base', r.base, r.current), col(same ? 'edited (same continuation)' : 'edited', r.current, r.base)));
    return;
  }
  // chat. Each model has its own conversation: st.chat is the edited model's (its replies carry the steering they
  // ran with), st.baseChat the base model's. One message goes to both models in one request (the default), or, with
  // separate messages, each column sends its own message to its own model. One request runs at a time.
  // Both conversations live in memory only (app.keep): a reload or opening another model starts them afresh.
  if (!st.baseChat) st.baseChat = [];
  const drafts = st.chatDrafts || (st.chatDrafts = { current: '', base: '' });
  const draftOf = (at) => (at === 'both' ? st.chatDraft : drafts[at]) || '';
  const setDraft = (at, v) => { if (at === 'both') st.chatDraft = v; else drafts[at] = v; };
  // Switching between one message and separate messages takes what is typed to the boxes now shown, so no message
  // is left in a hidden box: the one message goes into both column boxes; a column message (one, or the same in
  // both) goes into the one box. Two different column messages cannot become one, so that switch is refused with
  // a reason. The notices of the last send are cleared: they spoke of a box that is no longer there.
  const switchMode = (split) => {
    if (split === !!st.chatSplit) return;
    const b = drafts.base || '', c = drafts.current || '';
    if (!split && b.trim() && c.trim() && b !== c) {
      st.chatNotice = st.chatError = ''; st.chatAt = 'both';
      st.chatValidation = 'The two boxes hold different messages. Send or clear one of them before switching to one message to both.';
      return again();
    }
    if (split) { if ((st.chatDraft || '').trim()) drafts.base = drafts.current = st.chatDraft; st.chatDraft = ''; }
    else { st.chatDraft = b.trim() ? b : c; drafts.base = drafts.current = ''; }
    st.chatSplit = split; st.chatNotice = st.chatError = st.chatValidation = ''; st.chatAt = 'both';
    again();
  };
  const NAME = { current: 'edited', base: 'base' };
  el.append(h('p', { class: 'hint' }, st.templated === true ? 'Using the model\'s own chat template.'
    : 'If the model ships a chat template it is used. Otherwise this is completion with the turn prefixes below, cut off where the next user turn would start.'));
  el.append(h('div', { class: 'row', style: { margin: '6px 0' } },
    app.state.open.capabilities?.architecture === 'qwen35' ? field('thinking',
      h('select', { onchange: (e) => { st.thinking = e.target.value; } },
        ['model default', 'off', 'low', 'medium', 'xhigh'].map(v => h('option', { value:v, selected:(st.thinking || 'model default') === v }, v)))) : null,
    st.templated === true ? null : ['system', 'user', 'assistant'].map((k) => field(k + ' prefix', h('input', { type: 'text', value: st.tpl[k], style: { width: '140px' }, onchange: (e) => { st.tpl[k] = e.target.value; } }))),
    field('temperature', h('input', { type: 'number', value: st.ctemp ?? 0.7, min: 0, max: 2, step: 0.1, style: { width: '70px' }, onchange: (e) => { st.ctemp = +e.target.value; } })),
    field('max tokens', h('input', { type: 'number', 'aria-label':'Chat output tokens', value: st.cmax ?? output, min: 1, max: maxOutput, step:1, style: { width: '70px' }, oninput: (e) => {
      st.cmax = e.target.value;
      if (st.chatValidation) { st.chatValidation = outputError(st.cmax,1,maxOutput); validation.textContent = st.chatValidation; }
    } })),
    h('button', { style: { alignSelf: 'flex-end' }, disabled:st.chatBusy, onclick: () => { if(!st.chatBusy){
      st.chat = []; st.baseChat = []; st.chatNotice = st.chatError = st.chatValidation = ''; st.chatEvent = st.chatPartial = null; st.chatGenerations = {}; again(); } } }, 'clear')));
  el.append(steerPanel(app));
  // The response ceiling of each model's last reply, fitted to its own conversation: one line while the two agree.
  const gens = st.chatGenerations || {}, cut = ['base', 'current'].filter((v) => gens[v] && gens[v].effective_output < gens[v].requested_output);
  const tokensText = (n) => `${n} token${n === 1 ? '' : 's'}`;
  if (cut.length === 2 && gens.base.effective_output === gens.current.effective_output) el.append(h('p',{class:'hint'},`Last response ceiling: ${tokensText(gens.current.effective_output)} after fitting the full history in context.`));
  else for (const v of cut) el.append(h('p',{class:'hint'},`Last ${NAME[v]} response ceiling: ${tokensText(gens[v].effective_output)} after fitting the ${NAME[v]} model's full history in context.`));
  if (st.chatEvent) el.append(runtimeEvent(st.chatEvent));
  if (st.chatPartial) {
    const p = st.chatPartial, ran = (key) => !p.variants || key in p.variants;   // only the models the request asked for
    const part = (label, key) => h('p',{},label+': '+(p.variants?.[key] === 'not_run' ? 'not run after the interruption' : p[key]));
    const steered = p.steering ? steerLabel(p.steering.entries) : '';   // what the edited part ran with
    el.append(h('div',{class:'card'},h('b',{},'Interrupted — partial output (not added to conversation)'),
      ran('current') ? [part('Edited','current'),steeredLine(steered)] : null, ran('base') ? part('Base','base') : null));
  }
  const live = st.chatBusy ? st.live?.chat : null;
  const nodes = live ? { base: liveNode(live, 'base'), current: liveNode(live, 'current') } : null;
  if (live) live.paint = () => live.variants.forEach((v) => nodes[v].paint());
  const conv = { current: st.chat, base: st.baseChat }, runs = (v) => !!live && live.variants.includes(v);
  // The messages each conversation has been sent, the pending one included.
  const asked = (v) => conv[v].filter((m) => m.role === 'user').map((m) => m.content).concat(runs(v) ? [live.turn.content] : []);
  const aligned = JSON.stringify(asked('current')) === JSON.stringify(asked('base'));
  const userMsg = (content) => h('div', { class: 'chat-msg user' }, content);
  const reply = (v, m) => m ? h('div', { class: 'chat-msg bot' }, h('div', { class: 'small muted' }, NAME[v]), v === 'current' ? steeredLine(m.steer) : null, m.content) : h('div', {});
  const pending = (v) => h('div', { class: 'chat-msg bot' }, h('div', { class: 'small muted' }, NAME[v]), v === 'current' ? steeredLine(live.steer) : null, nodes[v].body, nodes[v].status);
  const log = h('div', {});
  if (aligned) {
    // The same messages in the same order: each message once, across, with the two replies to it side by side.
    // The pending message is the last one; only a model that is answering it shows a pending reply, so when one
    // column catches up with the other the message and the other model's reply to it are not shown twice.
    const said = asked('current'), replies = { base: conv.base.filter((m) => m.role === 'assistant'), current: conv.current.filter((m) => m.role === 'assistant') };
    const answer = (v, i) => replies[v][i] ? reply(v, replies[v][i]) : runs(v) && i === said.length - 1 ? pending(v) : h('div', {});
    said.forEach((content, i) => log.append(userMsg(content), h('div', { class: 'split' }, answer('base', i), answer('current', i))));
  } else {
    // The conversations differ: each column is one model's conversation, top to bottom.
    const column = (v) => h('div', { 'aria-label': `${NAME[v]} conversation` }, h('div', { class: 'small muted' }, `${NAME[v]} conversation`),
      conv[v].map((m) => m.role === 'user' ? userMsg(m.content) : reply(v, m)), runs(v) ? [userMsg(live.turn.content), pending(v)] : null);
    log.append(h('div', { class: 'split' }, column('base'), column('current')));
  }
  // Refusals, errors and the stop notice appear at the input that was used (in separate mode, under its column).
  const validation = h('p',{role:'alert',style:{color:'var(--crit)'}},st.chatValidation || st.chatError || '');
  const notice = st.chatNotice ? h('p', { class: 'hint', role: 'status' }, st.chatNotice) : null;
  const where = st.chatSplit && (st.chatAt === 'base' || st.chatAt === 'current') ? st.chatAt : 'both';
  const messages = (at) => at === where ? [notice, validation] : null;
  const inputs = {}, runAt = st.chatBusy ? (st.live?.chat?.at || 'both') : null;
  const send = async (at) => {
    const inp = inputs[at];
    if (st.chatBusy || !active() || !inp || !inp.value.trim()) return;
    setDraft(at, inp.value);
    const moved = (st.chatAt || 'both') !== at;
    if (moved) { st.chatError = ''; st.chatNotice = ''; }
    st.chatAt = at;
    const refuse = (message) => { st.chatValidation = message; validation.textContent = message; if (moved) again(); };
    st.chatValidation = outputError(st.cmax ?? output,1,maxOutput); validation.textContent = st.chatValidation;
    if (st.chatValidation) return refuse(st.chatValidation);
    const variants = at === 'both' ? ['current', 'base'] : [at];
    // The base model is never steered: a base send carries no steering and is not held up by the steering panel.
    const steer = variants.includes('current') ? steerRequest(steerState(app)) : { fields: null, label: '', problems: [] };
    if (steer.problems.length) return refuse('Steering: ' + steer.problems.join('; '));
    st.chatBusy = true;
    st.chatError = ''; st.chatEvent = null; st.chatPartial = null; st.chatNotice = '';
    const turn = {role:'user',content:inp.value};
    const sent = (c) => [...c, turn].map((m) => ({ role: m.role, content: m.content }));
    const live = startLive(st, 'chat', { turn, steer: steer.label, variants, at });
    again();
    try {
      const chat_options = thinkingOptions(st.thinking);
      const r = await api('/api/chat', { ...(variants.includes('current') ? { messages: sent(st.chat) } : {}), ...(variants.includes('base') ? { base_messages: sent(st.baseChat) } : {}),
        ...(at === 'both' ? {} : { variants }), template: st.tpl, chat_options, max_new_tokens: Number(st.cmax ?? output), temperature: st.ctemp ?? 0.7, stream: live.id, ...(steer.fields || {}) });
      st.templated = r.chat_template;
      st.chatGenerations = { ...(st.chatGenerations || {}), ...(r.generations || Object.fromEntries(variants.map((v) => [v, r.generation]))) };
      if (r.stopped) {
        // Stopped: the turn is in no conversation, and the message is back in its box to correct.
        setDraft(at, restoredDraft(turn.content, draftOf(at))); st.chatFocus = at;
        const steered = r.steering?.resolved && r.variants?.current && r.variants.current !== 'not_run' ? steerLabel(r.steering.entries) : '';
        st.chatNotice = 'Stopped. The turn was not added; your message is back in the box.' + (steered ? ` (The edited reply was steered: ${steered}.)` : '');
        if (r.event) st.chatEvent = r.event;  // the worker was also interrupted: that stays visible
      }
      else if (r.interrupted) { st.chatEvent=r.event; st.chatPartial=r; }
      else {
        setDraft(at, '');
        const steered = r.steering ? steerLabel(r.steering.entries) : '';   // each turn records the steering it ran with
        if (variants.includes('current')) st.chat.push({ ...turn }, { role: 'assistant', content: r.current, ...(steered ? { steer: steered } : {}) });
        if (variants.includes('base')) st.baseChat.push({ ...turn }, { role: 'assistant', content: r.base });
      }
    } catch (e) { st.chatEvent=e.event; st.chatError = 'Request failed: '+e.message.replace(/[.!?]+$/, '')+'. Your message is preserved for retry.'; }
    finally { st.chatBusy = false; endLive(st, 'chat', live); }
    again();
  };
  const box = (at) => {
    const to = at === 'both' ? 'both models' : `the ${NAME[at]} model`;
    const inp = inputs[at] = h('input', { type: 'text', placeholder: at === 'both' ? 'message' : `message to ${to}`, 'aria-label': `Message to ${to}`, disabled:st.chatBusy,
      value: draftOf(at), oninput: (e) => setDraft(at, e.target.value), style: { flex: 1 } });
    inp.addEventListener('keydown', (e) => { if (e.key === 'Enter') send(at); });
    return h('div', { class: 'row', style: { marginTop: '8px' } }, inp,
      h('button', { class: 'primary', disabled: st.chatBusy, 'aria-label': at === 'both' ? null : `Send to ${to}`, onclick: () => send(at) }, runAt === at ? 'Sending…' : 'Send'),
      runAt === at && st.live?.chat ? stopControl(st.live.chat, again) : null);
  };
  // How a message is sent, on its own line just above the box(es), where the person about to type sees it. It shows
  // the mode in force: a refused switch leaves it as it was. Disabled while a chat request is pending.
  const sendChoice = h('div', { class: 'row send-choice', style: { marginTop: '10px' } }, h('span', {}, 'Send:'),
    tabs(['both', 'separate'], st.chatSplit ? 'separate' : 'both', (v) => { if (!st.chatBusy) switchMode(v === 'separate'); },
      { labels: { both: 'One message to both models', separate: 'A separate message to each model' },
        label: 'Send one message to both models, or a separate message to each', disabled: !!st.chatBusy }));
  // In separate mode each column says which model its box talks to; the placeholder goes once something is typed.
  const toModel = (v) => h('div', { class: 'small muted', style: { marginTop: '8px' } }, `To the ${NAME[v]} model`);
  el.append(log, sendChoice, h('div', {}, !st.chatSplit ? [box('both'), aligned ? null : h('div', { class: 'small muted' }, 'One message still goes to both conversations.')]
    : h('div', { class: 'split' }, h('div', {}, toModel('base'), box('base'), messages('base')), h('div', {}, toModel('current'), box('current'), messages('current'))), messages('both')));
  if (st.chatFocus) { const at = st.chatFocus === true ? 'both' : st.chatFocus; st.chatFocus = false; const target = inputs[at] || Object.values(inputs)[0]; setTimeout(() => target?.focus?.(), 0); }
}

// ------------------------------------------------------------------------------------------------ trace
export async function traceView(el, app) {
  const st = app.keep.trace || (app.keep.trace = {});
  if (st.prompt == null) st.prompt = defaultPrompt(app);
  st.mode = st.mode || 'forward pass';
  const again = () => app.rerender();
  el.append(h('div', { class: 'row' }, h('h2', {}, 'Trace'), tabs(['forward pass', 'activation patching'], st.mode, (m) => { st.mode = m; again(); })));
  if (cannotRun(el, app)) return;
  if (st.mode === 'activation patching') return patchView(el, app, st);
  const go = async () => {
    st.result = { busy: true }; again();
    try { st.result = await api('/api/trace', { prompt: st.prompt, target: st.target || '', versus: st.versus || '', variant: app.variant === 'base' ? 'base' : 'current' }); st.key = key(); app.keep.lastTrace = st.result; }
    catch (e) { st.result = { error: e.message, event:e.event }; }
    again();
  };
  const key = () => app.state.status.version + '|' + (app.variant === 'base');
  const p = h('input', { type: 'text', value: st.prompt, style: { flex: 1, fontFamily: 'var(--mono)' }, oninput: (e) => { st.prompt = e.target.value; } });
  p.addEventListener('keydown', (e) => { if (e.key === 'Enter') go(); });
  el.append(h('div', { class: 'row', style: { margin: '8px 0', alignItems: 'flex-end' } }, field('prompt', p),
    field('target token', h('input', { type: 'text', value: st.target || '', placeholder: 'top prediction', style: { width: '120px' }, onchange: (e) => { st.target = e.target.value; } })),
    field('versus', h('input', { type: 'text', value: st.versus || '', placeholder: '(none)', style: { width: '120px' }, onchange: (e) => { st.versus = e.target.value; } })),
    h('button', { class: 'primary', onclick: go }, 'Trace')));
  el.querySelector('.field').style.flex = '1';
  const r = st.result;
  if (!r) {
    if (app.state.open.kind === 'gguf') el.append(h('p',{class:'hint'},'Choose a prompt and click Trace; opening this tab does not run the model.'));
    else go();
    return;
  }
  if (r.busy) { el.append(h('p', { class: 'muted' }, 'running…')); return; }
  if (r.error) { el.append(h('p', { style: { color: 'var(--crit)' } }, r.error)); return; }
  if (st.key !== key()) {
    if (app.state.open.kind === 'gguf') el.append(h('p',{class:'hint'},'Weights changed; click Trace to evaluate again.'));
    else go();
    return;
  }
  const T = r.tokens.length;
  el.append(h('p', { class: 'muted small' }, `${app.variant === 'base' ? 'base' : 'edited'} model · target ${JSON.stringify(r.target)}` + (r.versus ? ` versus ${JSON.stringify(r.versus)}` : '') + ` · logit${r.versus ? ' difference' : ''} ${fmt(r.dla_total)}`));

  // logit lens
  el.append(h('h3', {}, 'logit lens — the residual stream decoded after ' + (r.lens.length < r.patterns.length + 1 ? 'selected layers' : 'each layer') + ' (shade = probability of that token)'));
  const tbl = h('table', { class: 'lens' });
  tbl.append(h('tr', {}, h('td', { class: 'muted' }, ''), r.tokens.map((t, i) => h('td', { class: 'muted', title: `position ${i}` }, visible(t)))));
  [...r.lens].reverse().forEach((row) => tbl.append(h('tr', {}, h('td', { class: 'muted' }, row.layer),
    row.top.map((t, i) => h('td', { style: { background: seq(row.top_p[i]), color: row.top_p[i] > 0.55 ? '#fff' : '' }, title: `p = ${row.top_p[i]}\np(target) = ${row.target_p[i]}` }, visible(t))))));
  el.append(h('div', { class: 'scroll', style: { overflowX: 'auto' } }, tbl));

  // attribution + neurons
  const dla = [...r.dla].sort((a, b) => Math.abs(b.value) - Math.abs(a.value)).slice(0, 18);
  const addrOf = (d) => (d.kind === 'head' ? `blocks.${d.layer}.attn.head.${d.index}` : d.kind === 'mlp' ? `blocks.${d.layer}.mlp` : d.kind === 'bias' || d.kind === 'attn' ? `blocks.${d.layer}.attn` : 'embed');
  const left = h('div', { class: 'card' }, h('h3', { style: { marginTop: 0 } }, 'direct logit attribution at the last position'),
    hbars(dla.map((d) => ({ label: d.label, value: d.value, title: addrOf(d), onClick: () => app.select(addrOf(d), 'Weights') })), { width: 330, labelW: 64 }),
    h('p', { class: 'hint' }, 'How much each component\'s own output moves the target logit (final LayerNorm scale frozen). Click a bar to open the component.'));
  const right = h('div', { class: 'card' }, h('h3', { style: { marginTop: 0 } }, 'neurons with the largest direct effect'),
    r.neurons.length ? hbars(r.neurons.slice(0, 16).map((n) => ({ label: n.addr, value: n.effect, title: `${n.addr} · activation ${n.act}`, onClick: () => app.select(`blocks.${n.layer}.mlp.neuron.${n.neuron}`, 'Weights') })), { width: 330, labelW: 72 })
      : h('p', { class: 'muted' }, 'this model has no MLPs'));
  el.append(h('div', { class: 'split', style: { marginTop: '12px' } }, left, right));

  // attention patterns
  if (!r.patterns.some((p) => p)) {
    el.append(h('p', { class: 'hint', style: { marginTop: '10px' } }, 'Attention patterns are omitted for this prompt and model size (layers × heads × positions² is too large to send); shorten the prompt to see them.'));
    return;
  }
  el.append(h('div', { class: 'row', style: { marginTop: '6px' } }, h('h3', {}, 'attention patterns — row = destination, column = source'), legend(1, 'seq')));
  const nH = (r.patterns.find((p) => p) || []).length;
  const sz = Math.max(nH > 16 ? 34 : 64, Math.min(120, T * 9));
  r.patterns.forEach((layer, l) => {
    if (!layer) return;
    const rowEl = h('div', { class: 'row', style: { alignItems: 'flex-start', marginBottom: '6px', gap: '4px' } }, h('span', { class: 'muted small', style: { width: '26px' } }, `L${l}`));
    layer.forEach((pat, hh) => {
      const flat = new Float32Array(T * T);
      pat.forEach((rw, i) => rw.forEach((v, j) => { flat[i * T + j] = v; }));
      const cv = heatmap({ rows: T, cols: T, data: flat, scale: 1, mode: 'seq', width: sz, height: sz,
        onHover: (i, j, v, e) => (i == null ? tip(null) : tip(`L${l}H${hh}<br>${esc(r.tokens[i])} (${i}) attends to ${esc(r.tokens[j])} (${j})<br><b>${fmt(v)}</b>`, e)),
        onClick: () => app.select(`blocks.${l}.attn.head.${hh}`, 'Weights') });
      rowEl.append(h('div', { title: `L${l}H${hh}` }, cv, sz >= 46 ? h('div', { class: 'small muted', style: { textAlign: 'center' } }, `${l}.${hh}`) : null));
    });
    el.append(rowEl);
  });
}

async function patchView(el, app, st) {
  const again = () => app.rerender();
  const name = app.state.open.name;
  if (st.clean == null) { st.clean = st.prompt; st.corrupt = name.startsWith('facts') ? '<bos> Bob lives in' : st.prompt; }
  const go = async () => {
    st.patch = { busy: true }; again();
    try { st.patch = await api('/api/patching', { clean: st.clean, corrupt: st.corrupt, target: st.ptarget || '', versus: st.pversus || '', variant: app.variant === 'base' ? 'base' : 'current' }); }
    catch (e) { st.patch = { error: e.message }; }
    again();
  };
  const mk = (k, w) => h('input', { type: 'text', value: st[k] || '', style: { width: w, fontFamily: 'var(--mono)' }, oninput: (e) => { st[k] = e.target.value; } });
  el.append(h('p', { class: 'hint' }, 'Run the corrupted prompt, copy one activation over from the clean run, and see how much of the clean answer comes back. 1 = fully restored. This is how you find where a behaviour lives before editing it.'),
    h('div', { class: 'row', style: { margin: '8px 0', alignItems: 'flex-end' } }, field('clean prompt', mk('clean', '300px')), field('corrupted prompt (same length)', mk('corrupt', '300px')),
      field('target', mk('ptarget', '90px')), field('versus', mk('pversus', '90px')), h('button', { class: 'primary', onclick: go }, 'Patch')));
  const r = st.patch;
  if (!r) return;
  if (r.busy) { el.append(h('p', { class: 'muted' }, 'running…')); return; }
  if (r.error) { el.append(h('p', { style: { color: 'var(--crit)' } }, r.error)); return; }
  el.append(h('p', { class: 'muted small' }, `metric: logit(${JSON.stringify(r.target)}) − logit(${JSON.stringify(r.versus)}) · clean ${fmt(r.clean)} · corrupted ${fmt(r.corrupt)}`), legend(1));
  const T = r.tokens_clean.length, L = r.by_position[0].grid.length;
  if (!r.per_position) el.append(h('p', { class: 'hint' }, 'This model is large, so each activation is patched at all positions at once (one forward pass per layer) and per-head patching is skipped.'));
  const grid = h('div', { class: 'row', style: { alignItems: 'flex-start', gap: '18px', marginTop: '8px' } });
  for (const k of r.by_position) {
    const W = k.grid[0].length;
    const flat = new Float32Array(L * W);
    k.grid.forEach((row, l) => row.forEach((v, t) => { flat[(L - 1 - l) * W + t] = v == null ? NaN : v; }));
    grid.append(h('div', {}, h('div', { class: 'small', style: { marginBottom: '3px' } }, h('b', {}, k.kind), h('span', { class: 'muted' }, W > 1 ? '  layer (top = last) × position' : '  layer (top = last)')),
      heatmap({ rows: L, cols: W, data: flat, scale: 1, width: Math.max(W > 1 ? 120 : 46, W * 34), height: Math.max(40, L * (L > 16 ? 12 : 22)),
        onHover: (i, j, v, e) => (i == null ? tip(null) : tip(`${k.kind} · layer ${L - 1 - i}` + (W > 1 ? ` · position ${j} ${esc(r.tokens_clean[j])}` : '') + `<br><b>${fmt(v)}</b> of the clean answer restored`, e)),
        onClick: (i) => app.select(`blocks.${L - 1 - i}`, 'Weights') }),
      W > 1 ? h('div', { class: 'small muted mono' }, r.tokens_clean.join(' ')) : null));
  }
  if (r.heads) {
    const H = r.heads[0].length, flat = new Float32Array(L * H);
    r.heads.forEach((row, l) => row.forEach((v, hh) => { flat[(L - 1 - l) * H + hh] = v; }));
    grid.append(h('div', {}, h('div', { class: 'small', style: { marginBottom: '3px' } }, h('b', {}, 'heads'), h('span', { class: 'muted' }, '  layer × head (all positions)')),
      heatmap({ rows: L, cols: H, data: flat, scale: 1, width: Math.max(120, H * 26), height: Math.max(40, L * (L > 16 ? 12 : 22)),
        onHover: (i, j, v, e) => (i == null ? tip(null) : tip(`L${L - 1 - i}H${j}<br><b>${fmt(v)}</b> restored`, e)),
        onClick: (i, j) => app.select(`blocks.${L - 1 - i}.attn.head.${j}`, 'Weights') })));
  }
  el.append(grid);
}
