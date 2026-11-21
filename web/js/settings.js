import { h, api } from './util.js';

const GiB = 2**30;
const gib = (bytes) => bytes == null ? 'unknown' : `${(bytes/GiB).toFixed(2)} GiB`;
const FIELDS = [
  ['context', 'Context tokens', 'Prompt, all chat history, template tokens and response share this limit.', 1, Number.MAX_SAFE_INTEGER, 1],
  ['new_tokens', 'Default output tokens', 'Shared by Run and Chat. Each request may ask for more or less, up to remaining context.', 1, Number.MAX_SAFE_INTEGER, 1],
  ['resident_seconds', 'Model idle timeout (seconds)', 'Measured from the end of the last operation. Expiry unloads the model; it never interrupts a running operation.', 10, 86400, 60],
  ['threads', 'CPU threads', 'Packed CPU kernel threads, capped by detected performance cores (physical-core fallback) and the pinned bridge maximum of 32. The separate legacy acceptance path stays capped at 8.', 1, Number.MAX_SAFE_INTEGER, 1],
  ['floor_gib', 'Runtime headroom floor (GiB)', 'Stops the operation when available memory falls below this reserve; persistent low headroom unloads the model. Auto: 10% of physical RAM, at least 0.5 GiB.', 0, 100000, 0.5],
  ['preferred_gib', 'Admission headroom reserve (GiB)', 'Fresh admission check: cold-load forecast, or new operation buffers when loaded, must leave this reserve. Auto: 15% of RAM, at least 0.75 GiB.', 0, 100000, 0.5],
  ['budget_gib', 'Memory budget (GiB)', 'Explicit budgets bound the full forecast and live memory. Auto uses fresh availability at admission; the operation memory limit stays fixed only while that operation runs.', 0, 100000, 0.5],
  ['ceiling_gib', 'Memory ceiling (GiB)', 'Hard total cap alongside an explicit budget and physical RAM minus the admission reserve. Auto: physical RAM minus the admission reserve.', 0, 100000, 0.5],
  ['patch_limit_gib', 'Largest weight patch (GiB)', 'The largest patch one edit may write, as FP32 edits on disk (4 bytes per weight: 1 GiB is about 268 million weights). Applies as soon as it is saved, no restart. Auto: what memory and disk allow now. It never overrides the memory check. A selection edit above 60 million weights asks before it runs. At least 1 MiB (0.0009765625 GiB).', 0.0009765625, 100000, 'any'],
  ['reservation_gib', 'Minimum reservation (GiB)', 'Raises the forecast reservation. It does not allocate or reserve physical RAM. Auto: the model forecast.', 0, 100000, 0.5],
  ['swap_growth_mib', 'Swap growth warning (MiB)', 'Warn above this growth from the watchdog baseline while pressure is normal. Default: 64 MiB. Abnormal pressure stops operations independently.', 0, 100000, 1],
  ['compression_growth_mib', 'Compression growth warning (MiB)', 'Warn above this growth from the watchdog baseline while pressure is normal. Default: 1 GiB. Abnormal pressure stops operations independently.', 0, 100000, 1],
];

export function memoryImpact(el, p, label = 'Memory impact') {
  const machine = p.machine;
  el.replaceChildren();
  el.append(h('h3', {}, label), h('p', { class: 'hint' }, p.qualification));
  if (!machine.total || machine.available == null) {
    el.append(h('p', { class: 'muted' }, 'Current memory is unknown: ' + (machine.error || 'telemetry unavailable')));
  } else {
    const total = machine.total, used = total-machine.available;
    const additional = p.incremental;
    const segments = [
      ['In use / unavailable now', used, 'var(--muted)'],
      [additional == null && !p.resident ? 'Model reservation' : 'Estimated additional reservation', additional == null ? null : Math.min(machine.available, additional), 'var(--s2)'],
      [additional == null ? 'Measured available' : 'Estimated remaining headroom', Math.max(0, machine.available-(additional ?? 0)), 'var(--s1)'],
    ];
    const bar = h('div', { role: 'img', 'aria-label': segments.map(([n,v]) => `${n}: ${gib(v)}`).join('; '),
      style: { display: 'flex', height: '32px', borderRadius: '5px', overflow: 'hidden', margin: '10px 0' } });
    for (const [name, value, color] of segments) bar.append(h('div', { title: `${name}: ${gib(value)}`, style: { width: `${100*value/total}%`, background: color } }));
    el.append(bar, h('div', { class: 'row small' }, segments.map(([name,value,color]) => h('span', {},
      h('span', { style: { display: 'inline-block', width: '10px', height: '10px', background: color, marginRight: '5px' } }), `${name}: ${gib(value)}`))),
      h('p', { class: 'hint' }, `Physical RAM ${gib(total)} · available/reclaimable ${gib(machine.available)} · raw free ${gib(machine.free)}. “In use / unavailable” is total minus measured available, including the current resident model.`));
    if (additional == null) el.append(h('p', { class: 'muted' }, p.resident ? 'Additional model memory is unknown because the active reservation estimate is unavailable. The bar shows measured availability; the proposed total below is a separate estimate.' : 'Additional model memory is unknown. The bar currently shows measured availability only. Open a GGUF workspace to estimate it.'));
    else if (additional > machine.available) el.append(h('p', { style: { color: 'var(--crit)' } }, `Estimated deficit: ${gib(additional-machine.available)}`));
  }
  if (p.required == null && p.minimum_reservation) el.append(h('p', {}, `Minimum reservation alone: ${gib(p.minimum_reservation)}. The full model requirement is unknown.`));
  if (p.required != null) {
    el.append(h('p', {}, `Proposed model reservation: ${gib(p.required)}${p.resident ? ` · active reservation estimate: ${gib(p.active_reservation)} · additional above active: ${gib(p.incremental)}` : ''}.`));
    if (p.resident) el.append(h('details',{},h('summary',{},'Hypothetical second reservation without unloading'),
      h('p',{class:'hint'},`Current available minus the full proposed reservation: ${gib(p.cold_remaining)}. This assumes no memory is recovered by unloading; it is not a prediction of headroom after restart. Actual startup headroom must be measured after unload.`)));
    else el.append(h('p',{class:'hint'},`Cold-load headroom at current availability: ${gib(p.cold_remaining)}. Fresh admission still applies.`));
  }
  if (p.policy) {
    const cap = gib(Math.min(p.policy.budget,p.policy.ceiling));
    if (p.policy_hypothetical) {
      el.append(h('p',{class:'hint'},'Automatic budget after unload/restart is unknown until memory is measured again.'),
        h('details',{},h('summary',{},'Hypothetical admission using current availability'),h('p',{class:'hint'},`Auto cap at current availability: ${cap}; admission reserve ${gib(p.policy.preferred)}. This includes the already resident model in current usage and does not predict the next startup cap.`)));
      if (p.explicit_cap != null) el.append(h('p',{},`Explicit configured cap: ${gib(p.explicit_cap)}.`));
    } else el.append(h('p', {}, `Configured next-admission cap ${cap} · admission reserve ${gib(p.policy.preferred)} · runtime floor ${gib(p.policy.floor)}.`));
  }
  for (const message of p.info || []) el.append(h('p',{class:'hint'},message));
  if (p.active_policy) {
    const cap = p.active_operation ? `Current operation limit ${gib(Math.min(p.active_policy.budget,p.active_policy.ceiling))} (fixed until it finishes)` : `Idle static limit ${gib(p.active_static_cap)}`;
    el.append(h('p', {}, `${cap} · active admission reserve ${gib(p.active_policy.preferred)} · active runtime floor ${gib(p.active_policy.floor)}. Each new operation checks fresh memory.${p.active_operation ? ' Saved memory changes apply when this operation finishes.' : ''}`));
  }
  if (p.forecast) {
    const table = h('table', { class: 'grid' });
    for (const [name, bytes] of Object.entries(p.forecast)) table.append(h('tr', {}, h('td', {}, name.replaceAll('_',' ')), h('td', { class: 'num' }, gib(bytes))));
    el.append(h('details', {}, h('summary', {}, 'Forecast components (estimates)'), table));
  }
  for (const warning of p.warnings || []) el.append(h('p', { class: 'pill warn', style: { display: 'block', whiteSpace: 'normal', overflowWrap: 'anywhere' } }, warning));
  for (const error of p.errors || []) el.append(h('p', { style: { color: 'var(--crit)' } }, error));
}

async function settingsRequest(path, body, csrf) {
  const r = await fetch(path, { method:'POST', headers:{'Content-Type':'application/json','X-WSP-Settings':csrf}, body:JSON.stringify(body) });
  const data = await r.json();
  if (!r.ok) {
    // A save that would unload the loaded model answers with its effect and a token for this exact candidate.
    const detail = data.detail;
    const error = new Error(data.error || (detail && typeof detail === 'object' ? detail.message : detail) || r.statusText);
    if (detail && typeof detail === 'object' && detail.confirm) { error.confirm = detail.confirm; error.effect = detail.effect; }
    throw error;
  }
  return data;
}

const LABELS = Object.fromEntries([['backend', 'Backend'], ...FIELDS.map(([key, label]) => [key, label])]);
const names = (keys) => keys.map((k) => LABELS[k] || k).join(', ');
const WHEN = { saved: 'When saved', next_load: 'Next model load', next_start: 'Next server start', override: 'Launch override wins' };
const loadedText = (load) => `context ${load.context}, ${load.threads} threads, ${load.selected} backend`;

// What happened to each saved change (the server's `applied`), one sentence per kind. Every sentence matches
// wsp/settings.py SettingsStore.save: applied now, at the next model load, at the next start, or held by a launch override.
export function savedReport(applied, values) {
  const lines = [];
  if (!applied) return ['Saved.'];
  if (applied.now.length) {
    lines.push(`Applied now: ${names(applied.now)}.`);
    if (applied.now.includes('patch_limit_gib')) lines.push('Largest weight patch applies now: ' + String(values.patch_limit_gib ?? 'Auto') + (values.patch_limit_gib == null ? '' : ' GiB') + '.');
    if (applied.effect && ['unload', 'after_operation', 'unknown'].includes(applied.effect.kind)) lines.push(applied.effect.message);
    if (applied.late) lines.push('The memory watchdog answered late; asked again, it runs the new values.');
  }
  if (applied.next_load.length) {
    lines.push(`Applies at the next model load: ${names(applied.next_load)}.`);
    lines.push(applied.loaded ? `A model is loaded now with ${loadedText(applied.loaded)}; it keeps these until it is unloaded. Unload it and load it again to use the new values.`
                              : 'No model is loaded; the next load uses the new values.');
  }
  if (applied.next_start.length) lines.push(`Saved for the next server start (this server has no memory supervisor to apply them to): ${names(applied.next_start)}.`);
  if (applied.overridden.length) lines.push(`Saved, but a launch override (command line or original environment) still wins until the server starts without it: ${names(applied.overridden)}.`);
  if (applied.warning) lines.push(applied.warning);
  if (!lines.length) lines.push('Saved. No setting changed.');
  return lines;
}

export async function settingsView(el, app, active = () => true) {
  const result = await api('/api/settings');
  if (!active()) return;
  let model = result, draft = {}, pending = false, previewPending = false, previewFailed = false, previewSeq = 0, timer, preset = result.interactive_defaults;
  const fields = {}, activeCells = {}, whenCells = {};
  const changes = () => Object.fromEntries(Object.entries(draft).filter(([key,value]) => value !== model.saved.values[key]));
  const msg = h('p', { role: 'status', 'aria-live': 'polite' });
  const memory = h('div', { class: 'card', style: { marginTop: '16px' } });
  const saved = h('p', { class: 'hint' });
  const limits = h('p', { class: 'hint' });
  const recommendation = h('p', {class:'hint'});
  // The supervisor runs settings other than the file's, or it is not known which (wsp/settings.py out_of_step).
  const outOfStep = h('p', { class: 'pill warn', role: 'alert', style: { display: 'none', whiteSpace: 'normal', overflowWrap: 'anywhere' } });
  const fill = () => {
    outOfStep.textContent = model.out_of_step ? `${model.out_of_step} Until an accepted save, Active now may not be what the memory watchdog runs.` : '';
    outOfStep.style.display = model.out_of_step ? 'block' : 'none';
    preset = model.interactive_defaults; previewPending = previewFailed = false;
    draft = Object.fromEntries(model.editable.map((k) => [k, model.saved.values[k]]));
    for (const [key,input] of Object.entries(fields)) input.value = draft[key] ?? '';
    for (const [key,cell] of Object.entries(activeCells)) {
      // Active now: what this server uses; a loaded model keeps the load settings it was loaded with.
      const loaded = model.loaded && key in model.loaded && model.loaded[key] !== model.active.values[key] ? ` (loaded model: ${model.loaded[key]})` : '';
      cell.textContent = String(model.active.values[key] ?? 'Auto') + loaded;
    }
    for (const [key,cell] of Object.entries(whenCells)) cell.textContent = WHEN[model.takes_effect?.[key]] || '';
    const pending = Object.entries(model.pending || {});
    const later = pending.filter(([,when]) => when === 'next_start').map(([k]) => k), nextSave = pending.filter(([,when]) => when === 'next_save').map(([k]) => k);
    saved.textContent = `${model.note}${later.length ? ' Saved, waiting for the next server start: '+names(later)+'.' : ''}${nextSave.length ? ' Changed in the file outside Settings, not in use yet (the next save applies them): '+names(nextSave)+'.' : ''}${model.overridden.length ? ' Saved values masked by launch overrides: '+names(model.overridden)+'.' : ''}`;
    limits.textContent = `Saved effective context ${model.preview.effective_context} · model capacity ${model.preview.model_context ?? 'unknown'}. Native ${model.preview.backend.selected} calls use at most ${model.preview.kernel_token_rows} token rows and bounded scratch; generation prefills ${model.preview.prefill_chunk} tokens per chunk. These call sizes do not cap total context.`;
    memoryImpact(memory, model.preview, 'Saved settings memory estimate');
    presetButton.disabled = !preset;
    recommendation.textContent = model.capacity_recommendation?.note || '';
  };
  const updatePreview = () => {
    clearTimeout(timer); const mine = ++previewSeq;
    previewPending = true; previewFailed = false; presetButton.disabled = true;
    msg.replaceChildren(); msg.textContent = 'Unsaved draft. Updating estimate…';   // a pending confirmation is for the old draft
    limits.textContent = 'Draft limits have changed; the previous estimate is no longer current.';
    memory.replaceChildren(h('p',{role:'status',class:'hint'},'Calculating a new metadata-only estimate for the unsaved draft…'));
    timer = setTimeout(async () => {
      if (!active()) return;
      try {
        const next = await settingsRequest('/api/settings/preview', { revision:model.revision, changes:changes() }, model.csrf);
        if (!active() || mine !== previewSeq) return;
        preset = next.interactive_defaults;
        previewPending = false; presetButton.disabled = pending || !preset;
        memoryImpact(memory, next.preview, 'Unsaved draft memory estimate');
        recommendation.textContent = next.capacity_recommendation?.note || '';
        limits.textContent = `Unsaved draft effective context ${next.preview.effective_context} · model capacity ${next.preview.model_context ?? 'unknown'}. ${next.preview.prefill_chunk}-token prefill chunks preserve full history; full trace needs a separate reservation.`;
        msg.textContent = 'Estimate updated. Changes are not saved.';
      } catch (e) { if (active() && mine === previewSeq) {
        previewPending = false; previewFailed = true; presetButton.disabled = true;
        msg.textContent = 'Unsaved draft estimate unavailable: '+e.message;
        limits.textContent = 'Draft limits are not validated. Correct the fields or reload saved settings.';
        memory.replaceChildren(h('p',{class:'hint'},'No current draft memory estimate is available.'));
      } }
    }, 250);
  };
  el.append(h('h2', {}, 'Settings'), h('p', {}, 'Each setting takes effect as its Takes effect column says. Saving never restarts the server and never stops a running operation. If new memory limits or a shorter idle timeout would make the memory watchdog unload the loaded model, now or when the running operation finishes, Save says so and asks you to confirm before anything changes; the draft estimate also warns.'),
    h('p', { class: 'mono small', style:{overflowWrap:'anywhere'} }, `${model.config} · editing ${model.target}`),
    !model.supervised ? h('p', { class: 'pill warn', style: { display: 'block', whiteSpace: 'normal' } }, 'This server has no memory supervisor, so memory limits and the idle timeout have nothing to apply to here. They are saved for the next start of a supervised server (regular wsp serve), with context, CPU threads and backend. Default output tokens and the largest weight patch apply when saved.')
      : h('p', { class:'hint' }, model.loaded ? `A model is loaded with ${loadedText(model.loaded)}; it keeps these until it is unloaded.` : 'No model is loaded; the next load uses the saved context, CPU threads and backend.'), outOfStep, saved);
  const table = h('table', { class: 'grid settings-fields', style: { width:'100%' } }, h('tr', {}, h('th', {}, 'Setting'), h('th', {}, 'Saved / draft'), h('th', {}, 'Active now'), h('th', {}, 'Takes effect'), h('th', {}, 'Active source')));
  const backend = h('select', { 'aria-label':'Backend', onchange: (e) => { draft.backend=e.target.value; updatePreview(); } }, ['auto','cpu','metal'].map((v) => h('option', { value:v }, v)));
  fields.backend = backend;
  activeCells.backend = h('td', {}); whenCells.backend = h('td', {class:'small'});
  table.append(h('tr', {}, h('td', {}, 'Backend'), h('td', {}, backend), activeCells.backend, whenCells.backend, h('td', {}, model.active.sources.backend)));
  for (const [key,label,description,min,max,step] of FIELDS) {
    const nullable = key.endsWith('_gib');
    const input = h('input', { type:'number', 'aria-label':label, min,max:key === 'threads' ? model.thread_limit : max,step, placeholder: nullable ? 'Auto' : '', style:{width:'120px'},
      oninput: (e) => { draft[key] = nullable && e.target.value === '' ? null : Number(e.target.value); updatePreview(); } });
    fields[key] = input;
    activeCells[key] = h('td', {}); whenCells[key] = h('td', {class:'small'});   // filled by fill(), which runs once the table is built
    table.append(h('tr', {}, h('td', {}, label, h('div', {class:'hint'}, description)), h('td', {}, input),
      activeCells[key], whenCells[key], h('td', {class:'small muted'}, model.active.sources[key])));
  }
  const lock = (on) => { pending=on; save.disabled=on; reloadButton.disabled=on; for(const input of Object.values(fields)) input.disabled=on;
    presetButton.disabled = on || previewPending || previewFailed || !preset; };
  // Sends one candidate (the changes as they were when Save was clicked); confirm: the token the server gave for it.
  const submit = async (sent, confirm) => {
    if (pending || !active()) return;
    lock(true); ++previewSeq; clearTimeout(timer); msg.replaceChildren(); msg.textContent = 'Saving…';
    try {
      const next = await settingsRequest('/api/settings', {revision:model.revision,changes:sent,...(confirm ? {confirm} : {})},model.csrf);
      if (!active()) return;
      model=next; fill();
      const lines = savedReport(next.applied, model.active.values);
      msg.replaceChildren(); msg.textContent = lines.join(' ');
      if (next.applied?.next_load?.length && next.applied.loaded) {
        // The existing unload route (the Run tab's Unload control): the next load uses the saved values.
        const unloaded = h('span', {role:'status'});
        const unload = h('button', {onclick: async () => {
          unload.disabled = true; unloaded.textContent = ' Unloading…';
          try { await api('/api/runtime/unload', {}); unloaded.textContent = ' Model unloaded. The next load uses the saved settings.'; }
          catch (e) { unloaded.textContent = ' Unload failed: ' + e.message; unload.disabled = false; }
        }}, 'Unload model');
        msg.append(' ', unload, unloaded);
      }
    } catch(e) {
      if (!active()) return;
      msg.replaceChildren(); msg.textContent = e.message;
      if (e.confirm) {
        // Nothing was saved. The confirmation is for exactly this candidate; editing the draft discards it.
        const confirmButton = h('button', {class:'primary', onclick: () => submit(sent, e.confirm)}, 'Save and unload the model');
        const cancel = h('button', {onclick: () => { msg.replaceChildren(); msg.textContent = 'Not saved. The draft is unchanged.'; }}, 'Cancel');
        msg.append(' ', confirmButton, ' ', cancel);
      }
    }
    finally { lock(false); }
  };
  const save = h('button', {class:'primary', onclick: () => submit(changes())}, 'Save settings');
  const presetButton = h('button',{onclick:()=>{ if(pending || previewPending || previewFailed || !preset)return; for(const [key,value] of Object.entries(preset)){draft[key]=value;fields[key].value=value ?? '';} updatePreview(); }},'Use capacity-aware defaults');
  const reloadButton = h('button',{onclick:()=>{ if(pending)return; ++previewSeq;clearTimeout(timer);app.rerender(); }},'Reload saved settings');
  el.append(h('div',{class:'row settings-actions'},save,presetButton,reloadButton),
    h('p',{class:'hint'},'Capacity-aware defaults recommend context/output for this model, set a four-hour idle timeout, and set budget, ceiling and minimum reservation to Auto. A resident model requires fresh headroom measurement after unload. Runtime floor and admission reserve are preserved. Review the draft, then save it. Existing profile values are never changed automatically.'),msg,recommendation,limits,memory,table);
  fill();
}
