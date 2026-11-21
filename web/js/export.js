import { h, api, field } from './util.js';

export async function exportView(el, app) {
  if (app.state.open.capabilities?.format !== 'gguf') {
    el.append(h('p', {}, 'Use the existing Export action for this checkpoint.')); return;
  }
  const st = app.keep.export || (app.keep.export = { precision: 'F16' });
  const again = () => app.rerender();
  el.append(h('h2', {}, 'Export edited GGUF'),
    h('p', {}, 'Companion preserves your FP32 patches and undo history with the exact original base. Standalone bakes edits into a standard GGUF and loses undo; it can change or erase the edit.'),
    h('a', { href: '/api/export' }, 'Download faithful companion'),
    h('p', {}, 'Only tensors affected by enabled, nonzero patches change precision. Even one changed weight can promote its whole tensor. Untouched packed tensor bytes stay identical.'),
    field('Standalone precision', h('select', { onchange: e => { st.precision=e.target.value; st.preview=null; again(); } },
      [['F16','F16 — default, preserve edits with moderate size increase'],['Q8_0','Q8_0 — smaller, may erase subtle edits'],['F32','F32 — largest, avoid additional weight rounding'],['SAME','Same scheme — lossy; may erase the edit']].map(([value,label]) => h('option', {value,selected:st.precision===value},label)))),
    h('button', { onclick: async () => { try { st.preview=await api('/api/gguf/export/preview',{precision:st.precision}); st.error=null; } catch(e) {st.error=e.message;} again(); } }, 'Preview size (no inference)'));
  if (st.preview) {
    const p=st.preview;
    el.append(h('p', {}, `${p.changed_tensor_count} tensor${p.changed_tensor_count===1?'':'s'} · ${p.output_bytes.toLocaleString()} bytes total · ${p.increase_bytes.toLocaleString()} bytes increase`),
      h('pre', {}, p.tensors.map(t=>`${t.gguf_name}: ${t.before} → ${t.after} (${t.after_bytes.toLocaleString()} bytes)`).join('\n')),
      h('button', {class:'primary',disabled:st.job?.status==='RUNNING',onclick:async()=>{try{st.job=await api('/api/gguf/export',{precision:st.precision});st.error=null;}catch(e){st.error=e.message;}again();}}, 'Export standalone (UNVALIDATED)'));
  }
  if (st.job) {
    const j=st.job;
    el.append(h('p', {class:'pill '+(j.validation_status==='VERIFIED'?'good':'warn')}, `${j.status} · behavioral status: ${j.validation_status}`));
    if (j.error) el.append(h('p', {style:{color:'var(--crit)'}},j.error));
    if (j.progress) el.append(h('p', {}, `${j.progress.tensor} · ${j.progress.bytes_written.toLocaleString()} / ${j.progress.total_bytes.toLocaleString()} bytes`));
    el.append(h('button', {onclick:async()=>{try{st.job=await api('/api/gguf/export/'+j.id);st.error=null;}catch(e){st.error=e.message;}again();}},'Refresh export / validation status'));
    if (j.status==='RUNNING') el.append(h('button',{onclick:async()=>{st.job=await api('/api/gguf/export/'+j.id+'/cancel',{});again();}},'Cancel export'));
    if (j.status==='COMPLETE') el.append(h('a',{href:'/api/gguf/export/'+j.id+'/download'},'Download standalone — '+j.validation_status));
    if (j.validation) el.append(h('pre',{},JSON.stringify(j.validation,null,2)));
  }
  if (st.error) el.append(h('p',{style:{color:'var(--crit)'}},st.error));
  el.append(h('p',{class:'hint'},'Serialization success is not behavioral verification. Export never loads a model. An explicitly authorized, guarded validation run must compare edit-gain retention, unrelated top tokens, drift, logits and chat identity; coarse next-token tests alone do not qualify. If validation fails, use the companion. VERIFIED means only the recorded fixed suite passed.'));
}
