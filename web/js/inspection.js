// Read-only requests are shared within one workspace revision. Failed requests can be retried.
import { api, q, h } from './util.js';

export const inspectionScope = (app) => `${app.state?.status?.workspace_id}|${app.state?.status?.version}`;
export class InspectionClient {
  constructor(send = api, limit = 64) { this.send = send; this.limit = limit; this.entries = new Map(); this.scope = null; }
  key(app, path, params, post = false) {
    const scope = inspectionScope(app);
    if (scope !== this.scope) { this.entries.clear(); this.scope = scope; }
    return JSON.stringify([scope, path, post, Object.entries(params).sort(([a], [b]) => a.localeCompare(b))]);
  }
  peek(app, path, params = {}, post = false) { return this.entries.get(this.key(app, path, params, post)); }
  read(app, path, params = {}, post = false) {
    const key = this.key(app, path, params, post);
    if (this.entries.has(key)) return this.entries.get(key).promise;
    if ([...this.entries.values()].filter((e) => e.state === 'pending').length >= 8) return Promise.reject(new Error('Other requests are still running. Try again when they finish.'));
    const entry = { state: 'pending' };
    const scope = this.scope;
    const args = { ...params, workspace_id: app.state.status.workspace_id, version: app.state.status.version };
    this.entries.set(key, entry);
    entry.promise = Promise.resolve().then(() => post ? this.send(path, args) : this.send(path + '?' + q(args))).then((value) => {
      entry.state = 'complete'; entry.value = value;
      if (scope === this.scope) {
        // Evict completed entries only: a pending request must remain joinable.
        for (const [k, v] of this.entries) {
          if (this.entries.size <= this.limit) break;
          if (v.state === 'complete') this.entries.delete(k);
        }
      }
      return value;
    }).catch((err) => {
      if (this.entries.get(key) === entry) this.entries.delete(key);
      throw err;
    });
    return entry.promise;
  }
}
export const inspection = new InspectionClient();

// Mounting a control never starts work. It only observes an already requested result.
export function calculation(el, app, { path, params = {}, label, description, render, active = () => true, post = false, tabBound = true, cancellable = false, progressEvent = null }) {
  const scope = inspectionScope(app), addr = app.sel, tab = app.tab;
  const valid = () => active() && scope === inspectionScope(app) && addr === app.sel && (!tabBound || tab === app.tab);
  const box = h('div', { class: 'card', style: { margin: '10px 0' } });
  const status = h('span', { class: 'muted small', role: 'status', 'aria-live': 'polite' });
  const result = h('div', {});
  let busy = false, complete = false;
  const cancel = cancellable ? h('button',{disabled:true,onclick:async()=>{
    if (!valid()) return;
    cancel.disabled=true;
    try { await api(path+'/cancel',{...params,workspace_id:app.state.status.workspace_id,version:app.state.status.version}); }
    catch(err) {if(valid())status.textContent=err.message;}
  }},'Cancel') : null;
  const observe = async (promise) => {
    if (busy) return;
    busy = true; button.disabled = true; status.textContent = 'Calculating… This can take time. Leaving this view does not cancel it.';
    if (cancel) cancel.disabled=false;
    const progress = (event) => {
      if (valid() && event.detail.workspace_id===app.state.status.workspace_id && event.detail.version===app.state.status.version)
        status.textContent=`Read ${(event.detail.values/1e6).toFixed(1)}M logical values (512M ceiling)…`;
    };
    if (progressEvent) window.addEventListener(progressEvent,progress);
    try {
      const value = await promise;
      if (!valid()) return;
      render(result, value);
      complete = true; status.textContent = 'Complete'; button.style.display = 'none';
    } catch (err) {
      if (!valid()) return;
      status.textContent = err.message; button.textContent = 'Retry ' + label.toLowerCase();
    } finally { busy = false; button.disabled = complete; if(cancel)cancel.disabled=true; if(progressEvent)window.removeEventListener(progressEvent,progress); }
  };
  const button = h('button', { onclick: () => {
    if (!busy && !complete && valid()) observe(inspection.read(app, path, params, post));
  } }, label);
  box.append(h('div', { class: 'row' }, button, ...(cancel ? [cancel] : []), status));
  if (description) box.append(h('p', { class: 'hint' }, description));
  box.append(result);
  el.append(box);
  const existing = inspection.peek(app, path, params, post);
  if (existing) observe(existing.promise);
  return box;
}
