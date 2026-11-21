// A fake DOM that behaves like a browser's where a looser fake does not, so that page code which would throw or
// misbehave in a browser fails here too. Born from the first real run of the Programmer tab, which failed
// with "node.children[0].children.at is not a function" while every loose fake-DOM test passed.
//
//  - `children` is an HTMLCollection: element nodes only, indexable and iterable, no array methods. `childNodes` is a
//    NodeList (forEach, no map/filter). Text lives in Text nodes.
//  - Appending a node that already has a parent moves it (it leaves the old parent), as in a browser.
//  - `isConnected` is true only inside `document.body`'s tree.
//  - Scrolling: a detached element has no scroll position (scrollTop reads 0 and ignores writes) and an element
//    loses its position when it is removed from the document (its layout box is gone). scrollTop is clamped to
//    scrollHeight - clientHeight, also when the content shrinks; a change fires a `scroll` event asynchronously
//    (an element detached by then still receives it, as in a browser). Layout is a fixed model: each
//    element is 20 px plus its descendants, and an element with style.maxHeight is at most that tall.
//  - click() on a disabled control does nothing; attributes a browser does not reflect as properties (aria-*,
//    colspan, tabindex) are attributes only.
// Not modelled: CSS, real layout, focus rules beyond document.activeElement, event bubbling.

export class HTMLCollection {
  constructor(nodes) { nodes.forEach((n, i) => { this[i] = n; }); this.length = nodes.length; }
  item(i) { return this[i] ?? null; }
  *[Symbol.iterator]() { for (let i = 0; i < this.length; i++) yield this[i]; }
}
export class NodeList extends HTMLCollection {
  forEach(fn) { for (let i = 0; i < this.length; i++) fn(this[i], i, this); }
}

let DOC_ROOT = null;
const connected = (node) => { let n = node; while (n.parentNode) n = n.parentNode; return n === DOC_ROOT; };
const forgetScroll = (node) => { if (node.nodeType === 1) { node._scrollTop = 0; node._nodes.forEach(forgetScroll); } };
// A scrolled box whose content shrinks is clamped, and fires `scroll` (asynchronously), as in a browser.
const reclamp = (el) => {
  for (let a = el; a && a.nodeType === 1; a = a.parentNode) {
    if (!a._scrollTop || !a.isConnected) continue;
    const max = Math.max(0, a.scrollHeight - a.clientHeight);
    if (a._scrollTop > max) { a._scrollTop = max; setTimeout(() => a.dispatchEvent({ type: 'scroll' }), 0); }
  }
};
const detach = (node) => {
  const p = node.parentNode;
  if (!p) return;
  const was = connected(node);
  p._nodes = p._nodes.filter((x) => x !== node);
  node.parentNode = null;
  if (was) { forgetScroll(node); reclamp(p); }
};

export class Text {
  constructor(text) { this.nodeType = 3; this.data = String(text); this.parentNode = null; }
  get textContent() { return this.data; }
  set textContent(v) { this.data = String(v); }
  get parentElement() { return this.parentNode; }
  get isConnected() { return connected(this); }
  remove() { detach(this); }
}

const REFLECTED = ['id', 'title', 'role', 'type', 'placeholder', 'href', 'name', 'target', 'rel', 'autocomplete'];

export class Element {
  constructor(tag) {
    this.tagName = String(tag).toUpperCase(); this.nodeType = 1; this._nodes = []; this._attrs = {}; this._listeners = {};
    this.style = {}; this.dataset = {}; this.disabled = false; this.value = ''; this.checked = false; this.selected = false;
    this.className = ''; this.parentNode = null; this._scrollTop = 0;
    if (this.tagName === 'DETAILS') this.open = false;
  }
  get tag() { return this.tagName.toLowerCase(); }
  get parentElement() { return this.parentNode instanceof Element ? this.parentNode : null; }
  get children() { return new HTMLCollection(this._nodes.filter((n) => n.nodeType === 1)); }
  get childNodes() { return new NodeList(this._nodes); }
  get firstChild() { return this._nodes[0] ?? null; }
  get lastChild() { return this._nodes[this._nodes.length - 1] ?? null; }
  get isConnected() { return connected(this); }
  get textContent() { return this._nodes.map((n) => n.textContent).join(''); }
  set textContent(v) { for (const n of [...this._nodes]) detach(n); if (String(v) !== '') this.append(new Text(v)); }
  get classList() {
    const el = this, list = () => el.className.split(/\s+/).filter(Boolean);
    return { add: (...c) => { el.className = [...new Set([...list(), ...c])].join(' '); }, remove: (...c) => { el.className = list().filter((x) => !c.includes(x)).join(' '); },
      toggle: (c, on) => { const has = list().includes(c); if (on ?? !has) el.classList.add(c); else el.classList.remove(c); }, contains: (c) => list().includes(c) };
  }
  append(...nodes) {
    for (const n of nodes) {
      const node = typeof n === 'string' ? new Text(n) : n;
      detach(node);                                  // a node lives in one place: appending moves it
      node.parentNode = this; this._nodes.push(node);
    }
  }
  prepend(...nodes) { const rest = this._nodes; this._nodes = []; this.append(...nodes); this._nodes.push(...rest); }
  replaceChildren(...nodes) { for (const n of [...this._nodes]) detach(n); this.append(...nodes); }
  removeChild(n) { if (n.parentNode !== this) throw new Error('NotFoundError: not a child of this node'); detach(n); return n; }
  remove() { detach(this); }
  contains(n) { for (let x = n; x; x = x.parentNode) if (x === this) return true; return false; }
  setAttribute(k, v) { this._attrs[k] = String(v); }
  getAttribute(k) { return k in this._attrs ? this._attrs[k] : null; }
  hasAttribute(k) { return k in this._attrs; }
  removeAttribute(k) { delete this._attrs[k]; }
  addEventListener(type, fn) { (this._listeners[type] ||= []).push(fn); }
  removeEventListener(type, fn) { this._listeners[type] = (this._listeners[type] || []).filter((f) => f !== fn); }
  dispatchEvent(ev) { ev.target ??= this; for (const fn of [...(this._listeners[ev.type] || [])]) fn.call(this, ev); return true; }
  click() { if (this.disabled) return undefined; return this.dispatchEvent({ type: 'click' }); }
  focus() { if (this.isConnected) document.activeElement = this; }
  scrollIntoView(opts) { this.scrolledIntoView = (this.scrolledIntoView || 0) + 1; this.scrollIntoViewOptions = opts; }
  // ------------------------------------------------------------------ the layout model (see the top)
  get scrollHeight() { return this.isConnected ? 20 * (1 + this.querySelectorAll('*').length) : 0; }
  get clientHeight() {
    if (!this.isConnected) return 0;
    const max = parseInt(this.style.maxHeight, 10);
    return Number.isFinite(max) ? Math.min(max, this.scrollHeight) : this.scrollHeight;
  }
  get scrollTop() { return this.isConnected ? this._scrollTop : 0; }
  set scrollTop(v) {
    if (!this.isConnected) return;                  // a detached element has no scroll position
    const next = Math.max(0, Math.min(Number(v) || 0, this.scrollHeight - this.clientHeight));
    if (next === this._scrollTop) return;
    this._scrollTop = next;
    setTimeout(() => this.dispatchEvent({ type: 'scroll' }), 0);
  }
  // ------------------------------------------------------------------ selectors: tag, #id, .class, [attr], [attr="v"], descendant, comma
  querySelectorAll(selector) {
    const all = [];
    const walk = (n) => { for (const c of n._nodes) if (c.nodeType === 1) { all.push(c); walk(c); } };
    walk(this);
    const groups = String(selector).split(',').map((s) => s.trim().split(/\s+/));
    const hit = (el) => groups.some((parts) => {
      if (!matches(el, parts[parts.length - 1])) return false;
      let i = parts.length - 2;
      for (let a = el.parentElement; a && i >= 0; a = a.parentElement) if (matches(a, parts[i])) i--;
      return i < 0;
    });
    return all.filter(hit);
  }
  querySelector(selector) { return this.querySelectorAll(selector)[0] ?? null; }
}
for (const k of REFLECTED) Object.defineProperty(Element.prototype, k, { get() { return this.getAttribute(k) ?? ''; }, set(v) { this.setAttribute(k, v); }, configurable: true });

function matches(el, simple) {
  if (simple === '*') return true;
  const m = /^([a-zA-Z][\w-]*)?((?:[#.][\w-]+|\[[^\]]+\])*)$/.exec(simple);
  if (!m) throw new Error('strict_dom: unsupported selector ' + simple);
  if (m[1] && el.tag !== m[1].toLowerCase()) return false;
  for (const part of m[2].match(/[#.][\w-]+|\[[^\]]+\]/g) || []) {
    if (part[0] === '#' && el.id !== part.slice(1)) return false;
    if (part[0] === '.' && !el.className.split(/\s+/).includes(part.slice(1))) return false;
    if (part[0] === '[') {
      const [, k, v] = /^\[([\w-]+)(?:="?([^"]*)"?)?\]$/.exec(part);
      const have = k in el._attrs ? el._attrs[k] : (k in el && typeof el[k] !== 'function' ? String(el[k]) : null);
      if (have == null || (v !== undefined && have !== v)) return false;
    }
  }
  return true;
}

// Installs the fake as the global document. Returns the document and a `view` element in the body, the place where
// the shell mounts a view (app.js renderView). With `shell`, the body also has the other elements of index.html that
// app.js fills when it is imported (#topbar, #tabs, #tree, ...), and `window` and `CSS` exist, so that app.js itself
// can be imported and driven.
export function installStrictDom({ shell = false } = {}) {
  const html = new Element('html'), body = new Element('body');
  DOC_ROOT = html; html.append(body);
  const view = new Element('div'); view.id = 'view'; view.style.maxHeight = '800px';
  if (shell) {
    const root = new Element('div'); root.id = 'shell'; body.append(root);
    for (const id of ['topbar', 'tabs', 'tree', 'groups', 'inspector', 'patches', 'console', 'level-legend']) {
      const el = new Element('div'); el.id = id; root.append(el);
    }
    root.append(view);
    global.window = { devicePixelRatio: 1, innerWidth: 1400, innerHeight: 1000, addEventListener() {}, removeEventListener() {}, dispatchEvent() { return true; } };
    global.CSS = { escape: (s) => String(s).replace(/["\\]/g, '\\$&') };
  } else body.append(view);
  global.document = {
    documentElement: html, body, activeElement: null,
    createElement: (t) => new Element(t), createElementNS: (ns, t) => new Element(t), createTextNode: (t) => new Text(t),
    getElementById: (id) => html.querySelector('#' + id), querySelector: (s) => html.querySelector(s), querySelectorAll: (s) => html.querySelectorAll(s),
  };
  return { document: global.document, view };
}

// What the shell does with a built view (app.js renderView, same tab): read the view's scroll position, mount the
// box, call its afterMount, then put the scroll position back.
export function mount(view, box) {
  const y = view.scrollTop;
  view.replaceChildren(box);
  if (typeof box.afterMount === 'function') box.afterMount();
  view.scrollTop = y;
  return box;
}

// Every element under a node, in document order (for tests; page code must not rely on this).
export const all = (root) => [root, ...root.querySelectorAll('*')];
export const byText = (root, tag, text) => all(root).find((n) => n.tag === tag && n.textContent === text);
export const flush = async (n = 6) => { for (let k = 0; k < n; k++) await new Promise((r) => setTimeout(r, 0)); };
