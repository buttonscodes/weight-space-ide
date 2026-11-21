// The shell's (app.js) selection, tab bar and view mounting, kept here so they can be tested in a fake DOM: app.js
// itself starts the page when it is imported. tests/shell_frontend.test.mjs checks these and their wiring in app.js.
import { h, clear } from './util.js';

// Views whose drawing does not read the tree selection (app.sel): a click in the tree updates the inspector and
// leaves them as they are, unrebuilt and at the same scroll position. Checked by reading each view: Weights and
// Systems draw the selected node (and `calculation` in inspection.js is bound to it), so they stay out. Atlas
// does not read it today but is being changed on another branch, so it stays out too. Directions works on saved
// measurements and the whole model, never on the tree selection.
export const SELECTION_FREE = new Set(['Programmer', 'Run', 'Probe', 'Directions', 'Trace', 'Tests', 'Chunks', 'Export', 'Settings']);

// The key under which renderView keeps the view's scroll position across re-renders of the same view.
export const viewKey = (tab, sel) => (SELECTION_FREE.has(tab) ? tab : tab + sel);

// Whether a selection change redraws the view. `shown` is the tab whose view is in the page now: a caller that set
// app.tab first and then calls app.select (the Tests view's trace button does) still gets the new view drawn.
export const renderOnSelect = (tab, shown) => !SELECTION_FREE.has(tab) || shown !== tab;

// A tab button's text: its name, and for the Programmer tab what its conversation is doing (programmerMarker).
export const tabText = (name, marker) => (marker ? `${name} · ${marker.text}` : name);

// app.select: a node chosen in the tree (or the inspector). The inspector always follows it; the view is redrawn
// only when it draws the selection (renderOnSelect). `ui` is the shell's: markSelected(addr), renderTabs(),
// renderView(), renderInspector() and shownView() (the tab whose view is in the page).
export function selectNode(app, addr, tab, ui) {
  app.sel = addr;
  if (tab) app.tab = tab;
  ui.markSelected(addr);
  ui.renderTabs();
  if (renderOnSelect(app.tab, ui.shownView())) ui.renderView();
  ui.renderInspector();
}

// The tab bar. The Programmer button says, from every tab, whether its conversation is working, waiting for the
// person's answer, or has finished its turn (text, so a screen reader reads it with the name; `marker()` is
// programmerMarker of the tab's state). update() redraws the bar only when that changes, and never the view.
// Clicking the shown Programmer tab again scrolls to its top, where the open run is.
export function tabBar(app, names, { host, marker, renderView, scrollToTop }) {
  let shown = null;
  const bar = {
    render() {
      const t = clear(host());
      const m0 = marker();
      shown = m0 ? m0.state : null;
      for (const name of names) {
        const m = name === 'Programmer' ? m0 : null;
        t.append(h('button', { class: (name === app.tab ? 'on' : '') + (m ? ' marked ' + m.state : ''), title: m ? m.title : null,
          onclick: () => {
            if (app.tab === name) { if (name === 'Programmer') scrollToTop(); return; }
            app.tab = name; bar.render(); renderView();
          } }, tabText(name, m)));
      }
    },
    update() {
      const m = marker();
      if ((m ? m.state : null) !== shown) bar.render();
    },
  };
  return bar;
}

// renderView's last step, once the view `box` is built: put it in the page, let it finish (afterMount: the
// Programmer tab moves its log in and puts the log where the person left it), and only then restore the page's
// scroll position, so that the page has its full height by then. The position is kept across re-renders of the
// same view (viewKey); another view starts at its top.
export function mountView(el, box, tab, sel) {
  const key = viewKey(tab, sel);
  const y = el.dataset.tab === key ? el.scrollTop : 0;
  clear(el).append(box);
  el.dataset.tab = key;
  el.dataset.view = tab;
  if (typeof box.afterMount === 'function') { try { box.afterMount(); } catch (e) { console.error(e); } }
  el.scrollTop = y;
}
