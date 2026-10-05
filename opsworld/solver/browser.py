"""
browser.py — Set-of-Marks (SoM) Playwright adapter for OpsWorld.

Because we own the DOM, SoM is reliable: we number every interactable element,
overlay numbered boxes for the screenshot the model sees, and execute actions on
the REAL element via its data-som id (no pixel grounding). We also capture a CLEAN
screenshot + the chosen element's bbox-center coords so trajectories are
Fara-training-ready later.
"""
from __future__ import annotations

from playwright.sync_api import sync_playwright

# JS: find visible interactable elements, tag with data-som, draw numbered
# overlays, and return their metadata. Overlays are pointer-events:none so they
# never block real clicks.
_SOM_JS = r"""
() => {
  document.querySelectorAll('.__som').forEach(e => e.remove());
  const sel = 'button, a, input, select, textarea, .acct, tr[onclick], [onclick]';
  const seen = new Set();
  const els = [...document.querySelectorAll(sel)].filter(el => {
    const r = el.getBoundingClientRect();
    const st = getComputedStyle(el);
    if (r.width < 2 || r.height < 2) return false;
    if (st.visibility === 'hidden' || st.display === 'none' || st.opacity === '0') return false;
    if (r.bottom < 0 || r.top > innerHeight || r.right < 0 || r.left > innerWidth) return false;
    return true;
  });
  const out = [];
  els.forEach((el, i) => {
    el.setAttribute('data-som', i);
    const r = el.getBoundingClientRect();
    let text = (el.innerText || el.value || el.getAttribute('placeholder') || '').trim().replace(/\s+/g, ' ').slice(0, 70);
    let role = el.tagName.toLowerCase() + (el.type ? (':' + el.type) : '');
    let options = null;
    if (el.tagName.toLowerCase() === 'select') {
      options = [...el.options].map(o => o.value);
    }
    out.push({id: i, role, text, options,
              x: Math.round(r.left + r.width / 2), y: Math.round(r.top + r.height / 2),
              w: Math.round(r.width), h: Math.round(r.height)});
    const box = document.createElement('div');
    box.className = '__som';
    box.style.cssText = `position:fixed;left:${r.left}px;top:${r.top}px;width:${r.width}px;height:${r.height}px;border:2px solid #e11;z-index:2147483646;pointer-events:none;box-sizing:border-box;`;
    const lab = document.createElement('div');
    lab.className = '__som';
    lab.textContent = i;
    lab.style.cssText = `position:fixed;left:${r.left}px;top:${Math.max(0, r.top - 14)}px;background:#e11;color:#fff;font:bold 11px monospace;padding:0 3px;z-index:2147483647;pointer-events:none;`;
    document.body.appendChild(box); document.body.appendChild(lab);
  });
  return out;
}
"""

_CLEAR_JS = "() => document.querySelectorAll('.__som').forEach(e => e.remove())"


class SoMBrowser:
    def __init__(self, headful: bool = False, viewport=(1440, 900)):
        self._pw = sync_playwright().start()
        self.browser = self._pw.chromium.launch(headless=not headful)
        self.page = self.browser.new_context(
            viewport={"width": viewport[0], "height": viewport[1]}).new_page()

    def goto(self, url: str):
        self.page.goto(url, wait_until="load")
        self.page.wait_for_timeout(300)

    def mark(self) -> list[dict]:
        """Tag + number interactable elements; return their metadata."""
        return self.page.evaluate(_SOM_JS)

    def clear_marks(self):
        self.page.evaluate(_CLEAR_JS)

    def screenshot(self, annotated: bool = True) -> bytes:
        if not annotated:
            self.clear_marks()
        png = self.page.screenshot()
        return png

    def url(self) -> str:
        return self.page.url

    # ---- actions on a marked element ----
    def _loc(self, mark: int):
        return self.page.locator(f'[data-som="{mark}"]').first

    def click(self, mark: int):
        self.clear_marks()
        self._loc(mark).click(timeout=5000)
        self.page.wait_for_timeout(400)

    def type(self, mark: int, text: str, enter: bool = True):
        self.clear_marks()
        el = self._loc(mark)
        el.click(timeout=5000)
        el.fill("")
        el.type(text, delay=10)
        if enter:
            el.press("Enter")
        self.page.wait_for_timeout(400)

    def select(self, mark: int, value: str):
        self.clear_marks()
        self._loc(mark).select_option(value, timeout=5000)
        self.page.wait_for_timeout(300)

    def close(self):
        try:
            self.browser.close()
        finally:
            self._pw.stop()
