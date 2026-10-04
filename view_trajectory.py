#!/usr/bin/env python3
"""
view_trajectory.py — render a Fara-7B run as an HTML timeline.

Fara-7B saves screenshots (screenshot0.png, screenshot1.png, ...) and prints
Thought/Action/Observation lines to the console. run_task.sh tees that output to
<run>/console.log. This script pairs each step's thought + action + observation
with its screenshot. If console.log is missing/empty, it falls back to a
screenshots-only gallery.

Usage:
    python view_trajectory.py runs/<folder>
"""
import base64
import html
import re
import sys
from pathlib import Path


def parse_console(log: str):
    steps: dict[int, dict] = {}
    marker = re.compile(r"(Thought #\d+:|Action #\d+:|Observation ?#\d+:)")
    parts = marker.split(log)
    for i in range(1, len(parts) - 1, 2):
        head = parts[i].strip()
        body = parts[i + 1].strip()
        n = int(re.search(r"#(\d+)", head).group(1))
        s = steps.setdefault(n, {"thought": "", "action": "", "obs": ""})
        if head.startswith("Thought"):
            s["thought"] = body
        elif head.startswith("Action"):
            s["action"] = body
        else:
            s["obs"] = body
    final_answer = ""
    m = re.search(r"Final Answer:\s*(.*)", log)
    if m:
        final_answer = m.group(1).strip().splitlines()[0]
    task = ""
    mt = re.search(r"Task:\s*(.*)", log)
    if mt:
        task = mt.group(1).strip().splitlines()[0]
    return steps, final_answer, task


def img_tag(base: Path, idx: int) -> str:
    cand = base / f"screenshot{idx}.png"
    if cand.exists():
        data = base64.b64encode(cand.read_bytes()).decode()
        return f'<img src="data:image/png;base64,{data}" loading="lazy"/>'
    return '<div class="noimg">(no screenshot)</div>'


def shot_count(base: Path) -> int:
    nums = [int(m.group(1)) for p in base.glob("screenshot*.png")
            if (m := re.search(r"screenshot(\d+)\.png", p.name))]
    return (max(nums) + 1) if nums else 0


def render_action(action: str) -> str:
    am = re.search(r"executing tool '([^']+)' with arguments (\{.*\})", action, re.S)
    if am:
        return (f'<span class="action">{html.escape(am.group(1))}</span>'
                f'<span class="args">{html.escape(am.group(2))}</span>')
    return f'<span class="action">{html.escape(action)}</span>'


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit("Usage: python view_trajectory.py <run_folder>")
    base = Path(sys.argv[1])
    log_path = base / "console.log"
    log = log_path.read_text() if log_path.exists() else ""

    steps, final_answer, task = parse_console(log) if log.strip() else ({}, "", "")

    rows = []
    if steps:
        for n in sorted(steps):
            s = steps[n]
            shot = img_tag(base, n - 1)  # screenshot the model saw at step n
            rows.append(f"""
            <div class="card">
              <div class="shot">{shot}</div>
              <div class="meta">
                <div class="step">Step {n} {render_action(s['action'])}</div>
                {f'<div class="think"><b>thinking</b><pre>{html.escape(s["thought"])}</pre></div>' if s['thought'] else ''}
                {f'<div class="obs"><b>observation</b> {html.escape(s["obs"])}</div>' if s['obs'] else ''}
              </div>
            </div>""")
        banner = ""
    else:
        # screenshots-only fallback
        for i in range(shot_count(base)):
            rows.append(f"""
            <div class="card">
              <div class="shot">{img_tag(base, i)}</div>
              <div class="meta"><div class="step">Screenshot {i}</div></div>
            </div>""")
        banner = ('<div class="answer">No console.log captured for this run — showing '
                  'screenshots only. Re-run via <code>./run_task.sh</code> to capture thoughts.</div>')

    doc = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Fara-7B trajectory</title>
<style>
  body {{ font-family:-apple-system,system-ui,sans-serif; margin:0; background:#0f1117; color:#e6e6e6; }}
  header {{ padding:16px 24px; background:#161a23; position:sticky; top:0; border-bottom:1px solid #262b36; z-index:5; }}
  header h1 {{ font-size:15px; margin:0 0 6px; color:#9ad; }}
  header .task {{ font-size:16px; }}
  header .answer {{ margin-top:8px; padding:8px 10px; background:#12241a; border-left:3px solid #3fae6b; border-radius:4px; }}
  code {{ background:#0c0f16; padding:1px 5px; border-radius:4px; }}
  .card {{ display:flex; gap:16px; padding:16px 24px; border-bottom:1px solid #1c212b; }}
  .shot img {{ width:560px; max-width:48vw; border-radius:6px; border:1px solid #2a2f3a; }}
  .noimg {{ width:560px; max-width:48vw; height:180px; display:flex; align-items:center; justify-content:center; background:#161a23; color:#667; border-radius:6px; }}
  .meta {{ flex:1; }}
  .step {{ font-weight:600; margin-bottom:6px; }}
  .action {{ color:#7ad; font-family:ui-monospace,monospace; margin-left:6px; }}
  .args {{ color:#c99; font-family:ui-monospace,monospace; font-size:12px; margin-left:6px; word-break:break-all; }}
  .think {{ margin-top:8px; background:#12151d; border-radius:6px; padding:8px 10px; }}
  .think pre {{ white-space:pre-wrap; margin:4px 0 0; font-size:13px; color:#b9c2d0; }}
  .obs {{ margin-top:8px; font-size:13px; color:#9aa7ba; }}
</style></head><body>
<header>
  <h1>Fara-7B — Computer Use Trajectory</h1>
  <div class="task">{html.escape(task) or '(task)'}</div>
  {f'<div class="answer"><b>answer:</b> {html.escape(final_answer)}</div>' if final_answer else banner}
</header>
{''.join(rows) if rows else '<div class="card">Nothing to show.</div>'}
</body></html>"""

    out = base / "trajectory.html"
    out.write_text(doc)
    print(f"Wrote {out}")
    print(f"Open it:  open '{out}'")


if __name__ == "__main__":
    main()
