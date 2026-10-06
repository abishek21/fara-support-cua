"""
view_trajectory.py — render a collected OpsWorld trajectory as an HTML timeline
(screenshots + thought + action per step) so you can eyeball quality before upload.

Usage:
    python -m solver.view_trajectory runs/pilot/C4_churn/seed4
    python -m solver.view_trajectory --sample 2 --category C6_noerror   # open N random from a category
"""
from __future__ import annotations

import argparse
import base64
import glob
import html
import json
import os
import random
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OPSROOT = os.path.dirname(HERE)


def _img(run_dir, name):
    p = os.path.join(run_dir, name)
    if not os.path.exists(p):
        return '<div class="noimg">(missing)</div>'
    b64 = base64.b64encode(open(p, "rb").read()).decode()
    return f'<img src="data:image/png;base64,{b64}" loading="lazy"/>'


def render(run_dir: str) -> str:
    tj = json.load(open(os.path.join(run_dir, "trajectory.json")))
    steps = tj["steps"]
    rows = []
    for i, s in enumerate(steps):
        act = s["action"]
        a = f'<span class="act">{html.escape(act.get("type",""))}</span>'
        for k in ("text", "value", "url", "answer"):
            if act.get(k):
                a += f' <span class="arg">{k}={html.escape(str(act[k])[:80])}</span>'
        if s.get("coord"):
            a += f' <span class="coord">@{s["coord"]}</span>'
        el = s.get("element") or {}
        eltxt = (f' <span class="el">[{html.escape(el.get("role",""))} '
                 f'"{html.escape(el.get("text",""))}"]</span>') if el else ""
        rows.append(f"""
        <div class="card">
          <div class="shot">{_img(run_dir, s.get("screenshot", f"step{i}.png"))}</div>
          <div class="meta">
            <div class="step">Step {i+1} {a}{eltxt}</div>
            <div class="think">{html.escape(s.get("thought",""))}</div>
            <div class="url">{html.escape(s.get("url",""))}</div>
          </div>
        </div>""")
    name = os.path.basename(run_dir.rstrip("/"))
    out = f"""<!doctype html><html><head><meta charset="utf-8"><title>{name}</title>
<style>
 body{{font-family:-apple-system,system-ui,sans-serif;margin:0;background:#0f1117;color:#e6e6e6}}
 header{{padding:14px 22px;background:#161a23;position:sticky;top:0;border-bottom:1px solid #262b36}}
 header b{{color:#9ad}} .fa{{margin-top:6px;color:#8aa}}
 .card{{display:flex;gap:16px;padding:14px 22px;border-bottom:1px solid #1c212b}}
 .shot img{{width:600px;max-width:52vw;border:1px solid #2a2f3a;border-radius:6px}}
 .noimg{{width:600px;height:160px;background:#161a23;color:#667;display:flex;align-items:center;justify-content:center;border-radius:6px}}
 .meta{{flex:1}} .step{{font-weight:600;margin-bottom:6px}}
 .act{{color:#7ad;font-family:ui-monospace,monospace}} .arg{{color:#c99;font-size:12px}}
 .coord{{color:#9c6;font-size:12px}} .el{{color:#888;font-size:12px}}
 .think{{background:#12151d;padding:8px 10px;border-radius:6px;color:#b9c2d0}}
 .url{{color:#6b84b4;font-size:12px;margin-top:4px;word-break:break-all}}
</style></head><body>
<header><b>{name}</b> — {tj.get('n_steps')} steps · final answer:
 <span class="fa">{html.escape(str(tj.get('final_answer',''))[:160])}</span></header>
{''.join(rows)}
</body></html>"""
    path = os.path.join(run_dir, "trajectory.html")
    open(path, "w").write(out)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", nargs="?", help="a run dir (runs/pilot/C4_churn/seedN)")
    ap.add_argument("--category", help="pick random runs from this category")
    ap.add_argument("--sample", type=int, default=1, help="how many random runs to open")
    ap.add_argument("--pilot", default="runs/pilot")
    ap.add_argument("--no-open", action="store_true")
    args = ap.parse_args()

    runs = []
    if args.run:
        runs = [args.run if os.path.isabs(args.run) else os.path.join(OPSROOT, args.run)]
    else:
        base = os.path.join(OPSROOT, args.pilot)
        pat = os.path.join(base, args.category or "*", "seed*", "trajectory.json")
        cand = [os.path.dirname(p) for p in glob.glob(pat)]
        runs = random.sample(cand, min(args.sample, len(cand))) if cand else []

    if not runs:
        sys.exit("no trajectories found")
    for rd in runs:
        p = render(rd)
        print("rendered:", p)
        if not args.no_open:
            subprocess.run(["open", p])


if __name__ == "__main__":
    main()
