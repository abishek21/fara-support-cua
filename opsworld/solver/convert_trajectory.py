"""
convert_trajectory.py — turn a solver trajectory into Fara-7B SFT samples.

The solver logs, per step: a CLEAN screenshot, a natural-language THOUGHT, the
chosen ACTION, and the chosen element's bbox-center COORD. This converter emits
one training sample PER STEP in Fara's native format:

    thought  +  <tool_call>{computer_use: {action, coordinate/text/url/...}}</tool_call>

with the clean screenshot(s) as the image(s). SoM actions are translated to the
STUDENT's action space (what Fara-7B can emit at inference):

    click(mark)      -> left_click(coordinate=[x,y])
    type(mark,text)  -> type(coordinate=[x,y], text=..., press_enter=true)
    goto(url)        -> visit_url(url)
    terminate(ans)   -> terminate(status="success")   (answer becomes the thought)
    select(...)      -> UNSUPPORTED  (Fara has no select; such steps are flagged
                        and the trajectory is rejected so we never train on
                        actions the student cannot replicate)

Context: history thoughts/actions are provided as text; only the most recent
`max_images` screenshots are kept as images (Fara keeps 3). Loss is on the single
assistant turn (the current step's thought+action).

Usage:
    python -m solver.convert_trajectory --run runs/sc_churn_ACC-1046
    python -m solver.convert_trajectory --all          # convert every run/*/trajectory.json
Output: <run>/sft_samples.jsonl
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "..", "runs")

# Short Fara-style system prompt (the real contract the student sees).
FARA_SYSTEM = (
    "You are a web automation agent that performs actions on websites to fulfill "
    "user requests by calling tools. You perceive the page via a screenshot and the "
    "current URL, and you output a brief reasoning followed by a single tool call. "
    "Available actions include left_click, type, scroll, visit_url, wait, and terminate. "
    "Predict click/type coordinates directly in the screenshot. Stop at Critical Points "
    "that require user permission or personal data."
)


class Unsupported(Exception):
    pass


def map_action(step: dict) -> tuple[str, dict]:
    """Return (thought, fara_args). Raises Unsupported for non-replicable actions."""
    act = step["action"]
    t = act.get("type")
    thought = step.get("thought", "")
    coord = step.get("coord")

    if t == "click":
        if not coord:
            raise Unsupported("click without coord")
        return thought, {"action": "left_click", "coordinate": coord}
    if t == "type":
        if not coord:
            raise Unsupported("type without coord")
        return thought, {"action": "type", "coordinate": coord,
                         "text": act.get("text", ""), "press_enter": True}
    if t == "goto":
        return thought, {"action": "visit_url", "url": act.get("url", "")}
    if t == "terminate":
        # the student ends with a thought summarizing the result
        return (act.get("answer") or thought or "Task complete."), \
               {"action": "terminate", "status": "success"}
    if t == "select":
        raise Unsupported("select is not in Fara's action space (use clickable controls)")
    raise Unsupported(f"unknown action type: {t}")


def label_for(thought: str, args: dict) -> str:
    tool = {"name": "computer_use", "arguments": args}
    return f"{thought}\n<tool_call>\n{json.dumps(tool)}\n</tool_call>"


def convert_run(run_dir: str, instruction: str | None = None,
                max_images: int = 3) -> dict:
    tj = json.load(open(os.path.join(run_dir, "trajectory.json")))
    steps = tj["steps"]
    task = instruction or tj.get("instruction") or "(task)"

    # pre-map all steps; reject trajectory if any step is unsupported
    mapped = []
    for i, s in enumerate(steps):
        try:
            th, args = map_action(s)
        except Unsupported as e:
            return {"ok": False, "reason": f"step {i}: {e}", "n_samples": 0}
        mapped.append((s, th, args))

    samples = []
    for t in range(len(mapped)):
        s, th, args = mapped[t]
        # images: current + up to (max_images-1) prior, as absolute paths
        imgs = []
        for k in range(max(0, t - (max_images - 1)), t + 1):
            p = os.path.abspath(os.path.join(run_dir, steps[k].get("screenshot", f"step{k}.png")))
            if os.path.exists(p):
                imgs.append(p)

        # history as text (prior thoughts + actions), no images
        hist_lines = []
        for k in range(t):
            _, pth, pargs = mapped[k]
            hist_lines.append(f"{k+1}. {pth}  [{pargs['action']}"
                              f"{(' ' + str(pargs.get('coordinate'))) if 'coordinate' in pargs else ''}"
                              f"{(' ' + pargs.get('url','')) if pargs.get('url') else ''}]")
        history_text = ("\nPrior steps:\n" + "\n".join(hist_lines) if hist_lines else "")

        user_text = (f"Task: {task}\n"
                     f"Current URL: {s.get('url','')}{history_text}\n"
                     "Here is the current screenshot. Think about what to do next.")

        content = ([{"type": "image", "path": p} for p in imgs]
                   + [{"type": "text", "text": user_text}])
        sample = {
            "id": f"{tj.get('task_id','task')}_step{t}",
            "images": imgs,
            "messages": [
                {"role": "system", "content": FARA_SYSTEM},
                {"role": "user", "content": content},
                {"role": "assistant", "content": label_for(th, args)},  # loss target
            ],
        }
        samples.append(sample)

    out = os.path.join(run_dir, "sft_samples.jsonl")
    with open(out, "w") as f:
        for smp in samples:
            f.write(json.dumps(smp) + "\n")
    return {"ok": True, "n_samples": len(samples), "out": out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", help="a single run dir (e.g. runs/sc_churn_ACC-1046)")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--instruction", default=None)
    ap.add_argument("--show", action="store_true", help="print the first sample's label")
    args = ap.parse_args()

    run_dirs = []
    if args.all:
        run_dirs = [os.path.dirname(p) for p in glob.glob(os.path.join(RUNS, "*", "trajectory.json"))]
    elif args.run:
        run_dirs = [args.run if os.path.isabs(args.run) else os.path.join(HERE, "..", args.run)]
    else:
        sys.exit("pass --run <dir> or --all")

    total = 0
    for rd in run_dirs:
        res = convert_run(rd, instruction=args.instruction)
        name = os.path.basename(rd.rstrip("/"))
        if res["ok"]:
            total += res["n_samples"]
            print(f"  ✅ {name}: {res['n_samples']} samples -> {os.path.basename(res['out'])}")
            if args.show:
                first = json.loads(open(res["out"]).readline())
                print("     --- sample label (loss target) ---")
                print("     " + first["messages"][-1]["content"].replace("\n", "\n     "))
                print(f"     images: {len(first['images'])}")
        else:
            print(f"  ❌ {name}: REJECTED ({res['reason']})")
    print(f"\nTotal SFT samples: {total}")


if __name__ == "__main__":
    main()
