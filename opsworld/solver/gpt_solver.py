"""
gpt_solver.py — GPT-5.4 agent that solves OpsWorld tasks via the SoM adapter.

Observe (SoM screenshot + numbered element list + URL) -> GPT-5.4 picks an
element number + action -> execute on the real DOM node -> repeat. Logs a
trajectory (clean screenshots + thought/action/coords) for later distillation.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.azure_llm import AzureLLM
from solver.browser import SoMBrowser

SYSTEMS = {
    "CRM": "http://localhost:8001",
    "Billing": "http://localhost:8002",
}


def _system_prompt(crm_url: str, billing_url: str) -> str:
    return f"""You are an expert Contoso support operations agent. You work across TWO internal web systems and must reconcile information between them:
- CRM  = {crm_url}  (account, status active/churned/suspended, plan, case notes; actions: add a case note, update status)
- Billing = {billing_url}  (invoices, subscription billing on/off, status, failed payments, refunds; actions: refund an invoice, cancel billing, block subscription; a Reports tab with filters)

You are shown a screenshot with numbered red boxes (Set-of-Marks) and a list of the numbered interactable elements. Choose elements BY NUMBER. Only use the systems above; never use web search or external sites.

To switch systems or pages, use the "goto" action with the URL (e.g. the Billing Reports page is {billing_url}/reports). To find an account, type its name into the search box, then click the matching result. When a confirmation dialog appears, click its Confirm button to complete the action.

Think step by step, act carefully, and do NOT take actions that aren't required by the task (no over-reach, no punitive actions unless the task/policy calls for it). When the task is fully complete, use "terminate".

In the "thought" field, describe your INTENT in plain language (e.g. "cancel billing for this account", "open the account to review failed payments"). NEVER mention element numbers, boxes, or marks in the thought — the thought must read like a human agent's reasoning, independent of the UI annotations.

Respond ONLY as JSON:
{{"thought": "<one concise sentence>",
 "action": {{"type": "click|type|select|goto|terminate",
            "mark": <element number, for click/type/select>,
            "text": "<for type>", "value": "<for select>",
            "url": "<for goto>", "answer": "<for terminate>"}}}}"""


# default system prompt (single-env convenience)
SYSTEM_PROMPT = _system_prompt(SYSTEMS["CRM"], SYSTEMS["Billing"])


def _marks_text(marks: list[dict]) -> str:
    lines = []
    for m in marks:
        extra = f" options={m['options']}" if m.get("options") else ""
        lines.append(f"[{m['id']}] {m['role']} \"{m['text']}\"{extra}")
    return "\n".join(lines)


def _rl_call(rate_limiter, llm, system, user, image):
    with rate_limiter:
        return llm.chat_json(system, user, image_png=image, temperature=0)


def solve(task: dict, headful: bool = False, max_steps: int = 25,
          out_dir: str | None = None, send_image: bool = True, verbose: bool = True,
          crm_url: str | None = None, billing_url: str | None = None,
          llm=None, rate_limiter=None):
    """Run GPT-5.4 on one task. Returns {final_answer, steps:[...]}.
    crm_url/billing_url override the default system URLs (for parallel workers).
    `llm` (shared AzureLLM) and `rate_limiter` enable safe concurrency.
    """
    crm_url = crm_url or SYSTEMS["CRM"]
    billing_url = billing_url or SYSTEMS["Billing"]
    system_prompt = _system_prompt(crm_url, billing_url)
    llm = llm or AzureLLM()
    br = SoMBrowser(headful=headful)
    start = task.get("start_url", crm_url + "/")
    br.goto(start)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    history, steps = [], []
    final_answer = "<no answer>"
    try:
        for i in range(max_steps):
            marks = br.mark()
            som_png = br.screenshot(annotated=True)
            clean_png = br.screenshot(annotated=False)  # also clears overlays
            if out_dir:
                open(os.path.join(out_dir, f"step{i}.png"), "wb").write(clean_png)

            user = (
                f"Current URL: {br.url()}\n"
                f"Task: {task['instruction']}\n\n"
                f"Interactable elements (choose by number):\n{_marks_text(marks)}\n\n"
                + ("Recent steps:\n" + "\n".join(history[-8:]) + "\n\n" if history else "")
                + "Respond with the JSON action."
            )
            resp = llm.chat_json(system_prompt, user,
                                 image_png=som_png if send_image else None,
                                 temperature=0) if rate_limiter is None else _rl_call(
                                     rate_limiter, llm, system_prompt, user,
                                     som_png if send_image else None)
            thought = resp.get("thought", "")
            action = resp.get("action", {})
            atype = action.get("type")
            if verbose:
                print(f"\nStep {i+1}: {thought}\n  action: {json.dumps(action)}")

            # record the chosen element's coords + metadata (for training later)
            coord, element = None, None
            if "mark" in action and isinstance(action["mark"], int):
                hit = next((m for m in marks if m["id"] == action["mark"]), None)
                if hit:
                    coord = [hit["x"], hit["y"]]
                    element = {"role": hit["role"], "text": hit["text"]}
            steps.append({"thought": thought, "action": action, "coord": coord,
                          "element": element, "url": br.url(),
                          "screenshot": f"step{i}.png"})
            history.append(f"{i+1}. {thought} -> {atype} {action.get('mark','')}"
                           f"{action.get('url','')}")

            try:
                if atype == "terminate":
                    final_answer = action.get("answer", "done")
                    break
                elif atype == "goto":
                    br.goto(action["url"])
                elif atype == "click":
                    br.click(action["mark"])
                elif atype == "type":
                    br.type(action["mark"], action.get("text", ""))
                elif atype == "select":
                    br.select(action["mark"], action.get("value", ""))
                else:
                    history.append(f"   (unknown action {atype})")
            except Exception as e:
                if verbose:
                    print(f"  ! action error: {e}")
                history.append(f"   (action error: {e})")
    finally:
        br.close()

    result = {"task_id": task.get("task_id", "task"), "final_answer": final_answer,
              "n_steps": len(steps), "steps": steps}
    if out_dir:
        json.dump(result, open(os.path.join(out_dir, "trajectory.json"), "w"), indent=2)
    return result
