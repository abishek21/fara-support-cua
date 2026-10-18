"""Seeded training tasks for the OpsWorld computer-use RL environment."""
from __future__ import annotations

import asyncio
from pathlib import Path
from urllib.parse import urlsplit

from common.world import generate_world
from solver.collect import _get, _post, spawn_env
from solver.phrase import CATEGORY, phrase
from verify import verify_task

TRAIN_SEEDS = range(1, 319)


def training_task(seed: int, category: str) -> dict:
    if seed not in TRAIN_SEEDS:
        raise ValueError(f"seed {seed} is not in the training split")
    kind = next((kind for kind, label in CATEGORY.items() if label == category), None)
    if kind is None:
        raise ValueError(f"unknown category: {category}")
    accounts, scenarios = generate_world(seed)
    scenario = next(scenario for scenario in scenarios if scenario.kind == kind)
    return {
        "task_id": f"{category}_seed{seed}",
        "category": category,
        "seed": seed,
        "start_system": "crm" if kind == "reconcile_churn" else "billing",
        "instruction": phrase(scenario, accounts),
        "verify": scenario.verify,
    }


class OpsWorldEpisode:
    """One isolated browser episode; the caller supplies each sampled Fara response."""

    def __init__(self, worker: int, output_dir: str | Path, max_rounds: int = 20,
                 keep_all_screenshots: bool = False):
        self.worker = worker
        self.output_dir = Path(output_dir)
        self.max_rounds = max_rounds
        self.keep_all_screenshots = keep_all_screenshots
        self.environment = None
        self.agent = None
        self.task = None
        self.before = None
        self.rounds = 0
        self.done = False

    async def start(self, seed: int, category: str) -> list[dict]:
        from fara.fara_7b import FaraAgent
        from fara.fara_7b.browser.browser_bb import BrowserBB
        from fara.fara_7b.fara_types import ImageObj, UserMessage

        if self.environment is not None:
            raise RuntimeError("episode already started")
        self.task = training_task(seed, category)
        try:
            self.environment = await asyncio.to_thread(
                spawn_env, self.worker, str(self.output_dir / "_envs"))
            crm, billing = self.environment["crm"], self.environment["billing"]
            await asyncio.to_thread(_post, f"{crm}/api/reset?seed={seed}")
            await asyncio.to_thread(_post, f"{billing}/api/reset?seed={seed}")
            self.before = await self._snapshot()
            start_url = (crm if self.task["start_system"] == "crm" else billing) + "/"
            browser = BrowserBB(
                headless=True, viewport_height=900, viewport_width=1440,
                page_script_path=None, browser_channel="firefox", browser_data_dir=None,
                downloads_folder=None, to_resize_viewport=True, single_tab_mode=True,
                animate_actions=False, use_browser_base=False,
            )
            self.agent = FaraAgent(browser_manager=browser,
                                   client_config={"api_key": "not-needed", "base_url": "http://localhost:1/v1"},
                                   start_page=start_url, save_screenshots=False,
                                   max_rounds=self.max_rounds)
            if self.keep_all_screenshots:
                self.agent.max_n_images = self.max_rounds + 1
            await self.agent.initialize()
            image = await self.agent._get_scaled_screenshot()
            instruction = (self.task["instruction"] + f"\n\nThe Contoso CRM system is at {crm} "
                           f"and the Contoso Billing system is at {billing}. Use only these "
                           "two systems; to switch systems, navigate to the appropriate URL.")
            self.agent._chat_history.append(UserMessage(
                content=[ImageObj.from_pil(image), instruction], is_original=True))
            return self._messages(image)
        except BaseException:
            await self.close()
            raise

    def _messages(self, image) -> list[dict]:
        from fara.fara_7b.fara_types import message_to_openai_format

        history = self.agent.maybe_remove_old_screenshots(self.agent._chat_history)
        system, _ = self.agent._get_system_message(image)
        return [message_to_openai_format(message) for message in system + history]

    async def _snapshot(self) -> dict:
        crm, billing = self.environment["crm"], self.environment["billing"]
        return {"crm": await asyncio.to_thread(_get, crm + "/api/snapshot"),
                "billing": await asyncio.to_thread(_get, billing + "/api/snapshot")}

    def _allowed_url(self, url: str) -> bool:
        target = urlsplit(url)
        return target.scheme == "http" and target.netloc in {
            urlsplit(self.environment["crm"]).netloc,
            urlsplit(self.environment["billing"]).netloc,
        } and target.path not in ("/api",) and not target.path.startswith("/api/")

    async def step(self, response: str) -> tuple[list[dict] | None, bool, float, dict]:
        from fara.fara_7b.fara_types import AssistantMessage, FunctionCall, ImageObj, UserMessage
        from fara.fara_7b.utils import get_trimmed_url

        if self.agent is None or self.done:
            raise RuntimeError("episode is not active")
        self.rounds += 1
        try:
            thoughts, action = self.agent._parse_thoughts_and_action(response)
            arguments = action["arguments"]
            name = arguments["action"]
            if action["name"] != "computer_use":
                raise ValueError("unexpected tool")
            if name == "visit_url" and not self._allowed_url(arguments["url"]):
                raise ValueError("navigation outside the two sandbox UIs")
            if name == "web_search":
                raise ValueError("web search is outside the sandbox")
            self.agent._chat_history.append(AssistantMessage(content=response))
            arguments["thoughts"] = thoughts
            await self.agent.execute_action([FunctionCall(id="dummy", **action)])
        except (ValueError, KeyError, TypeError, SyntaxError) as exc:
            self.done = True
            return None, True, 0.0, {"reason": f"invalid action: {exc}"}
        except Exception as exc:
            self.done = True
            return None, True, 0.0, {"reason": f"action failed: {exc}"}

        self.done = name == "terminate" or self.rounds >= self.max_rounds
        if self.done:
            result = verify_task(self.task["verify"], self.before, await self._snapshot())
            return None, True, float(result["success"]), result

        image = await self.agent._get_scaled_screenshot()
        url = await self.agent._playwright_controller.get_page_url(self.agent._page)
        user_text = (f"Current URL: {get_trimmed_url(url, max_len=self.agent.max_url_chars)}\n"
                     + self.agent.USER_MESSAGE)
        self.agent._chat_history.append(UserMessage(content=[ImageObj.from_pil(image), user_text]))
        return self._messages(image), False, 0.0, {}

    async def finish(self) -> dict:
        if self.agent is None:
            raise RuntimeError("episode is not active")
        self.done = True
        return verify_task(self.task["verify"], self.before, await self._snapshot())

    async def close(self) -> None:
        if self.agent is not None:
            try:
                await self.agent.close()
            finally:
                self.agent = None
        if self.environment is not None:
            for process in self.environment["procs"]:
                process.terminate()
                await asyncio.to_thread(process.wait, 10)
            self.environment = None