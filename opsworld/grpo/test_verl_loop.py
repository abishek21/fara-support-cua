"""Contract tests for the verl rollout adapter, without a GPU dependency."""
import base64
import io
import json
import os
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from PIL import Image


class FakeOutput:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class FakeServer:
    def __init__(self):
        self.outputs = [FakeOutput(token_ids=[11], log_probs=[-0.5], extra_fields={"global_steps": 0}),
                        FakeOutput(token_ids=[12], log_probs=[-0.2], extra_fields={"global_steps": 1})]

    async def generate(self, **kwargs):
        assert len(kwargs["image_data"]) == 3 - len(self.outputs)
        return self.outputs.pop(0)


class FakeBase:
    def __init__(self, *args, **kwargs):
        self.rollout_config = types.SimpleNamespace(prompt_length=256, response_length=32,
                                                     max_model_len=512, calculate_log_probs=True)
        self.server_manager = FakeServer()
        self.tokenizer = types.SimpleNamespace(decode=lambda ids, **kwargs: {11: "wait", 12: "terminate"}[ids[0]])

    def _assert_mm_supported(self, present):
        assert present

    def _get_mm_processor_kwargs(self):
        return {}

    async def process_multi_modal_info(self, messages):
        return {"images": [block["image"] for message in messages
                if isinstance(message["content"], list)
                for block in message["content"] if block["type"] == "image"]}

    async def ct_build_initial_tokens(self, messages, images):
        assert len(images) == 1
        return [1, 2]

    async def ct_merge_assistant_token(self, ids, generated, mask, logprobs, assistant_logprobs):
        return (types.SimpleNamespace(token_ids=ids + generated), mask + [1] * len(generated),
                logprobs + assistant_logprobs)

    async def ct_merge_context_msg(self, previous, updated, ids, mask, logprobs):
        assert len(updated) == len(previous) + 1
        return types.SimpleNamespace(token_ids=ids + [5, 6, 7]), mask + [0, 0, 0], logprobs + [0, 0, 0]


def screenshot():
    image = Image.new("RGB", (28, 28), "red")
    data = io.BytesIO()
    image.save(data, format="PNG")
    return {"type": "image_url", "image_url": {
        "url": "data:image/png;base64," + base64.b64encode(data.getvalue()).decode()}}


class FakeEpisode:
    instances = []

    def __init__(self, worker, output_dir, max_rounds, keep_all_screenshots):
        assert keep_all_screenshots
        self.max_rounds = max_rounds
        self.done = False
        self.actions = []
        self.messages = [{"role": "system", "content": "system"},
                         {"role": "user", "content": [screenshot(), {"type": "text", "text": "task"}]}]
        self.instances.append(self)

    async def start(self, seed, category):
        assert (seed, category) == (1, "C5_below")
        return self.messages

    async def step(self, text):
        self.actions.append(text)
        if text == "wait":
            self.messages.extend(({"role": "assistant", "content": text},
                                  {"role": "user", "content": [screenshot(), {"type": "text", "text": "next"}]}))
            return self.messages, False, 0.0, {}
        self.done = True
        return None, True, 1.0, {"success": True}

    async def close(self):
        self.closed = True


class VerlLoopTests(unittest.IsolatedAsyncioTestCase):
    async def test_mask_images_reward_and_cleanup(self):
        from importlib import import_module

        root = types.ModuleType("verl")
        experimental = types.ModuleType("verl.experimental")
        package = types.ModuleType("verl.experimental.agent_loop")
        module = types.ModuleType("verl.experimental.agent_loop.agent_loop")
        module.AgentLoopBase = FakeBase
        module.AgentLoopMetrics = lambda: FakeOutput()
        module.AgentLoopOutput = FakeOutput
        with patch.dict(sys.modules, {"verl": root, "verl.experimental": experimental,
                                      "verl.experimental.agent_loop": package,
                                      "verl.experimental.agent_loop.agent_loop": module}):
            from grpo import verl_loop

            with tempfile.TemporaryDirectory() as directory:
                reward_log = os.path.join(directory, "rewards.jsonl")
                with patch.dict(os.environ, {"OPSWORLD_GRPO_REWARD_LOG": reward_log}):
                    with patch.object(verl_loop, "OpsWorldEpisode", FakeEpisode):
                        result = await verl_loop.OpsWorldAgentLoop().run(
                            {"temperature": 1}, global_steps=1,
                            extra_info={"seed": 1, "category": "C5_below", "max_rounds": 3})
                with open(reward_log, encoding="utf-8") as records:
                    logged = json.loads(records.readline())
                self.assertEqual((logged["step"], logged["seed"], logged["category"], logged["reward"]),
                                 (1, 1, "C5_below", 1.0))
            self.assertEqual(result.prompt_ids, [1, 2])
            self.assertEqual(result.response_ids, [11, 5, 6, 7, 12])
            self.assertEqual(result.response_mask, [1, 0, 0, 0, 1])
            self.assertEqual(result.response_logprobs, [-0.5, 0, 0, 0, -0.2])
            self.assertEqual(len(result.multi_modal_data["images"]), 2)
            self.assertEqual(result.reward_score, 1.0)
            self.assertEqual(result.extra_fields["min_global_steps"], 0)
            self.assertEqual(result.extra_fields["max_global_steps"], 1)
            self.assertTrue(FakeEpisode.instances[-1].closed)
            no_version = verl_loop.OpsWorldAgentLoop()
            for output in no_version.server_manager.outputs:
                output.extra_fields["global_steps"] = None
            with patch.object(verl_loop, "OpsWorldEpisode", FakeEpisode):
                fallback = await no_version.run(
                    {"temperature": 1}, global_steps=1,
                    extra_info={"seed": 1, "category": "C5_below", "max_rounds": 3})
            self.assertEqual(fallback.extra_fields["min_global_steps"], 0)
            self.assertEqual(fallback.extra_fields["max_global_steps"], 0)
            newer = verl_loop.OpsWorldAgentLoop()
            for step, output in zip((3, 4), newer.server_manager.outputs):
                output.extra_fields["global_steps"] = step
            with patch.object(verl_loop, "OpsWorldEpisode", FakeEpisode):
                reported = await newer.run(
                    {"temperature": 1}, global_steps=3,
                    extra_info={"seed": 1, "category": "C5_below", "max_rounds": 3})
            self.assertEqual(reported.extra_fields["min_global_steps"], 3)
            self.assertEqual(reported.extra_fields["max_global_steps"], 4)
            with patch.dict(os.environ, {"OPSWORLD_GRPO_MAX_ROUNDS": "2"}):
                with patch.object(verl_loop, "OpsWorldEpisode", FakeEpisode):
                    await verl_loop.OpsWorldAgentLoop().run(
                        {"temperature": 1}, global_steps=1,
                        extra_info={"seed": 1, "category": "C5_below", "max_rounds": 3})
            self.assertEqual(FakeEpisode.instances[-1].max_rounds, 2)
            with verl_loop.leased_worker() as first:
                with verl_loop.leased_worker() as second:
                    self.assertNotEqual(first, second)
            missing = verl_loop.OpsWorldAgentLoop()
            missing.server_manager.outputs[0].log_probs = None
            with patch.object(verl_loop, "OpsWorldEpisode", FakeEpisode):
                with self.assertRaisesRegex(ValueError, "action logprobs"):
                    await missing.run({"temperature": 1}, global_steps=1, extra_info={
                        "seed": 1, "category": "C5_below", "max_rounds": 3})
            self.assertTrue(FakeEpisode.instances[-1].closed)


if __name__ == "__main__":
    unittest.main()