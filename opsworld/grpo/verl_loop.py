"""OpsWorld multi-turn multimodal agent loop for verl v0.9.1."""
from __future__ import annotations

import base64
import fcntl
import io
import json
import os
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

from PIL import Image
from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopMetrics, AgentLoopOutput

from grpo.episode import OpsWorldEpisode


@contextmanager
def leased_worker():
    """Reserve one local CRM/Billing port pair across concurrent Ray workers."""
    root = Path(os.environ.get("OPSWORLD_GRPO_LOCKS", "/tmp/opsworld-grpo-ports"))
    root.mkdir(parents=True, exist_ok=True)
    for worker in range(1000, 2000):
        descriptor = os.open(root / f"{worker}.lock", os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(descriptor)
            continue
        try:
            yield worker
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)
        return
    raise RuntimeError("no available OpsWorld port pairs")


def vision_messages(messages: list[dict]) -> list[dict]:
    """Turn Fara's OpenAI data URLs into verl's in-memory image blocks."""
    converted = []
    for message in messages:
        content = message["content"]
        if isinstance(content, list):
            blocks = []
            for block in content:
                if block["type"] == "image_url":
                    url = block["image_url"]["url"]
                    prefix = "data:image/png;base64,"
                    if not url.startswith(prefix):
                        raise ValueError("expected a Fara PNG screenshot")
                    image = Image.open(io.BytesIO(base64.b64decode(url[len(prefix):])))
                    blocks.append({"type": "image", "image": image.convert("RGB")})
                else:
                    blocks.append(block)
            converted.append({"role": message["role"], "content": blocks})
        else:
            converted.append(message)
    return converted


class OpsWorldAgentLoop(AgentLoopBase):
    """Generate Fara actions on-policy and return a masked token trace to verl."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt_length = self.rollout_config.prompt_length
        self.response_length = self.rollout_config.response_length

    async def run(self, sampling_params: dict, priority: int = 0, **kwargs) -> AgentLoopOutput:
        task = kwargs["extra_info"]
        seed, category = int(task["seed"]), str(task["category"])
        model_step = int(kwargs["global_steps"]) - 1
        metrics = AgentLoopMetrics()
        response_mask: list[int] = []
        response_logprobs: list[float] | None = [] if self.rollout_config.calculate_log_probs else None
        request_id = uuid4().hex
        with leased_worker() as worker:
            episode = OpsWorldEpisode(worker, os.environ.get("OPSWORLD_GRPO_OUTPUT", "runs/grpo"),
                                      max_rounds=min(int(task.get("max_rounds", 20)),
                                                     int(os.environ.get("OPSWORLD_GRPO_MAX_ROUNDS", "20"))),
                                      keep_all_screenshots=True)
            try:
                return await self._rollout(episode, seed, category, sampling_params, priority,
                                           metrics, response_mask, response_logprobs, request_id, model_step)
            finally:
                await episode.close()

    async def _rollout(self, episode, seed, category, sampling_params, priority,
                       metrics, response_mask, response_logprobs, request_id, model_step):
        score = 0.0
        turns = 0
        min_model_step = None
        max_model_step = None
        images: list[Image.Image] = []
        messages = vision_messages(await episode.start(seed, category))
        self._assert_mm_supported(True)
        images = (await self.process_multi_modal_info(messages))["images"]
        prompt_ids = await self.ct_build_initial_tokens(messages, images=images)
        runtime_ids = prompt_ids
        while turns < episode.max_rounds and len(response_mask) < self.response_length:
            remaining = self.response_length - len(response_mask)
            if remaining <= 8:
                break
            params = {**sampling_params, "max_tokens": min(512, remaining - 8)}
            output = await self.server_manager.generate(
                request_id=request_id, prompt_ids=runtime_ids, sampling_params=params,
                image_data=images, mm_processor_kwargs=self._get_mm_processor_kwargs(),
                priority=priority,
            )
            if not output.token_ids:
                raise ValueError("verl returned no action tokens")
            if response_logprobs is not None and (
                output.log_probs is None or len(output.log_probs) != len(output.token_ids)
            ):
                raise ValueError("verl did not return aligned action logprobs")
            generated_step = output.extra_fields.get("global_steps")
            generated_step = model_step if generated_step is None else int(generated_step)
            min_model_step = generated_step if min_model_step is None else min(min_model_step, generated_step)
            max_model_step = generated_step if max_model_step is None else max(max_model_step, generated_step)
            turns += 1
            merge, response_mask, response_logprobs = await self.ct_merge_assistant_token(
                runtime_ids, output.token_ids, response_mask, response_logprobs,
                assistant_logprobs=output.log_probs if response_logprobs is not None else None,
            )
            runtime_ids = merge.token_ids
            text = self.tokenizer.decode(output.token_ids, skip_special_tokens=True)
            next_messages, done, score, _ = await episode.step(text)
            if done or len(response_mask) >= self.response_length:
                break
            updated = vision_messages(next_messages)
            appended = updated[len(messages):]
            if len(appended) != 2 or appended[0]["role"] != "assistant" or appended[1]["role"] != "user":
                raise ValueError("Fara conversation is not append-only")
            context = messages + appended
            merge, new_mask, new_logprobs = await self.ct_merge_context_msg(
                messages + appended[:1], context, runtime_ids, response_mask, response_logprobs,
            )
            if len(new_mask) > self.response_length:
                break
            if len(merge.token_ids) > self.rollout_config.max_model_len:
                raise ValueError("screenshot history exceeds verl max_model_len")
            runtime_ids, response_mask, response_logprobs = merge.token_ids, new_mask, new_logprobs
            images.extend((await self.process_multi_modal_info(appended[1:]))["images"])
            messages = context
        if not episode.done:
            result = await episode.finish()
            score = float(result["success"])
        reward_log = os.environ.get("OPSWORLD_GRPO_REWARD_LOG")
        if reward_log:
            path = Path(reward_log)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as record:
                fcntl.flock(record, fcntl.LOCK_EX)
                record.write(json.dumps({"step": model_step + 1, "seed": seed, "category": category,
                                         "reward": score, "actions": turns, "request_id": request_id}) + "\n")
        return AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=runtime_ids[len(prompt_ids):],
            response_mask=response_mask,
            response_logprobs=response_logprobs,
            multi_modal_data={"images": images},
            mm_processor_kwargs=self._get_mm_processor_kwargs(),
            reward_score=score,
            num_turns=turns * 2,
            metrics=metrics,
            extra_fields={"seed": seed, "category": category,
                          "min_global_steps": min_model_step if min_model_step is not None else model_step,
                          "max_global_steps": max_model_step if max_model_step is not None else model_step},
        )