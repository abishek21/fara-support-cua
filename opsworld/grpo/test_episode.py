"""Model-free smoke checks for seeded GRPO browser episodes."""
from collections import Counter
from itertools import islice
import os
from random import Random
import unittest
from unittest.mock import patch

from grpo.episode import OpsWorldEpisode, training_task
from grpo.make_dataset import rows


def action(name, **arguments):
    import json

    return ("Done.\n<tool_call>\n" + json.dumps({"name": "computer_use", "arguments": {
        "action": name, **arguments}}) + "\n</tool_call>")


class EpisodeTests(unittest.IsolatedAsyncioTestCase):
    def test_balanced_training_prefix(self):
        seeds = list(range(1, 300))
        Random(42).shuffle(seeds)
        tasks = list(islice(rows(seeds, spread_seeds=True), 300))
        first = Counter(task["extra_info"]["category"] for task in tasks[:100])
        extended = Counter(task["extra_info"]["category"] for task in tasks)
        self.assertEqual(sorted(first.values()), [16, 16, 17, 17, 17, 17])
        self.assertEqual(set(extended.values()), {50})
        self.assertEqual(len({task["extra_info"]["seed"] for task in tasks[:100]}), 100)
        self.assertEqual(len({task["extra_info"]["index"] for task in tasks}), 300)

    def test_train_seeds_only(self):
        self.assertEqual(training_task(1, "C4_churn")["start_system"], "crm")
        with self.assertRaises(ValueError):
            training_task(9000, "C5_below")

    async def test_terminal_reward_and_isolation(self):
        for category, expected in (("C5_below", 1.0), ("C2_cancel", 0.0)):
            episode = OpsWorldEpisode(20, "runs/grpo_test", max_rounds=2)
            try:
                messages = await episode.start(1, category)
                self.assertEqual([message["role"] for message in messages], ["system", "user"])
                _, done, reward, result = await episode.step(action("terminate", status="success"))
                self.assertTrue(done)
                self.assertEqual(reward, expected)
                self.assertEqual(result["success"], bool(expected))
            finally:
                await episode.close()

    async def test_blocked_navigation(self):
        episode = OpsWorldEpisode(20, "runs/grpo_test", max_rounds=2)
        try:
            await episode.start(1, "C5_below")
            _, done, reward, result = await episode.step(action("visit_url", url="file:///etc/passwd"))
            self.assertTrue(done)
            self.assertEqual(reward, 0.0)
            self.assertIn("outside", result["reason"])
        finally:
            await episode.close()

    async def test_multiturn_append_only_history(self):
        episode = OpsWorldEpisode(20, "runs/grpo_test", max_rounds=5,
                                  keep_all_screenshots=True)
        with patch.dict(os.environ, {"FARA_POST_ACTION_SLEEP": "0"}):
            try:
                messages = await episode.start(1, "C5_below")
                for turn in range(3):
                    updated, done, reward, _ = await episode.step(action("wait", time=0))
                    self.assertFalse(done)
                    self.assertEqual(reward, 0.0)
                    self.assertEqual(updated[:len(messages)], messages)
                    self.assertEqual([item["role"] for item in updated[len(messages):]],
                                     ["assistant", "user"])
                    messages = updated
                _, done, reward, _ = await episode.step(action("terminate", status="success"))
                self.assertTrue(done)
                self.assertEqual(reward, 1.0)
            finally:
                await episode.close()


if __name__ == "__main__":
    unittest.main()