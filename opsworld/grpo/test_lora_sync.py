"""Run on the verl pod: python -m unittest grpo.test_lora_sync -v."""

import unittest
import os
from collections import OrderedDict
from unittest.mock import patch

import torch
from peft import LoraConfig, get_peft_model
from peft.utils.save_and_load import get_peft_model_state_dict
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from verl.utils.fsdp_utils import collect_lora_params, get_fsdp_wrap_policy


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(8, 8)

    def forward(self, inputs):
        return self.q_proj(inputs)


@unittest.skipUnless(torch.cuda.is_available(), "FSDP requires a CUDA device")
class LoraSyncTest(unittest.TestCase):
    def test_nested_leaf_adapter_is_extracted_without_full_summon(self):
        if not torch.distributed.is_initialized():
            torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
            torch.distributed.init_process_group("nccl")
        self.assertEqual(torch.distributed.get_world_size(), 2)
        torch.manual_seed(23)
        model = get_peft_model(TinyModel(), LoraConfig(r=2, lora_alpha=4, target_modules=["q_proj"]))
        expected = {key: value.detach().cpu().clone() for key, value in get_peft_model_state_dict(model).items()}
        wrapped = FSDP(
            model.cuda(),
            auto_wrap_policy=get_fsdp_wrap_policy(model, config={"min_num_params": 0}, is_lora=True),
            device_id=torch.cuda.current_device(),
            use_orig_params=False,
            sync_module_states=True,
        )
        self.assertGreater(sum(isinstance(unit, FSDP) for unit in wrapped.modules()), 1)
        with patch.object(FSDP, "summon_full_params", wraps=FSDP.summon_full_params) as summon:
            actual = collect_lora_params(wrapped, layered_summon=True, base_sync_done=True)
            self.assertTrue(actual)
            self.assertEqual(set(expected), set(actual))
            for key, value in expected.items():
                torch.testing.assert_close(actual[key], value)
            self.assertTrue(all(call.args[0] is not wrapped for call in summon.call_args_list))
        with patch("verl.utils.fsdp_utils.layered_summon_lora_params", return_value=OrderedDict()):
            with self.assertRaisesRegex(RuntimeError, "refusing full-model summon"):
                collect_lora_params(wrapped, layered_summon=True, base_sync_done=True)


if __name__ == "__main__":
    unittest.main()