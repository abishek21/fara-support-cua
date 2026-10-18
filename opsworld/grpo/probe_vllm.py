"""Exercise one multimodal Fara generation with the GRPO vLLM/LoRA settings."""

import argparse

from PIL import Image
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--image", required=True)
    parser.add_argument("--tensor-parallel-size", type=int, default=2)
    parser.add_argument("--max-num-seqs", type=int)
    parser.add_argument("--disable-custom-all-reduce", action="store_true")
    args = parser.parse_args()

    image = Image.open(args.image).convert("RGB")
    processor = AutoProcessor.from_pretrained(args.model)
    prompt = processor.apply_chat_template(
        [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": "Inspect this customer screen and decide the next action."},
        ]}],
        tokenize=False,
        add_generation_prompt=True,
    )

    print(f"loading vLLM TP{args.tensor_parallel_size} {'with LoRA' if args.adapter else 'without LoRA'}", flush=True)
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        tensor_parallel_size=args.tensor_parallel_size,
        disable_custom_all_reduce=args.disable_custom_all_reduce,
        enable_lora=bool(args.adapter),
        max_lora_rank=16,
        max_model_len=32768,
        max_num_seqs=args.max_num_seqs or (1 if args.tensor_parallel_size == 1 else 16),
        gpu_memory_utilization=0.8 if args.tensor_parallel_size == 1 else 0.35,
        enforce_eager=True,
        enable_sleep_mode=True,
    )
    print("generating one image request", flush=True)
    output = llm.generate(
        [{"prompt": prompt, "multi_modal_data": {"image": image}}],
        sampling_params=SamplingParams(temperature=1.0, max_tokens=64),
        lora_request=LoRARequest("sft-probe", 1, args.adapter) if args.adapter else None,
    )[0]
    print(f"generated {len(output.outputs[0].token_ids)} tokens", flush=True)
    print(output.outputs[0].text, flush=True)


if __name__ == "__main__":
    main()