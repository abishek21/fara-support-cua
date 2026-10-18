"""
prepare_swift_data.py — convert the HF dataset `abishekk210698/opsworld-sft`.

Use --format fara for direct HF/PEFT training: reconstruct Fara's native tool
prompt and alternating screenshot/assistant turns from trajectory IDs, resize
the images using Fara's screenshot dimensions, transform the action coordinates
into that model-image frame, and save sample target overlays under inspect/.

The legacy default --format swift writes the ms-swift multimodal format:

    {"messages": [{"role": "system",    "content": "..."},
                  {"role": "user",      "content": "<image><image>Task: ..."},
                  {"role": "assistant", "content": "thought\n<tool_call>...</tool_call>"}],
     "images": ["/abs/path/train/000123_img0.png", ".../000123_img1.png"]}

ms-swift maps each `<image>` tag (in order) to the corresponding entry in `images`.
The embedded PNG bytes are written out to <out>/images/ so training reads from disk.

Loss masking: ms-swift trains on assistant turns by default, so the single
assistant turn (thought + <tool_call>) is the loss target — matching how the
samples were built in solver/convert_trajectory.py.

Usage (on the Mac with the Fara clone installed):
    python prepare_swift_data.py \
        --format fara \
        --repo abishekk210698/opsworld-sft \
        --out ../runs/sft_native

Produces:
    ../runs/sft_native/train.jsonl
    ../runs/sft_native/val.jsonl
    ../runs/sft_native/images/...
    ../runs/sft_native/inspect/...
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from urllib.parse import urlsplit

HERE = os.path.dirname(os.path.abspath(__file__))
OPSROOT = os.path.dirname(HERE)
sys.path.insert(0, OPSROOT)

try:
    from common.hf_auth import get_token
except Exception:  # pragma: no cover — allow running standalone on the pod
    def get_token():
        return os.environ.get("HF_TOKEN")


IMAGE_TAG = "<image>"
STEP_ID = re.compile(r"^(.*)_step(\d+)$")


def _flatten_content(content):
    """Turn a list-of-parts content into (text_with_image_tags, n_images).

    Image placeholders ({"type":"image"}) are replaced, in order, by <image> tags
    placed at the front of the text (ms-swift accepts tags anywhere, but Fara/Qwen
    expect the image(s) before the task text)."""
    if isinstance(content, str):
        return content, 0
    tags = []
    texts = []
    for part in content:
        if part.get("type") == "image":
            tags.append(IMAGE_TAG)
        elif part.get("type") == "text":
            texts.append(part.get("text", ""))
    return "".join(tags) + "".join(texts), len(tags)


def _save_images(images, out_img_dir, stem):
    """images is a list of PIL.Image (datasets decodes Image() -> PIL). Save as PNG."""
    paths = []
    for i, im in enumerate(images):
        p = os.path.join(out_img_dir, f"{stem}_img{i}.png")
        if im.mode not in ("RGB", "RGBA"):
            im = im.convert("RGB")
        im.save(p)
        paths.append(os.path.abspath(p))
    return paths


def convert_split(ds, split_name, out_dir, img_dir):
    out_path = os.path.join(out_dir, f"{split_name}.jsonl")
    n_rows = 0
    n_img_mismatch = 0
    with open(out_path, "w") as f:
        for idx, row in enumerate(ds):
            stem = f"{split_name}_{idx:06d}"
            msgs_raw = json.loads(row["messages"])
            images = row.get("images") or []
            img_paths = _save_images(images, img_dir, stem)

            out_msgs = []
            total_tags = 0
            for m in msgs_raw:
                text, n_tags = _flatten_content(m["content"])
                total_tags += n_tags
                out_msgs.append({"role": m["role"], "content": text})

            # sanity: number of <image> tags must equal number of images on disk
            if total_tags != len(img_paths):
                n_img_mismatch += 1
                # pad/truncate tags on the first user turn to keep swift happy
                # (prefer correctness: skip malformed rows)
                continue

            f.write(json.dumps({"id": row["id"], "messages": out_msgs,
                                "images": img_paths}) + "\n")
            n_rows += 1
    print(f"  {split_name}: wrote {n_rows} rows -> {out_path}"
          + (f"  (skipped {n_img_mismatch} image/tag mismatches)" if n_img_mismatch else ""))
    return n_rows


def _fara_prompt():
    fara_src = os.environ.get("FARA_SRC", os.path.join(OPSROOT, "..", "fara", "src"))
    if not os.path.isdir(fara_src):
        raise RuntimeError("Set FARA_SRC to the microsoft/fara src directory for --format fara")
    sys.path.insert(0, fara_src)
    from fara.fara_7b._prompts import get_computer_use_system_prompt
    return get_computer_use_system_prompt


def _user_text(row):
    return "".join(part.get("text", "") for part in json.loads(row["messages"])[1]["content"]
                   if part["type"] == "text")


def _assistant_text(row, image_size, model_size):
    content = json.loads(row["messages"])[-1]["content"]
    thought, marker, rest = content.partition("<tool_call>")
    if not marker:
        raise ValueError(f"missing tool call in {row['id']}")
    tool = json.loads(rest.partition("</tool_call>")[0])
    args = tool["arguments"]
    if "coordinate" in args:
        args["coordinate"] = [round(args["coordinate"][0] * model_size[0] / image_size[0]),
                              round(args["coordinate"][1] * model_size[1] / image_size[1])]
    return f"{thought.strip()}\n<tool_call>\n{json.dumps(tool)}\n</tool_call>"


def _task_with_system_urls(first_text):
    task, url_text = first_text[len("Task: "):].split("\nCurrent URL: ", 1)
    url = urlsplit(url_text.split("\n", 1)[0])
    if url.hostname != "localhost" or url.port is None:
        raise ValueError(f"unexpected sandbox URL: {url.geturl()}")
    crm_port = url.port if url.port % 2 == 0 else url.port - 1
    billing_port = crm_port + 1
    return (task + f"\n\nThe Contoso CRM system is at http://localhost:{crm_port} "
            f"and the Contoso Billing system is at http://localhost:{billing_port}. "
            "Use only these two systems; to switch systems, navigate to the appropriate URL.")


def convert_fara_split(ds, split_name, out_dir, img_dir):
    """Rebuild Fara's image-bearing turns from the verified trajectory steps."""
    from PIL import Image

    get_prompt = _fara_prompt()
    groups = defaultdict(dict)
    for row in ds:
        match = STEP_ID.fullmatch(row["id"])
        if not match:
            raise ValueError(f"invalid step id: {row['id']}")
        groups[match.group(1)][int(match.group(2))] = row

    out_path = os.path.join(out_dir, f"{split_name}.jsonl")
    count = 0
    with open(out_path, "w") as output:
        for task_id, steps in groups.items():
            if set(steps) != set(range(len(steps))):
                raise ValueError(f"incomplete trajectory: {task_id}")
            first_text = _user_text(steps[0])
            if not first_text.startswith("Task: ") or "\nCurrent URL: " not in first_text:
                raise ValueError(f"unexpected task prompt: {task_id}")
            task = _task_with_system_urls(first_text)
            for step_idx in range(len(steps)):
                row = steps[step_idx]
                screenshots = row["images"]
                first_image_step = max(0, step_idx - 2)
                if len(screenshots) != step_idx - first_image_step + 1:
                    raise ValueError(f"missing screenshot in {row['id']}")
                original_size = screenshots[-1].size
                system = get_prompt(screenshots[-1], {
                    "min_pixels": 3136, "max_pixels": 12845056,
                    "patch_size": 14, "merge_size": 2,
                }, include_input_text_key_args=True)
                model_size = system["im_size"]
                system_text = "".join(part["text"] for message in system["conversation"]
                                      for part in message["content"])
                messages = [{"role": "system", "content": system_text}]
                paths = []
                for prior_idx in range(step_idx + 1):
                    prior = steps[prior_idx]
                    if prior_idx == 0:
                        user_text = task
                    else:
                        url = _user_text(prior).split("\nCurrent URL: ", 1)[1].split("\n", 1)[0]
                        user_text = (f"Current URL: {url}\n"
                                     "Here is the next screenshot. Think about what to do next.")
                    if prior_idx >= first_image_step:
                        screenshot = screenshots[prior_idx - first_image_step]
                        path = os.path.abspath(os.path.join(
                            img_dir, f"{split_name}_{task_id}_step{prior_idx}.png"))
                        if not os.path.exists(path):
                            screenshot.resize(model_size, Image.Resampling.BICUBIC).convert("RGB").save(path)
                        paths.append(os.path.relpath(path, out_dir))
                        messages.append({"role": "user", "content": IMAGE_TAG + user_text})
                    elif prior_idx == 0:
                        messages.append({"role": "user", "content": user_text})
                    if prior_idx < step_idx:
                        messages.append({"role": "assistant", "content": _assistant_text(
                            prior, original_size, model_size)})
                messages.append({"role": "assistant", "content": _assistant_text(
                    row, original_size, model_size)})
                output.write(json.dumps({"id": row["id"], "messages": messages,
                                         "images": paths}) + "\n")
                count += 1
    print(f"  {split_name}: wrote {count} native Fara rows -> {out_path}")
    return count


def inspect_fara_samples(jsonl_path, out_dir, limit=8):
    """Save processed screenshots annotated with their supervised action targets."""
    from PIL import Image, ImageDraw

    os.makedirs(out_dir, exist_ok=True)
    saved = 0
    with open(jsonl_path) as source:
        for line in source:
            row = json.loads(line)
            content = row["messages"][-1]["content"]
            tool = json.loads(content.partition("<tool_call>")[2].partition("</tool_call>")[0])
            args = tool["arguments"]
            if "coordinate" not in args:
                continue
            image_path = os.path.join(os.path.dirname(jsonl_path), row["images"][-1])
            image = Image.open(image_path).convert("RGB")
            x, y = args["coordinate"]
            if not 0 <= x < image.width or not 0 <= y < image.height:
                raise ValueError(f"out-of-bounds action in {row['id']}: {(x, y)}")
            draw = ImageDraw.Draw(image)
            draw.ellipse((x - 10, y - 10, x + 10, y + 10), outline="red", width=3)
            draw.line((x - 15, y, x + 15, y), fill="red", width=2)
            draw.line((x, y - 15, x, y + 15), fill="red", width=2)
            image.save(os.path.join(out_dir, f"{row['id']}.png"))
            saved += 1
            if saved >= limit:
                break
    print(f"  inspect: {saved} target overlays -> {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="abishekk210698/opsworld-sft")
    ap.add_argument("--out", default="/root/opsworld-swift")
    ap.add_argument("--train-split", default="train")
    ap.add_argument("--val-split", default="validation")
    ap.add_argument("--format", choices=("swift", "fara"), default="swift")
    ap.add_argument("--inspect", type=int, default=8,
                    help="number of native Fara action overlays to save for visual inspection")
    args = ap.parse_args()

    from datasets import load_dataset

    os.makedirs(args.out, exist_ok=True)
    img_dir = os.path.join(args.out, "images")
    os.makedirs(img_dir, exist_ok=True)

    token = get_token()
    print(f"Loading {args.repo} (private) ...")
    dd = load_dataset(args.repo, token=token)
    print(dd)

    converter = convert_fara_split if args.format == "fara" else convert_split
    n_train = converter(dd[args.train_split], "train", args.out, img_dir)
    n_val = converter(dd[args.val_split], "val", args.out, img_dir)
    if args.format == "fara" and args.inspect:
        inspect_fara_samples(os.path.join(args.out, "train.jsonl"),
                             os.path.join(args.out, "inspect"), args.inspect)

    print(f"\nDone. train={n_train} val={n_val}")
    print(f"dataset dir: {args.out}")
    # show one sample for a quick eyeball
    with open(os.path.join(args.out, "train.jsonl")) as f:
        sample = json.loads(f.readline())
    print("\n--- sample (train[0]) ---")
    print("images:", sample["images"])
    for m in sample["messages"]:
        preview = m["content"][:200].replace("\n", " ")
        print(f"[{m['role']}] {preview}")


if __name__ == "__main__":
    main()
