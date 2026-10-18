"""Publish the corrected step-100 and step-236 PEFT adapters as private HF models."""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.hf_auth import get_token


BASE_MODEL = "microsoft/Fara-7B"
CHECKPOINTS = {100: "checkpoint-100", 236: "checkpoint-236"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", type=Path, default=Path("runs/sft_native_checkpoints"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    token = get_token()
    if not token:
        raise SystemExit("No HF token found (see common/hf_auth.py).")
    api = HfApi(token=token)
    owner = api.whoami()["name"]

    for step, directory in CHECKPOINTS.items():
        checkpoint = args.checkpoints / directory
        state = json.loads((checkpoint / "trainer_state.json").read_text())
        if state["global_step"] != step:
            raise ValueError(f"{checkpoint}: expected step {step}, got {state['global_step']}")
        config = json.loads((checkpoint / "adapter_config.json").read_text())
        config["base_model_name_or_path"] = BASE_MODEL
        weights = checkpoint / "adapter_model.safetensors"
        if not weights.is_file():
            raise FileNotFoundError(weights)

        repo_id = f"{owner}/opsworld-fara-native-step{step}"
        card = (f"---\nbase_model: {BASE_MODEL}\nlibrary_name: peft\n"
                "datasets:\n- abishekk210698/opsworld-sft\n---\n\n"
                f"# OpsWorld Fara-7B, corrected SFT step {step}\n\n"
                f"LoRA adapter at optimizer step {step} of the Fara-native OpsWorld SFT run. "
                "Apply to `microsoft/Fara-7B` with PEFT; this is not a standalone merged model.\n\n"
                "Held-out OpsWorld evaluation: 120/120 tasks (one rollout per task, "
                "seeds 9000-9119; strict multi-DB verifier). "
                "This result does not establish robustness across repeated rollouts or new task templates.\n")
        print(f"{repo_id}: {weights} ({weights.stat().st_size} bytes), private=True", flush=True)
        if args.dry_run:
            continue
        api.create_repo(repo_id, repo_type="model", private=True, exist_ok=False)
        api.create_commit(
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"Upload corrected OpsWorld SFT adapter at step {step}",
            operations=[
                CommitOperationAdd(path_in_repo="adapter_model.safetensors", path_or_fileobj=weights),
                CommitOperationAdd(path_in_repo="adapter_config.json", path_or_fileobj=io.BytesIO(
                    (json.dumps(config, indent=2) + "\n").encode())),
                CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=io.BytesIO(card.encode())),
            ],
        )
        print(f"Uploaded: https://huggingface.co/{repo_id}", flush=True)


if __name__ == "__main__":
    main()