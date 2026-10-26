"""Export the finetuned MiniLM emotion classifier to ONNX.

Examples (run from this directory):
    python export_onnx.py
    python export_onnx.py --checkpoint archive/emotion_head_best_4_new_loss.pt
    python export_onnx.py --output saved/emotion_classifier.onnx --opset 17

The exported graph accepts int64 ``input_ids`` and ``attention_mask`` tensors
with dynamic batch and sequence dimensions, and returns raw emotion logits.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import _1_Emotion_Predictor.Finetuned_MiniLM.hparam as hp
from _1_Emotion_Predictor.Finetuned_MiniLM.model import EmotionClassifierHybrid


SCRIPT_DIR = Path(__file__).resolve().parent


def resolve_checkpoint(checkpoint: str | None) -> Path:
    """Resolve the requested checkpoint or select the newest available one."""
    if checkpoint:
        path = Path(checkpoint).expanduser()
        if not path.is_absolute() and not path.is_file():
            path = SCRIPT_DIR / path
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return path

    candidates = [
        path
        for folder in (SCRIPT_DIR, SCRIPT_DIR / "saved", SCRIPT_DIR / "archive")
        for path in folder.glob("emotion_head_best*.pt")
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No emotion_head_best*.pt checkpoint found under {SCRIPT_DIR}; "
            "specify one with --checkpoint."
        )
    return max(candidates, key=lambda path: path.stat().st_mtime)


def export(checkpoint: Path, output: Path, opset: int) -> None:
    """Load a PyTorch checkpoint and export a checked, dynamic ONNX graph."""
    try:
        import onnx
    except ImportError as exc:
        raise RuntimeError(
            "The ONNX package is required for export. Install it in the active "
            "environment with: pip install onnx"
        ) from exc

    model = EmotionClassifierHybrid(
        n_emotions=hp.NUM_EMOTIONS,
        input_dim=hp.input_dim,
        dropout=0.25,
        minilm=hp.MODEL,
        n_keep=0,
    )
    state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    # Synthetic valid token IDs avoid needing to load a tokenizer just to trace
    # the graph. Batch and sequence lengths are made dynamic in the export.
    vocab_size = model.encoder.config.vocab_size
    dummy_ids = torch.randint(0, vocab_size, (2, 8), dtype=torch.long)
    dummy_mask = torch.ones_like(dummy_ids)

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output.with_name(f"{output.stem}.tmp{output.suffix}")
    try:
        with torch.inference_mode():
            torch.onnx.export(
                model,
                (dummy_ids, dummy_mask),
                str(temporary_output),
                input_names=["input_ids", "attention_mask"],
                output_names=["logits"],
                dynamic_axes={
                    "input_ids": {0: "batch_size", 1: "sequence_length"},
                    "attention_mask": {0: "batch_size", 1: "sequence_length"},
                    "logits": {0: "batch_size"},
                },
                opset_version=opset,
                do_constant_folding=True,
                dynamo=False,
            )

        exported_model = onnx.load(str(temporary_output))
        onnx.checker.check_model(exported_model)
        temporary_output.replace(output)
    except Exception:
        temporary_output.unlink(missing_ok=True)
        raise

    metadata_path = output.with_suffix(".json")
    metadata = {
        "checkpoint": str(checkpoint),
        "base_model": hp.MODEL,
        "max_length": hp.MAX_LEN,
        "input_names": ["input_ids", "attention_mask"],
        "output_name": "logits",
        "output_is_logits": True,
        "emotions": hp.EMOTIONS,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Exported and checked ONNX model: {output}")
    print(f"Emotion labels and input metadata: {metadata_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        help="PyTorch checkpoint (default: newest emotion_head_best*.pt in this folder, saved/, or archive/).",
    )
    parser.add_argument(
        "--output",
        default=str(SCRIPT_DIR / "saved" / "emotion_classifier.onnx"),
        help="Output ONNX path (default: saved/emotion_classifier.onnx).",
    )
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version (default: 17).")
    args = parser.parse_args()
    if args.opset < 11:
        parser.error("--opset must be 11 or newer")

    checkpoint = resolve_checkpoint(args.checkpoint)
    output = Path(args.output).expanduser()
    if not output.is_absolute():
        output = Path.cwd() / output
    print(f"Loading checkpoint: {checkpoint}")
    print(f"Exporting model to: {output}")
    export(checkpoint, output, args.opset)


if __name__ == "__main__":
    main()