"""Freeze the official InstructIR language pipeline into task vectors."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import torch
from torch.nn import functional as F
from transformers import AutoModel, AutoTokenizer


PROMPTS = {
    "denoise": "I need this image denoised ASAP.",
    "dehaze": "I need to remove the haziness from this image.",
    "derain": "Clear the rain from my picture.",
    "deblur": "I took this photo while I was running, can you stabilize the image? it is too blurry",
    "lowlight": "my image is too dark, I cannot see anything, can you fix it?",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--text-model", type=Path, required=True)
    parser.add_argument("--head-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    spec = importlib.util.spec_from_file_location("instructir_text_models", args.source / "text" / "models.py")
    if spec is None or spec.loader is None:
        raise ImportError("cannot import official InstructIR language head")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    head = module.LMHead(embedding_dim=384, hidden_dim=256, num_classes=7).eval()
    head.load_state_dict(torch.load(args.head_checkpoint, map_location="cpu", weights_only=False), strict=True)
    tokenizer = AutoTokenizer.from_pretrained(args.text_model, local_files_only=True)
    encoder = AutoModel.from_pretrained(args.text_model, local_files_only=True).eval()
    vectors = {}
    with torch.no_grad():
        for task, prompt in PROMPTS.items():
            tokens = tokenizer([prompt], padding=True, truncation=True, return_tensors="pt")
            hidden = encoder(**tokens)[0]
            mask = tokens["attention_mask"].unsqueeze(-1).expand(hidden.size()).float()
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1e-9)
            vector, _ = head(F.normalize(pooled, p=2, dim=1))
            vectors[task] = vector.squeeze(0).contiguous()
            if not torch.isfinite(vectors[task]).all():
                raise RuntimeError(f"non-finite InstructIR embedding: {task}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(vectors, args.output)
    print(f"saved official InstructIR task embeddings: {args.output}")


if __name__ == "__main__":
    main()
