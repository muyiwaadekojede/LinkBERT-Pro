"""Diagnose the LinkBERT checkpoint: head shape, label mapping, raw probabilities."""

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lb.model import CACHE_DIR, MODEL_NAME

tok = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
model, info = AutoModelForTokenClassification.from_pretrained(
    MODEL_NAME, cache_dir=CACHE_DIR, output_loading_info=True
)
model.eval()

print("config architectures:", model.config.architectures)
print("num_labels:", model.config.num_labels)
print("id2label:", model.config.id2label)
print("missing:", info.get("missing_keys"))
print("unexpected:", sorted(info.get("unexpected_keys") or [])[:10], "...")
print("classifier:", model.classifier.weight.shape, model.classifier.weight.mean().item(),
      model.classifier.weight.std().item())
print("bias:", model.classifier.bias.detach().tolist())

text = ("Why take out pet insurance for senior cats and dogs? Cats and dogs often require more "
        "veterinary care as they grow older; pet insurance can help by covering a portion of your "
        "eligible vet bills. Comprehensive pet insurance is usually available for pets up until they turn nine.")

enc = tok(text, return_tensors="pt", return_offsets_mapping=True)
with torch.no_grad():
    logits = model(**{k: v for k, v in enc.items() if k != "offset_mapping"}).logits[0]
probs = torch.softmax(logits, dim=-1)

offsets = enc["offset_mapping"][0].tolist()
for (s, e), p in zip(offsets, probs.tolist()):
    if s == e:
        continue
    frag = text[s:e]
    if p[1] > 0.3 or frag.lower() in {"why", "to", "pet", "insurance"}:
        print(f"  p1={p[1]:.3f} p0={p[0]:.3f}  {frag!r}")
