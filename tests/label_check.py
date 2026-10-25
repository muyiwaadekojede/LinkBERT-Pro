"""Which class means 'link'? Print both columns for known anchor phrases."""

import sys
from pathlib import Path

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lb.model import CACHE_DIR, MODEL_NAME

tok = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR).eval()

TEXTS = [
    "Comprehensive pet insurance is usually available for pets up until they turn nine.",
    "Some insurers offer accident-only cover for senior pets before or up to 16 years of age.",
    "Optional routine care is also available for older pets up to nine years of age.",
    "Cats and dogs often require more veterinary care as they grow older; pet insurance can help.",
    "The quick brown fox jumps over the lazy dog on a sunny Tuesday afternoon.",
]

for text in TEXTS:
    enc = tok(text, return_tensors="pt", return_offsets_mapping=True)
    with torch.no_grad():
        logits = model(**{k: v for k, v in enc.items() if k != "offset_mapping"}).logits[0]
    probs = torch.softmax(logits, dim=-1)
    print(f"\n{text}")
    row = []
    for (s, e), p in zip(enc["offset_mapping"][0].tolist(), probs.tolist()):
        if s == e:
            continue
        row.append((text[s:e], p[0], p[1]))
    # print only tokens where either class is confident
    for frag, p0, p1 in row:
        flag = "  <-- p0" if p0 > 0.6 else ("  <-- p1" if p1 > 0.6 else "")
        print(f"   {frag:<18} p0={p0:.3f} p1={p1:.3f}{flag}")
