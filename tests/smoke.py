"""Smoke test: does the checkpoint actually predict sensible anchors?"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lb.extract import parse_file
from lb.model import LinkPredictor

TEXT = """Why take out pet insurance for senior cats and dogs? Cats and dogs often require more
veterinary care as they grow older; pet insurance can help by covering a portion of your
eligible vet bills. Other benefits of pet insurance for senior pets may include cover for
accidental injuries, including car accidents, poisoning, snake and insect bites and wounds.
Comprehensive pet insurance is usually available for pets up until they turn nine. Optional
routine care is also available for older pets up to nine years of age as an add-on for new
policyholders. Some insurers offer accident-only cover for senior pets before or up to 16
years of age."""


def main() -> None:
    import time

    t0 = time.time()
    pred = LinkPredictor()
    load_s = time.time() - t0

    t1 = time.time()
    spans = pred.predict(TEXT)
    infer_s = time.time() - t1

    print(f"device: {pred.device}")
    print(f"load:   {load_s:.2f}s")
    print(f"infer:  {infer_s:.2f}s for {len(TEXT.split())} words")
    if pred.device.type == "cuda":
        import torch

        print(f"vram:   {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB peak")
    print(f"{len(spans)} spans\n")
    for s in spans:
        print(f"  {s.score:.3f}  {s.text!r}")
    print("\n" + pred.annotate(TEXT))


if __name__ == "__main__":
    main()
