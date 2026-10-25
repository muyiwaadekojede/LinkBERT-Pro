"""Run the exact example from the upstream README and compare."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lb.model import LinkPredictor

TEXT = (
    "Why take out pet insurance for senior cats and dogs? Cats and dogs often require more veterinary "
    "care as they grow older; pet insurance can help by covering a portion of your eligible vet bills. "
    "Other benefits of pet insurance for senior pets may include: Cover for accidental injuries, including "
    "car accidents, poisoning, snake/insect bites and wounds Policies for older animals with limited benefits "
    "for specified illnesses (age limits apply) Being able to choose your own vet or clinic for treatment A "
    "financial safety net and peace of mind if your senior pet needs emergency care Coosing pet insurance for "
    "senior pets If you're looking to buy or switch to a new pet insurance policy for your older furry "
    "companion, please consider the following: Basic policies for older pets often have a lower benefit "
    "percentage (i.e. reimbursement rate), so you'll have to pay a larger portion of eligible vet bills. For "
    "example, you may only be able to claim up to 60% back for eligible treatments, meaning you'd have to foot "
    "the remaining 40% of the bill. Comprehensive pet insurance is usually available for pets up until they "
    "turn nine. If you take out a suitable policy before that, your pooch or feline can be protected for life "
    "- if you maintain continuous cover past this age. Optional routine care is also available for older pets "
    "up to nine years of age as an add-on for new policyholders. Some insurers offer accident-only cover for "
    "senior pets before or up to 16 years of age. A basic level of cover for injuries will be cheaper than more "
    "comprehensive accident and illness cover."
)

pred = LinkPredictor()
spans = pred.predict(TEXT, high=0.5, low=0.25)
print(f"{len(spans)} spans at 0.5\n")
for s in sorted(spans, key=lambda x: -x.score):
    print(f"  {s.score:.3f}  {s.text!r}")
