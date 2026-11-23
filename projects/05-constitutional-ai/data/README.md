# Prompt provenance

`prompts.json` contains 72 original, AI-assisted teaching scenarios written for
this project from the 15 categories in the supplied brief. They are not drawn
from HH-RLHF and are not an established safety benchmark. Code and prompts are
released under the repository's MIT license.

The fixed split has 36 training, 18 validation, and 18 test prompts. Each covers
all 15 principles. There are 48 adversarial requests and 24 benign controls across
the full set. Sensitive-domain benign controls include authorized security work,
privacy protection, safeguarding, and honest communication.

Categories describe the intended behavior being tested, not a universal legal or
ethical classification. The prompts are English-only, short, single-turn scenarios.
They do not represent the languages, cultures, contexts, or adversarial strategies
that a deployed assistant would face.

Responses are generated only when an experiment runs. The synthetic smoke test
uses separate toy prompts and scripted answers, clearly labeled in its report.
The pretrained models may already have encountered similar requests during training.
Held-out here means held out from this project's fitting and checkpoint selection.
