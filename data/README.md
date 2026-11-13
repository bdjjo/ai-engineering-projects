# Data provenance

Tiny Shakespeare is a small subset of Shakespeare's works distributed with
[Andrej Karpathy's char-rnn](https://github.com/karpathy/char-rnn/tree/master/data/tinyshakespeare).
It is **not** the complete works of Shakespeare.

Run `python scripts/download_data.py` and `python scripts/prepare_data.py --tokenizer char`.
The downloader verifies a fixed SHA-256 before writing the raw file. Raw text,
token arrays, and model weights are excluded from Git.

Preparation splits the original text contiguously into 90% train, 5% validation,
and 5% test **before fitting the tokenizer**. Each split is encoded separately;
training windows cannot cross a boundary. Validation chooses the checkpoint.
The test partition is reserved for the final evaluation.

The manifest records the source hash, split offsets, tokenizer hash, token
counts, and array hashes so mismatched data is rejected when resuming.
