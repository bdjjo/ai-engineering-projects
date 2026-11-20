import argparse

from distributed_gpt.data import prepare_synthetic
from distributed_gpt.preparation import prepare_text


def main():
    parser = argparse.ArgumentParser(
        description="Prepare synthetic checks or streamed GPT-2 tokens"
    )
    parser.add_argument(
        "--source", choices=["synthetic", "wikitext", "openwebtext"], default="synthetic"
    )
    parser.add_argument("--out", default="data/synthetic")
    parser.add_argument("--train-tokens", type=int)
    parser.add_argument("--val-tokens", type=int)
    parser.add_argument("--dataset-revision")
    parser.add_argument("--tokenizer-revision")
    args = parser.parse_args()
    if args.source == "synthetic":
        prepare_synthetic(args.out, args.train_tokens or 16385, args.val_tokens or 4097)
    else:
        prepare_text(
            args.out,
            args.source,
            args.train_tokens or 131072,
            args.val_tokens or 16384,
            args.dataset_revision,
            args.tokenizer_revision,
        )


if __name__ == "__main__":
    main()
