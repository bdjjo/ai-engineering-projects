"""Publish an already reviewed Space package using a local Hugging Face login."""

import argparse
from pathlib import Path

from huggingface_hub import HfApi, get_token


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="bdjjo/gpt2-feature-explorer")
    parser.add_argument("--folder", default="space-build")
    args = parser.parse_args()
    required = [
        "app.py",
        "browser.py",
        "feature_index.json",
        "requirements.txt",
        "README.md",
        "LICENSE",
    ]
    folder = Path(args.folder)
    if not all((folder / name).is_file() for name in required):
        raise SystemExit("Package the Space first with scripts/package_space.py")
    if not get_token():
        raise SystemExit(
            "Not signed in. Run `hf auth login` locally with a write token; "
            "never paste a token into chat or commit it."
        )
    api = HfApi()
    owner = args.repo.split("/")[0]
    if api.whoami()["name"] != owner:
        raise SystemExit("Authenticated account does not match the requested Space owner")
    api.create_repo(repo_id=args.repo, repo_type="space", space_sdk="gradio", exist_ok=True)
    api.upload_folder(
        repo_id=args.repo,
        repo_type="space",
        folder_path=folder,
        allow_patterns=required,
        commit_message="Publish measured GPT-2 SAE feature explorer",
    )
    print(f"https://huggingface.co/spaces/{args.repo}")


if __name__ == "__main__":
    main()
