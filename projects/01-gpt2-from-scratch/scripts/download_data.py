"""Download the original Tiny Shakespeare text and verify its exact bytes."""

import argparse
import hashlib
import urllib.request
from pathlib import Path

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
SHA256 = "86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed"


def download(output: str | Path) -> Path:
    output = Path(output)
    if output.exists():
        content = output.read_bytes()
    else:
        with urllib.request.urlopen(URL, timeout=60) as response:
            content = response.read()
    actual = hashlib.sha256(content).hexdigest()
    if actual != SHA256:
        raise ValueError(f"dataset checksum mismatch: {actual}; refusing to use altered data")
    output.parent.mkdir(parents=True, exist_ok=True)
    if not output.exists():
        output.write_bytes(content)
    print(f"Verified {len(content):,} bytes | SHA-256 {actual}")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/raw/shakespeare.txt")
    download(parser.parse_args().out)
