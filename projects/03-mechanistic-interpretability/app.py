"""Launch the feature browser locally; sharing is opt-in."""

import argparse

from mech_interp.feature_browser import build_browser


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", default="reports/cpu-pilot/feature_index.json")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()
    build_browser(args.index).launch(
        server_name="127.0.0.1", server_port=args.port, share=args.share
    )


if __name__ == "__main__":
    main()
