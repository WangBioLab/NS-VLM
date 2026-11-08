from __future__ import annotations

import argparse
import json
from typing import Sequence

from ns_vlm_i import NSVLMI


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run NS-VLM-I with one prompt and one or more images."
    )
    parser.add_argument("--prompt", required=True, help="Question or instruction for the image study.")
    parser.add_argument(
        "--image",
        action="append",
        required=True,
        help="Image path. Repeat this argument for a multi-image study.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    engine = NSVLMI.from_environment()
    result = engine(arguments.prompt, arguments.image)
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
