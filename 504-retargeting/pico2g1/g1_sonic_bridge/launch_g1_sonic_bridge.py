#!/usr/bin/env python3
"""Convert recorded or live device body/glove poses for SONIC."""
from __future__ import annotations

import argparse
from pathlib import Path

from bridge_modes import live, replay




def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    replay_parser = modes.add_parser("replay")
    replay_parser.add_argument("input", type=Path)
    live_parser = modes.add_parser("live")
    live_parser.add_argument("host", help="RK3588 IP address, for example 192.168.2.111")
    live_parser.add_argument("--token", default="rynn_dev")
    args = parser.parse_args()
    replay(args.input) if args.mode == "replay" else live(args.host, args.token)


if __name__ == "__main__":
    main()
