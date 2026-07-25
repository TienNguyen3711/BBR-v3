#!/usr/bin/env python3
"""CLI: run offline A2C training (qbbr.train.loop) against qbbr.env.fluid_env.FluidSimEnv.

Not yet runnable -- depends on qbbr.env.fluid_env, qbbr.agents, and
qbbr.train.loop, none of which are implemented yet.

Usage (once implemented):
    .venv/bin/python qbbr/scripts/train.py --config qbbr/configs/base.yaml
"""
from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="path to a qbbr/configs/*.yaml file")
    parser.parse_args()
    raise NotImplementedError("Training loop not yet implemented; see qbbr.train.loop.")


if __name__ == "__main__":
    main()
