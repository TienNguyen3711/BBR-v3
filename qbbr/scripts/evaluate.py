from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=["a", "b"], required=True)
    parser.add_argument("--config", required=True, help="path to a qbbr/configs/eval_scenario*.yaml file")
    args = parser.parse_args()

    if args.scenario == "a":
        raise NotImplementedError("Scenario A runner not yet implemented; see qbbr.eval.scenario_a.")
    raise NotImplementedError("Scenario B runner not yet implemented; see qbbr.eval.scenario_b.")


if __name__ == "__main__":
    main()
