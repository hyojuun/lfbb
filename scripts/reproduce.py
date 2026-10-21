"""Runs the commands of a reproduction config (configs/reproduce/*.json) in order.

A config has "vars" and "steps". Each step is one command line with {VAR} placeholders, filled from "vars" (override them with
--set VAR=VALUE). A step with "for" runs once per value, its loop variables filled like the others. A step whose "creates" pattern
already matches a file is skipped, so an interrupted run resumes where it stopped. Commands run in bash from the repository root.

Configs:
  configs/reproduce/components.json   learn the partition and the foresight predictor (Appendices C and D)
  configs/reproduce/table1.json       comparison with prior watermarks (Table 1)
  configs/reproduce/table2.json       crossed ablation of the two components (Table 2)

Example:
  python scripts/reproduce.py configs/reproduce/table1.json --dry-run
  python scripts/reproduce.py configs/reproduce/table1.json --set PY_EVAL=/path/to/eval-env/bin/python
"""
import argparse
import glob
import itertools
import json
import subprocess
import sys


def expand(step, values):
    """(name, command, creates) of every run of a step, with its placeholders filled."""
    loops = step.get("for", {})
    keys = list(loops)
    for combo in itertools.product(*[loops[k] for k in keys]):
        v = dict(values)
        v.update({k: str(x) for k, x in zip(keys, combo)})
        yield step["name"].format(**v), step["cmd"].format(**v), step.get("creates", "").format(**v)


def main():
    ap = argparse.ArgumentParser(description="Run the commands of a reproduction config in order.")
    ap.add_argument("config", help="configs/reproduce/<name>.json")
    ap.add_argument("--set", action="append", default=[], metavar="VAR=VALUE", help="override a variable of the config")
    ap.add_argument("--dry-run", action="store_true", help="print the commands without running them")
    ap.add_argument("--only", help="comma-separated step names to run (a looped step by its own name or its expanded name)")
    ap.add_argument("--force", action="store_true", help="also run steps whose outputs exist")
    args = ap.parse_args()

    cfg = json.load(open(args.config))
    values = dict(cfg.get("vars", {}))
    for kv in args.set:
        k, v = kv.split("=", 1)
        values[k] = v
    only = set(args.only.split(",")) if args.only else None
    print(cfg.get("description", args.config), flush=True)
    for step in cfg["steps"]:
        for name, cmd, creates in expand(step, values):
            if only and name not in only and step["name"] not in only:
                continue
            if creates and not args.force and glob.glob(creates):
                print(f"[skip] {name}: {creates} exists", flush=True)
                continue
            print(f"[run] {name}\n  {cmd}", flush=True)
            if args.dry_run:
                continue
            code = subprocess.run(cmd, shell=True, executable="/bin/bash").returncode
            if code:
                sys.exit(f"step {name} failed with exit code {code}")


if __name__ == "__main__":
    main()
