# run_experiments.py

import argparse
import yaml
from pathlib import Path

from methods import load_method
from models import load_model_pool
from task_pool import load_task_pool

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to experiment config yaml",
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save report (json/yaml)",
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    cfg_path = Path(args.config)
    
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    
    method_cfg = cfg["method"]
    modelpool_cfg = cfg["modelpool"]
    taskpool_cfg = cfg["taskpool"] 
    
    method = load_method(method_cfg)
    model_pool = load_model_pool(modelpool_cfg)
    task_pool = load_task_pool(taskpool_cfg)
    
    report = method.run(model_pool, task_pool)
    
    if args.output is not None:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        import json
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2)
    else:
        print("[main] report:", report)

if __name__ == "__main__":
    main()