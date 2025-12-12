# run_experiments.py

import argparse
import yaml
from pathlib import Path

from agents import load_agent
from models import create_model
from datasets import create_stream, create_test_stream
from utils.cl_metrics import acc_from_matrix, bwt_from_matrix

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
    
    agent_cfg = cfg["agent"]
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    
    model = create_model(model_cfg)
    train_stream = create_stream(data_cfg)
    test_stream = None
    if "test_tasks" in data_cfg:
        test_stream = list(create_test_stream(data_cfg)) # list로 고정
    
    agent = load_agent(agent_cfg, model=model, stream=train_stream, test_stream=test_stream)
    report = agent.run()
    
    if "acc_matrix" in report:
        acc = acc_from_matrix(report["acc_matrix"])
        bwt = bwt_from_matrix(
            report["acc_matrix"],
            task_order=[t["name"] for t in data_cfg["tasks"]],
        )
        print(f"[CL Metrics] ACC = {acc:.4f}, BWT = {bwt:.4f}")
    
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