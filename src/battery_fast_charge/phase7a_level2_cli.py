"""运行 Phase 7A Level 2 2RC 三状态验证。"""
from __future__ import annotations
import argparse
from pathlib import Path
from .phase7a_level2_config import load_phase7a_level2_config
from .phase7a_level2_runner import run_phase7a_level2

def main()->None:
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--config",type=Path,default=Path("configs/phase7a_level2_2rc.yaml")); parser.add_argument("--project-root",type=Path,default=Path(".")); parser.add_argument("--resume",action="store_true"); args=parser.parse_args()
    result=run_phase7a_level2(load_phase7a_level2_config(args.config),args.project_root,args.resume); print(f"Phase 7A Level 2 success: {result['success']}")

if __name__=="__main__": main()

