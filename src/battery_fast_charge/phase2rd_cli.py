"""运行 Phase 2R-D pure DNN 最终判别实验。"""
from argparse import ArgumentParser
from pathlib import Path
from .phase2rd_runner import load_phase_two_rd_config, run_phase_two_rd

def main():
    parser=ArgumentParser(description=__doc__); parser.add_argument('--config',type=Path,default=Path('configs/phase2rd_final_pure_dnn_discrimination.yaml')); parser.add_argument('--project-root',type=Path,default=Path('.')); args=parser.parse_args()
    result=run_phase_two_rd(load_phase_two_rd_config(args.config),args.project_root)
    if result['status']!='completed': raise SystemExit(2)
if __name__=='__main__': main()

