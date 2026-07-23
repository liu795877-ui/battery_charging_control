from pathlib import Path
import json
import numpy as np
import pandas as pd
import pytest
from battery_fast_charge.phase2rd_runner import _slew_path, load_phase_two_rd_config
from battery_fast_charge.identification import build_ocv_function
from battery_fast_charge.mpc import ConstrainedMPC, ReducedBatteryModel
from battery_fast_charge.phase3_config import load_phase_three_config

ROOT=Path(__file__).resolve().parents[1]

def _model():
    config=load_phase_three_config(ROOT/'configs'/'phase3.yaml')
    parameters=json.loads((ROOT/config.artifacts.identified_parameters).read_text(encoding='utf-8'))
    ocv=build_ocv_function(pd.read_csv(ROOT/config.artifacts.ocv_curve))
    return ReducedBatteryModel(config,ocv,parameters),config

def test_phase2rd_contract():
    c=load_phase_two_rd_config(ROOT/'configs'/'phase2rd_final_pure_dnn_discrimination.yaml')
    assert c.state_count==100 and c.warm_start_count==15 and c.neighbor_counts==(5,10,25,50)

def test_slew_path_is_bounded():
    values=_slew_path(5.0,np.array([10.0,0.0,9.0]),10.0,2.0)
    assert np.all((values>=0)&(values<=10)); assert np.max(np.abs(np.diff(np.r_[5.0,values])))<=2.0

def test_explicit_warm_start_validation():
    model,config=_model(); controller=ConstrainedMPC(model,config); n=controller.number_of_blocks
    controller.set_initial_block_currents_a(np.ones(n)); assert np.array_equal(controller.last_optimal_block_currents_a,np.ones(n))
    with pytest.raises(ValueError): controller.set_initial_block_currents_a(np.ones(n+1))
