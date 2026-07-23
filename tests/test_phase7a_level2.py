from pathlib import Path
import json
import numpy as np

from battery_fast_charge.phase7a_level1_config import load_phase7a_level1_config
from battery_fast_charge.phase7a_level2_config import load_phase7a_level2_config
from battery_fast_charge.phase7a_level2_model import Level2MPC,Level2Model,Level2State
from battery_fast_charge.phase7a_level2_runner import design_initial_states

ROOT=Path(__file__).parents[1]; CONFIG=ROOT/"configs"/"phase7a_level2_2rc.yaml"

def test_level2_contract_adds_only_second_polarization_state():
    config=load_phase7a_level2_config(CONFIG); inherited=load_phase7a_level1_config(ROOT/config.source_level1_config)
    assert inherited.mpc.current_bounds_a==(0.0,10.0) and inherited.mpc.terminal_voltage_max_v==4.2
    assert inherited.network.hidden_layer_sizes==(32,32,16) and inherited.network.initialization_seeds==(22,42,73,101,137)
    raw=CONFIG.read_text(encoding="utf-8").lower(); assert "temperature" not in raw and "dfn" not in raw and "slew" not in raw

def test_level2_parameters_match_project_identification():
    config=load_phase7a_level2_config(CONFIG); identified=json.loads((ROOT/config.source_identified_parameters).read_text(encoding="utf-8"))["electrical_2rc"]
    for name in ("r0_ohm","r1_ohm","tau1_s","r2_ohm","tau2_s"): assert getattr(config.model,name)==identified[name]

def test_2rc_discrete_equations_and_voltage_sign():
    config=load_phase7a_level2_config(CONFIG); inherited=load_phase7a_level1_config(ROOT/config.source_level1_config); model=Level2Model(config,inherited,ROOT); state=Level2State(.5,.02,.03); nxt=model.step(state,5.0)
    assert np.isclose(nxt.soc,.5+25/(3600*5)); assert np.isclose(nxt.polarization_1_v,np.exp(-5/config.model.tau1_s)*.02+config.model.r1_ohm*(1-np.exp(-5/config.model.tau1_s))*5)
    assert np.isclose(nxt.polarization_2_v,np.exp(-5/config.model.tau2_s)*.03+config.model.r2_ohm*(1-np.exp(-5/config.model.tau2_s))*5); assert model.terminal_voltage(state,5)>model.terminal_voltage(state,0)

def test_level2_design_freezes_two_test_domains_and_is_initially_feasible():
    config=load_phase7a_level2_config(CONFIG); inherited=load_phase7a_level1_config(ROOT/config.source_level1_config); model=Level2Model(config,inherited,ROOT)
    global_design=design_initial_states(config,model,config.data.global_domain,"global","test",0); terminal=design_initial_states(config,model,config.data.terminal_domain,"terminal","terminal_test",1000)
    assert global_design.split.value_counts().to_dict()=={"train":168,"validation":36,"test":36}; assert terminal.split.value_counts().to_dict()=={"train":120,"validation":20,"terminal_test":20}
    for frame in (global_design,terminal):
        assert all(model.terminal_voltage(Level2State(r.initial_soc,r.initial_polarization_1_v,r.initial_polarization_2_v),0)<=4.195+1e-12 for r in frame.itertuples())

def test_level2_mpc_enforces_current_and_voltage_only():
    config=load_phase7a_level2_config(CONFIG); inherited=load_phase7a_level1_config(ROOT/config.source_level1_config); model=Level2Model(config,inherited,ROOT); result=Level2MPC(model).solve(Level2State(.70,.03,.04))
    assert result.optimizer_success and result.prediction_feasible and not result.used_fallback; assert result.maximum_voltage_v<=4.2+inherited.mpc.constraint_tolerance
