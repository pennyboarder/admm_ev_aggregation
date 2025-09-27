"""
Direct test of baseline enforcement with specific EV configurations
"""
import numpy as np
from models import EVAgent, GlobalParams, MarketInputs
from local_solver_oop import solve_ev_with_agent_methods

def test_direct_baseline_enforcement():
    """Direct test of baseline enforcement in single EV optimization"""
    print("Testing direct baseline enforcement...")
    
    # Create single EV with flexibility
    ev_data = {
        'id': 'TestEV',
        'arrival': 0,
        'departure': 12,
        'initial_soc': 30.0,  # %
        'target_soc': 80.0,   # %
        'battery_capacity': 40.0,  # kWh
        'charge_power_max': 8.0,   # kW
        'discharge_power_max': 6.0,
        'charge_efficiency': 0.9,
        'discharge_efficiency': 0.9,
        'can_discharge': False
    }
    
    # Test parameters
    T = 12
    B = 2
    dt_hr = 5.0/60.0  # 5-minute slots
    
    # Create baseline and mask for first block only (slots 0-5)
    P_base = [4.0] * 6 + [8.0] * 6  # Different baseline per block
    baseline_mask_t = [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]  # Enforce first block only
    
    print(f"Baseline: {P_base}")
    print(f"Baseline mask: {baseline_mask_t}")
    print(f"Expected: First 6 slots should be 4.0 kW, last 6 slots flexible")
    
    # ADMM parameters (minimal)
    mi = np.zeros((T, B))
    gp = np.zeros((T, B))
    rho_r = 1.0
    rho_e = 1.0
    rho_p = 1.0
    
    # Consensus variables that encourage different charging pattern
    c_r = [[0.0, 0.0] for _ in range(T)]
    c_e = [[0.0, 0.0] for _ in range(T)]
    c_p = [10.0, 2.0, 12.0, 1.0, 8.0, 3.0, 15.0, 5.0, 10.0, 2.0, 8.0, 4.0]  # Varies from baseline
    
    print(f"Consensus c_p: {c_p}")
    
    # Test without baseline enforcement
    print("\\n=== Without baseline enforcement ===")
    result_no_enforce = solve_ev_with_agent_methods(
        ev_data, mi, gp, c_r, c_e, c_p, T, B, dt_hr,
        rho_r, rho_e, rho_p, baseline_mask_t,
        P_base=P_base, enforce_baseline_for_reserve=False
    )
    
    if result_no_enforce and 'p_ch' in result_no_enforce:
        charging = result_no_enforce['p_ch']
        print(f"Charging pattern: {[f'{x:.1f}' for x in charging]}")
        
        deviations_first_block = [abs(charging[t] - P_base[t]) for t in range(6)]
        deviations_second_block = [abs(charging[t] - P_base[t]) for t in range(6, 12)]
        
        print(f"First block deviations: {[f'{d:.1f}' for d in deviations_first_block]}")
        print(f"Second block deviations: {[f'{d:.1f}' for d in deviations_second_block]}")
        print(f"Max first block deviation: {max(deviations_first_block):.1f}")
        print(f"Max second block deviation: {max(deviations_second_block):.1f}")
    
    # Test with baseline enforcement
    print("\\n=== With baseline enforcement ===")
    result_with_enforce = solve_ev_with_agent_methods(
        ev_data, mi, gp, c_r, c_e, c_p, T, B, dt_hr,
        rho_r, rho_e, rho_p, baseline_mask_t,
        P_base=P_base, enforce_baseline_for_reserve=True
    )
    
    if result_with_enforce and 'p_ch' in result_with_enforce:
        charging = result_with_enforce['p_ch']
        print(f"Charging pattern: {[f'{x:.1f}' for x in charging]}")
        
        deviations_first_block = [abs(charging[t] - P_base[t]) for t in range(6)]
        deviations_second_block = [abs(charging[t] - P_base[t]) for t in range(6, 12)]
        
        print(f"First block deviations: {[f'{d:.1f}' for d in deviations_first_block]}")
        print(f"Second block deviations: {[f'{d:.1f}' for d in deviations_second_block]}")
        print(f"Max first block deviation: {max(deviations_first_block):.1f}")
        print(f"Max second block deviation: {max(deviations_second_block):.1f}")
        
        # Check if enforcement worked
        first_block_enforced = max(deviations_first_block) < 0.01
        second_block_flexible = any(d > 0.01 for d in deviations_second_block)
        
        if first_block_enforced:
            print("✓ First block baseline enforcement working")
        else:
            print("✗ First block baseline enforcement failed")
            
        if second_block_flexible:
            print("✓ Second block remains flexible")
        else:
            print("? Second block may be over-constrained")
    
    # Test with all blocks enforced
    print("\\n=== With all blocks enforced ===")
    baseline_mask_all = [1] * 12  # Enforce all slots
    
    result_all_enforce = solve_ev_with_agent_methods(
        ev_data, mi, gp, c_r, c_e, c_p, T, B, dt_hr,
        rho_r, rho_e, rho_p, baseline_mask_all,
        P_base=P_base, enforce_baseline_for_reserve=True
    )
    
    if result_all_enforce and 'p_ch' in result_all_enforce:
        charging = result_all_enforce['p_ch']
        print(f"Charging pattern: {[f'{x:.1f}' for x in charging]}")
        
        all_deviations = [abs(charging[t] - P_base[t]) for t in range(12)]
        print(f"All deviations: {[f'{d:.1f}' for d in all_deviations]}")
        print(f"Max deviation: {max(all_deviations):.1f}")
        
        if max(all_deviations) < 0.01:
            print("✓ All slots baseline enforcement working")
        else:
            print("✗ Some slots not enforced properly")

if __name__ == "__main__":
    test_direct_baseline_enforcement()
