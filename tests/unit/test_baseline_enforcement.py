"""
Test baseline enforcement for reserve provision
"""
import numpy as np
from models import EVAgent
from local_solver_oop import solve_ev_with_agent_methods

def test_baseline_enforcement():
    """Test that reserve provision is blocked when baseline is not followed"""
    print("Testing baseline enforcement for reserve provision...")
    
    # Create EV data
    ev_data = {
        'id': 1,
        'arrival': 0,
        'departure': 10,
        'initial_soc': 40.0,  # %
        'target_soc': 60.0,   # %
        'battery_capacity': 50.0,  # kWh
        'charge_power_max': 10.0,  # kW
        'discharge_power_max': 8.0,  # kW
        'charge_efficiency': 0.9,
        'discharge_efficiency': 0.9,
        'can_discharge': True
    }
    
    # Time parameters
    T = 8  # 4 hours in 30-min slots
    B = 2  # 2 power blocks
    dt_hr = 0.5
    
    # Create baseline and enforcement mask
    P_base = [5.0, 5.0, 7.0, 7.0, 6.0, 6.0, 4.0, 4.0]  # Baseline power [kW]
    baseline_mask_t = [1, 1, 1, 1, 1, 1, 0, 0]  # Enforce first 6 slots
    
    print(f"Baseline power: {P_base}")
    print(f"Enforcement mask: {baseline_mask_t}")
    
    # ADMM parameters that would normally encourage reserve provision
    mi = np.zeros((T, B))
    gp = np.zeros((T, B))
    rho_r = 0.1
    rho_e = 0.1
    rho_p = 0.1
    
    # Consensus variables that encourage deviation from baseline
    c_r_incentive = [[10.0, 5.0], [8.0, 4.0]]  # Strong incentive for reserve
    c_e_incentive = [[0.0, 0.0], [0.0, 0.0]]
    c_p_deviation = [10.0, 2.0, 15.0, 1.0, 12.0, 3.0, 8.0, 2.0]  # Deviates from baseline
    
    print(f"Consensus c_p (encourages deviation): {c_p_deviation}")
    
    # Test 1: Without baseline enforcement (should provide reserve despite deviation)
    print("\\n=== Test 1: Without baseline enforcement ===")
    result_no_enforcement = solve_ev_with_agent_methods(
        ev_data, mi, gp, c_r_incentive, c_e_incentive, c_p_deviation,
        T, B, dt_hr, rho_r, rho_e, rho_p, baseline_mask_t,
        P_base=P_base, enforce_baseline_for_reserve=False
    )
    
    if result_no_enforcement:
        print(f"Charging pattern: {[f'{x:.1f}' for x in result_no_enforcement['p_ch']]}")
        print(f"Reserve cuts: {[f'{x:.1f}' for x in result_no_enforcement['r_cut']]}")
        
        # Check baseline adherence
        baseline_deviations = []
        for t in range(T):
            if baseline_mask_t[t] == 1:
                dev = abs(result_no_enforcement['p_ch'][t] - P_base[t])
                baseline_deviations.append(dev)
        
        max_deviation = max(baseline_deviations) if baseline_deviations else 0
        total_reserve = sum(result_no_enforcement['r_cut'])
        
        print(f"Max baseline deviation: {max_deviation:.2f} kW")
        print(f"Total reserve provided: {total_reserve:.1f} kW")
    
    # Test 2: With baseline enforcement (should block reserve if baseline violated)
    print("\\n=== Test 2: With baseline enforcement ===")
    result_with_enforcement = solve_ev_with_agent_methods(
        ev_data, mi, gp, c_r_incentive, c_e_incentive, c_p_deviation,
        T, B, dt_hr, rho_r, rho_e, rho_p, baseline_mask_t,
        P_base=P_base, enforce_baseline_for_reserve=True
    )
    
    if result_with_enforcement:
        print(f"Charging pattern: {[f'{x:.1f}' for x in result_with_enforcement['p_ch']]}")
        print(f"Reserve cuts: {[f'{x:.1f}' for x in result_with_enforcement['r_cut']]}")
        
        # Check baseline adherence
        baseline_deviations = []
        for t in range(T):
            if baseline_mask_t[t] == 1:
                dev = abs(result_with_enforcement['p_ch'][t] - P_base[t])
                baseline_deviations.append(dev)
        
        max_deviation = max(baseline_deviations) if baseline_deviations else 0
        total_reserve = sum(result_with_enforcement['r_cut'])
        
        print(f"Max baseline deviation: {max_deviation:.2f} kW")
        print(f"Total reserve provided: {total_reserve:.1f} kW")
        
        # Check if baseline constraint worked
        if max_deviation > 0.1 and total_reserve == 0:
            print("✓ Baseline enforcement working: Reserve blocked due to baseline violation")
        elif max_deviation <= 0.1 and total_reserve > 0:
            print("✓ Baseline enforcement working: Reserve allowed with baseline adherence")
        else:
            print("? Unexpected result - baseline enforcement may not be working correctly")
    
    # Test 3: With consensus variables that align with baseline
    print("\\n=== Test 3: Consensus aligned with baseline ===")
    c_p_aligned = P_base.copy()  # Consensus matches baseline exactly
    
    result_aligned = solve_ev_with_agent_methods(
        ev_data, mi, gp, c_r_incentive, c_e_incentive, c_p_aligned,
        T, B, dt_hr, rho_r, rho_e, rho_p, baseline_mask_t,
        P_base=P_base, enforce_baseline_for_reserve=True
    )
    
    if result_aligned:
        print(f"Charging pattern: {[f'{x:.1f}' for x in result_aligned['p_ch']]}")
        print(f"Reserve cuts: {[f'{x:.1f}' for x in result_aligned['r_cut']]}")
        
        baseline_deviations = []
        for t in range(T):
            if baseline_mask_t[t] == 1:
                dev = abs(result_aligned['p_ch'][t] - P_base[t])
                baseline_deviations.append(dev)
        
        max_deviation = max(baseline_deviations) if baseline_deviations else 0
        total_reserve = sum(result_aligned['r_cut'])
        
        print(f"Max baseline deviation: {max_deviation:.2f} kW")
        print(f"Total reserve provided: {total_reserve:.1f} kW")
        
        if max_deviation <= 0.1 and total_reserve > 0:
            print("✓ Reserve provision allowed when baseline is followed")
        else:
            print("? Unexpected result when consensus aligns with baseline")

    print("\\n" + "="*60)
    print("BASELINE ENFORCEMENT SUMMARY")
    print("="*60)
    
    if result_no_enforcement and result_with_enforcement:
        reserve_without = sum(result_no_enforcement['r_cut'])
        reserve_with = sum(result_with_enforcement['r_cut'])
        
        print(f"Reserve without enforcement: {reserve_without:.1f} kW")
        print(f"Reserve with enforcement: {reserve_with:.1f} kW")
        
        if reserve_with < reserve_without:
            print("✓ Baseline enforcement successfully reduces reserve when baseline is violated")
        else:
            print("? Baseline enforcement may not be working as expected")

if __name__ == "__main__":
    test_baseline_enforcement()
