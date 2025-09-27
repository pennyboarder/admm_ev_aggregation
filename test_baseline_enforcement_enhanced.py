"""
Enhanced test for baseline enforcement with stronger reserve incentives
"""
import numpy as np
from models import EVAgent
from local_solver_oop import solve_ev_with_agent_methods

def test_baseline_enforcement_enhanced():
    """Test baseline enforcement with enhanced reserve incentives"""
    print("Testing enhanced baseline enforcement...")
    
    # Create EV data - larger battery with more flexibility
    ev_data = {
        'id': 1,
        'arrival': 0,
        'departure': 10,
        'initial_soc': 20.0,  # %
        'target_soc': 90.0,   # %
        'battery_capacity': 100.0,  # kWh - larger battery
        'charge_power_max': 20.0,  # kW - higher power
        'discharge_power_max': 15.0,  # kW
        'charge_efficiency': 0.9,
        'discharge_efficiency': 0.9,
        'can_discharge': True
    }
    
    # Time parameters
    T = 12  # 6 hours in 30-min slots
    B = 3   # 3 power blocks for more reserve options
    dt_hr = 0.5
    
    # Create baseline - moderate charging needed
    P_base = [8.0, 8.0, 10.0, 10.0, 12.0, 12.0, 8.0, 8.0, 6.0, 6.0, 4.0, 4.0]
    baseline_mask_t = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]  # Enforce first 8 slots
    
    print(f"Baseline power: {P_base}")
    print(f"Enforcement mask: {baseline_mask_t}")
    
    # ADMM parameters
    mi = np.zeros((T, B))
    gp = np.zeros((T, B))
    rho_r = 1.0  # Higher penalty for better convergence
    rho_e = 1.0
    rho_p = 1.0
    
    # Strong incentives for reserve provision
    c_r_strong = [[-15.0, -10.0, -5.0] for _ in range(T)]  # Strong negative incentive (reward for reserve)
    c_e_normal = [[0.0, 0.0, 0.0] for _ in range(T)]
    
    # Test different consensus scenarios
    test_cases = [
        {
            'name': 'Consensus matches baseline exactly',
            'c_p': P_base.copy(),
            'expected_reserve': 'High (baseline followed)',
            'expected_deviation': 'Zero'
        },
        {
            'name': 'Consensus deviates significantly from baseline', 
            'c_p': [15.0, 2.0, 18.0, 3.0, 20.0, 1.0, 15.0, 2.0, 10.0, 5.0, 8.0, 3.0],
            'expected_reserve': 'Zero or low (baseline violated)',
            'expected_deviation': 'High'
        },
        {
            'name': 'Mixed: some slots match, some deviate',
            'c_p': [8.0, 8.0, 15.0, 15.0, 12.0, 12.0, 12.0, 12.0, 10.0, 5.0, 8.0, 3.0],
            'expected_reserve': 'Zero or low (some violations)',
            'expected_deviation': 'Medium'
        }
    ]
    
    results = {}
    
    for case in test_cases:
        print(f"\\n=== {case['name']} ===")
        print(f"Expected reserve: {case['expected_reserve']}")
        print(f"Expected deviation: {case['expected_deviation']}")
        print(f"Consensus c_p: {case['c_p']}")
        
        # Test without enforcement
        result_no_enforce = solve_ev_with_agent_methods(
            ev_data, mi, gp, c_r_strong, c_e_normal, case['c_p'],
            T, B, dt_hr, rho_r, rho_e, rho_p, baseline_mask_t,
            P_base=P_base, enforce_baseline_for_reserve=False
        )
        
        # Test with enforcement  
        result_with_enforce = solve_ev_with_agent_methods(
            ev_data, mi, gp, c_r_strong, c_e_normal, case['c_p'],
            T, B, dt_hr, rho_r, rho_e, rho_p, baseline_mask_t,
            P_base=P_base, enforce_baseline_for_reserve=True
        )
        
        if result_no_enforce and result_with_enforce:
            # Calculate metrics
            charging_no = result_no_enforce['p_ch']
            charging_with = result_with_enforce['p_ch']
            reserve_no = sum(result_no_enforce['r_cut'])
            reserve_with = sum(result_with_enforce['r_cut'])
            
            # Calculate baseline deviations
            deviations_no = []
            deviations_with = []
            for t in range(T):
                if baseline_mask_t[t] == 1:
                    deviations_no.append(abs(charging_no[t] - P_base[t]))
                    deviations_with.append(abs(charging_with[t] - P_base[t]))
            
            max_dev_no = max(deviations_no) if deviations_no else 0
            max_dev_with = max(deviations_with) if deviations_with else 0
            
            print(f"\\nWithout enforcement:")
            print(f"  Charging: {[f'{x:.1f}' for x in charging_no]}")
            print(f"  Reserve: {[f'{x:.1f}' for x in result_no_enforce['r_cut']]}")
            print(f"  Total reserve: {reserve_no:.1f} kW")
            print(f"  Max baseline deviation: {max_dev_no:.1f} kW")
            
            print(f"\\nWith enforcement:")
            print(f"  Charging: {[f'{x:.1f}' for x in charging_with]}")
            print(f"  Reserve: {[f'{x:.1f}' for x in result_with_enforce['r_cut']]}")
            print(f"  Total reserve: {reserve_with:.1f} kW")
            print(f"  Max baseline deviation: {max_dev_with:.1f} kW")
            
            # Analysis
            enforcement_effect = reserve_no - reserve_with
            baseline_improvement = max_dev_no - max_dev_with
            
            print(f"\\nAnalysis:")
            print(f"  Reserve reduction due to enforcement: {enforcement_effect:.1f} kW")
            print(f"  Baseline adherence improvement: {baseline_improvement:.1f} kW")
            
            if max_dev_with <= 0.1 and reserve_with > 0:
                print("  ✓ Baseline followed → Reserve allowed")
            elif max_dev_with > 0.1 and reserve_with == 0:
                print("  ✓ Baseline violated → Reserve blocked")
            elif max_dev_with > 0.1 and reserve_with < reserve_no:
                print("  ✓ Baseline violated → Reserve reduced")
            else:
                print("  ? Unexpected enforcement behavior")
            
            results[case['name']] = {
                'reserve_no_enforce': reserve_no,
                'reserve_with_enforce': reserve_with,
                'deviation_no_enforce': max_dev_no,
                'deviation_with_enforce': max_dev_with
            }
    
    # Summary
    print("\\n" + "="*70)
    print("BASELINE ENFORCEMENT ANALYSIS SUMMARY")
    print("="*70)
    
    for case_name, metrics in results.items():
        print(f"\\n{case_name}:")
        print(f"  Reserve: {metrics['reserve_no_enforce']:.1f} → {metrics['reserve_with_enforce']:.1f} kW")
        print(f"  Deviation: {metrics['deviation_no_enforce']:.1f} → {metrics['deviation_with_enforce']:.1f} kW")
        
        if metrics['deviation_with_enforce'] <= 0.1:
            print(f"  Status: ✓ Baseline adherence achieved")
        else:
            print(f"  Status: ⚠ Baseline still violated")

if __name__ == "__main__":
    test_baseline_enforcement_enhanced()
