"""
Test reserve provision with and without baseline enforcement
"""
import numpy as np
from models import EVAgent
from local_solver_oop import solve_ev_with_agent_methods

def test_reserve_with_baseline():
    """Test reserve provision behavior with baseline constraints"""
    print("Testing reserve provision with baseline enforcement...")
    
    # Create EV with flexibility for reserve provision
    ev_data = {
        'id': 1,
        'arrival': 0,
        'departure': 10,
        'initial_soc': 30.0,   # Start with some charge
        'target_soc': 70.0,    # Moderate target - room for flexibility
        'battery_capacity': 80.0,  # kWh
        'charge_power_max': 15.0,  # kW
        'discharge_power_max': 10.0, 
        'charge_efficiency': 0.9,
        'discharge_efficiency': 0.9,
        'can_discharge': True
    }
    
    # Time parameters
    T = 8   # 4 hours
    B = 2   # 2 power blocks
    dt_hr = 0.5
    
    # Create moderate baseline that allows some flexibility
    P_base = [6.0, 6.0, 8.0, 8.0, 5.0, 5.0, 3.0, 3.0]
    baseline_mask_t = [1, 1, 1, 1, 0, 0, 0, 0]  # Enforce first 4 slots only
    
    print(f"Baseline power: {P_base}")
    print(f"Enforcement mask: {baseline_mask_t}")
    
    # ADMM parameters
    mi = np.zeros((T, B))
    gp = np.zeros((T, B))
    rho_r = 2.0
    rho_e = 2.0  
    rho_p = 2.0
    
    # Create strong incentives for reserve provision
    # Negative consensus values reward reserve cuts
    c_r_incentive = [[-20.0, -15.0] for _ in range(T)]  # Strong incentive for reserve
    c_e_incentive = [[0.0, 0.0] for _ in range(T)]
    
    # Test cases
    test_cases = [
        {
            'name': 'Baseline-compatible consensus (allows reserve)',
            'c_p': [6.0, 6.0, 8.0, 8.0, 10.0, 10.0, 8.0, 5.0],  # Matches enforced slots
            'description': 'Consensus matches baseline in enforced slots'
        },
        {
            'name': 'Baseline-violating consensus (should block reserve)',
            'c_p': [12.0, 2.0, 15.0, 1.0, 10.0, 10.0, 8.0, 5.0],  # Violates enforced slots
            'description': 'Consensus deviates from baseline in enforced slots'
        }
    ]
    
    print(f"Reserve incentive c_r: {c_r_incentive[0]}")  # Show incentive structure
    
    for case in test_cases:
        print(f"\\n{'='*60}")
        print(f"Case: {case['name']}")
        print(f"Description: {case['description']}")
        print(f"Consensus c_p: {case['c_p']}")
        
        # Test WITHOUT baseline enforcement (baseline violation allowed)
        print(f"\\n--- Without baseline enforcement ---")
        result_no_enforce = solve_ev_with_agent_methods(
            ev_data, mi, gp, c_r_incentive, c_e_incentive, case['c_p'],
            T, B, dt_hr, rho_r, rho_e, rho_p, baseline_mask_t,
            P_base=P_base, enforce_baseline_for_reserve=False
        )
        
        if result_no_enforce:
            charging = result_no_enforce['p_ch']
            reserve = result_no_enforce['r_cut']
            total_reserve = sum(reserve)
            
            # Calculate baseline deviations in enforced slots
            deviations = []
            for t in range(T):
                if baseline_mask_t[t] == 1:
                    dev = abs(charging[t] - P_base[t])
                    deviations.append(dev)
            max_deviation = max(deviations) if deviations else 0
            
            print(f"Charging pattern: {[f'{x:.1f}' for x in charging]}")
            print(f"Reserve cuts: {[f'{x:.1f}' for x in reserve]}")
            print(f"Total reserve: {total_reserve:.1f} kW")
            print(f"Max baseline deviation: {max_deviation:.1f} kW")
        
        # Test WITH baseline enforcement (baseline must be followed for reserve)
        print(f"\\n--- With baseline enforcement ---")
        result_with_enforce = solve_ev_with_agent_methods(
            ev_data, mi, gp, c_r_incentive, c_e_incentive, case['c_p'],
            T, B, dt_hr, rho_r, rho_e, rho_p, baseline_mask_t,
            P_base=P_base, enforce_baseline_for_reserve=True
        )
        
        if result_with_enforce:
            charging = result_with_enforce['p_ch']
            reserve = result_with_enforce['r_cut']
            total_reserve = sum(reserve)
            
            # Calculate baseline deviations in enforced slots
            deviations = []
            for t in range(T):
                if baseline_mask_t[t] == 1:
                    dev = abs(charging[t] - P_base[t])
                    deviations.append(dev)
            max_deviation = max(deviations) if deviations else 0
            
            print(f"Charging pattern: {[f'{x:.1f}' for x in charging]}")
            print(f"Reserve cuts: {[f'{x:.1f}' for x in reserve]}")
            print(f"Total reserve: {total_reserve:.1f} kW")
            print(f"Max baseline deviation: {max_deviation:.1f} kW")
            
            # Analysis
            if result_no_enforce:
                reserve_change = sum(result_no_enforce['r_cut']) - total_reserve
                baseline_improvement = max([
                    abs(result_no_enforce['p_ch'][t] - P_base[t]) for t in range(T) 
                    if baseline_mask_t[t] == 1
                ]) - max_deviation
                
                print(f"\\nImpact of baseline enforcement:")
                print(f"  Reserve reduction: {reserve_change:.1f} kW")
                print(f"  Baseline adherence improvement: {baseline_improvement:.1f} kW")
                
                if max_deviation <= 0.01:
                    print(f"  Status: ✓ Perfect baseline adherence achieved")
                else:
                    print(f"  Status: ⚠ Some baseline deviation remains")
        
        print(f"\\n" + "-"*60)
    
    # Final summary
    print(f"\\n{'='*60}")
    print("BASELINE ENFORCEMENT SUMMARY")
    print("="*60)
    print("The baseline enforcement constraint ensures that:")
    print("1. In enforced time slots, charging power must exactly match baseline")
    print("2. This constraint is independent of reserve provision")
    print("3. Reserve cuts can still be optimized within other constraints")
    print("4. The system prioritizes baseline adherence over reserve optimization")

if __name__ == "__main__":
    test_reserve_with_baseline()
