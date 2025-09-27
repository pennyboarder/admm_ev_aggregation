"""
Test L2 regularization implementation with piecewise linear approximation
"""
import numpy as np
from models import EVAgent
from local_solver_oop import solve_ev_with_agent_methods

def test_l2_regularization():
    """Test L2 regularization approximation"""
    print("Testing L2 regularization approximation...")
    
    # Create simple EV data
    ev_data = {
        'id': 1,
        'arrival': 0,
        'departure': 10,
        'initial_soc': 30.0,  # %
        'target_soc': 80.0,   # %
        'battery_capacity': 50.0,  # kWh
        'charge_power_max': 10.0,  # kW
        'discharge_power_max': 8.0,  # kW
        'charge_efficiency': 0.9,
        'discharge_efficiency': 0.9,
        'can_discharge': True
    }
    
    # Time parameters
    T = 12
    B = 2  # 2 blocks for testing
    dt_hr = 0.5
    
    # Create simple multipliers (zeros for initial test)
    mi = np.zeros((T, B))
    gp = np.zeros((T, B))
    baseline_mask_t = np.ones(T)  # All time slots are baseline
    
    # ADMM parameters
    rho_r = 0.1
    rho_e = 0.1
    rho_p = 0.1
    
    # Test with L1 only first
    print("\n=== Testing L1 regularization ===")
    c_r_l1 = np.random.normal(0, 0.1, (T, B))
    c_e_l1 = np.random.normal(0, 0.1, (T, B))
    c_p_l1 = np.random.normal(0, 0.1, (T, B))
    
    result_l1 = solve_ev_with_agent_methods(
        ev_data, mi, gp, c_r_l1, c_e_l1, c_p_l1,
        T, B, dt_hr, rho_r, rho_e, rho_p, baseline_mask_t,
        use_l2=False
    )
    
    if result_l1:
        print(f"L1 optimization successful")
        print(f"Total charge power: {np.sum(result_l1['p_ch']):.2f}")
        print(f"Total discharge power: {np.sum(result_l1['p_dch']):.2f}")
        print(f"Final SOC: {result_l1['soc'][-1]*100:.1f}%")
    else:
        print("L1 optimization failed")
        return
    
    # Test with L2 regularization
    print("\n=== Testing L2 regularization ===")
    
    # L2 parameters
    rho_r_l2 = 0.05
    rho_e_l2 = 0.05
    rho_p_l2 = 0.05
    num_segments = 5  # Number of segments for piecewise approximation
    
    result_l2 = solve_ev_with_agent_methods(
        ev_data, mi, gp, c_r_l1, c_e_l1, c_p_l1,  # Same multipliers
        T, B, dt_hr, rho_r, rho_e, rho_p, baseline_mask_t,
        use_l2=True, rho_r_l2=rho_r_l2, rho_e_l2=rho_e_l2, rho_p_l2=rho_p_l2,
        num_segments=num_segments
    )
    
    if result_l2:
        print(f"L2 optimization successful")
        print(f"Total charge power: {np.sum(result_l2['p_ch']):.2f}")
        print(f"Total discharge power: {np.sum(result_l2['p_dch']):.2f}")
        print(f"Final SOC: {result_l2['soc'][-1]*100:.1f}%")
        
        # Compare L1 vs L2 results
        print("\n=== Comparison ===")
        l1_charge_variance = np.var(result_l1['p_ch'])
        l2_charge_variance = np.var(result_l2['p_ch'])
        
        print(f"Charge power variance - L1: {l1_charge_variance:.4f}, L2: {l2_charge_variance:.4f}")
        
        if l2_charge_variance < l1_charge_variance:
            print("✓ L2 regularization successfully smoothed the charging profile")
        else:
            print("? L2 regularization didn't smooth as expected")
            
    else:
        print("L2 optimization failed")
        return
    
    print("\n=== Testing different L2 strengths ===")
    l2_strengths = [0.01, 0.05, 0.1, 0.2]
    
    for strength in l2_strengths:
        result = solve_ev_with_agent_methods(
            ev_data, mi, gp, c_r_l1, c_e_l1, c_p_l1,
            T, B, dt_hr, rho_r, rho_e, rho_p, baseline_mask_t,
            use_l2=True, rho_r_l2=strength, rho_e_l2=strength, rho_p_l2=strength,
            num_segments=num_segments
        )
        
        if result:
            variance = np.var(result['p_ch'])
            print(f"L2 strength {strength:.2f}: charge variance = {variance:.4f}")
        else:
            print(f"L2 strength {strength:.2f}: optimization failed")

if __name__ == "__main__":
    test_l2_regularization()
