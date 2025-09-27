"""
Debug L2 regularization implementation
"""
import numpy as np
from models import EVAgent
from local_solver_oop import solve_ev_with_agent_methods

def debug_l2_implementation():
    """Debug L2 regularization implementation"""
    print("Debugging L2 regularization...")
    
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
    B = 2
    dt_hr = 0.5
    
    # Create multipliers with variation to encourage different charging patterns
    mi = np.zeros((T, B))
    gp = np.zeros((T, B))
    baseline_mask_t = np.ones(T)
    
    # ADMM parameters
    rho_r = 0.1
    rho_e = 0.1
    rho_p = 0.1
    
    # Create varying consensus variables to see L2 effect
    c_r_l1 = np.random.normal(0, 0.1, (T, B))
    c_e_l1 = np.random.normal(0, 0.1, (T, B))
    # Make p vary significantly to see smoothing effect
    c_p_l1 = np.array([5.0, -3.0, 8.0, -2.0, 6.0, -4.0, 9.0, -1.0, 7.0, -3.0, 5.0, -2.0])
    
    print("Consensus variable c_p:", c_p_l1)
    
    # Test without L2 first
    result_no_l2 = solve_ev_with_agent_methods(
        ev_data, mi, gp, c_r_l1, c_e_l1, c_p_l1,
        T, B, dt_hr, rho_r, rho_e, rho_p, baseline_mask_t,
        use_l2=False
    )
    
    if result_no_l2:
        print("\n=== Without L2 regularization ===")
        print(f"Charge pattern: {[f'{x:.2f}' for x in result_no_l2['p_ch']]}")
        print(f"Variance: {np.var(result_no_l2['p_ch']):.4f}")
    
    # Test with strong L2 regularization
    result_l2 = solve_ev_with_agent_methods(
        ev_data, mi, gp, c_r_l1, c_e_l1, c_p_l1,
        T, B, dt_hr, rho_r, rho_e, rho_p, baseline_mask_t,
        use_l2=True, rho_r_l2=10.0, rho_e_l2=10.0, rho_p_l2=10.0,
        num_segments=10
    )
    
    if result_l2:
        print("\n=== With L2 regularization (rho=10.0) ===")
        print(f"Charge pattern: {[f'{x:.2f}' for x in result_l2['p_ch']]}")
        print(f"Variance: {np.var(result_l2['p_ch']):.4f}")
        
        if np.var(result_l2['p_ch']) < np.var(result_no_l2['p_ch']):
            print("✓ L2 regularization successfully reduced variance")
        else:
            print("❌ L2 regularization didn't reduce variance as expected")

if __name__ == "__main__":
    debug_l2_implementation()
