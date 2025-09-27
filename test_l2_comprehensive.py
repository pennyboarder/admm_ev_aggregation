"""
Comprehensive L2 regularization test and visualization
"""
import numpy as np
import matplotlib.pyplot as plt
from models import EVAgent
from local_solver_oop import solve_ev_with_agent_methods

def comprehensive_l2_test():
    """Test L2 regularization with different strengths and visualize results"""
    print("Comprehensive L2 regularization test...")
    
    # Create EV data
    ev_data = {
        'id': 1,
        'arrival': 0,
        'departure': 10,
        'initial_soc': 20.0,  # %
        'target_soc': 90.0,   # %
        'battery_capacity': 50.0,  # kWh
        'charge_power_max': 10.0,  # kW
        'discharge_power_max': 8.0,  # kW
        'charge_efficiency': 0.9,
        'discharge_efficiency': 0.9,
        'can_discharge': True
    }
    
    # Time parameters
    T = 16  # 8 hours in 30-min slots
    B = 2
    dt_hr = 0.5
    
    # Create varying consensus variables to see L2 effect
    mi = np.zeros((T, B))
    gp = np.zeros((T, B))
    baseline_mask_t = np.ones(T)
    
    # ADMM parameters
    rho_r = 0.1
    rho_e = 0.1
    rho_p = 0.1
    
    # Create highly varying consensus variable
    c_r_l1 = np.random.normal(0, 0.1, (T, B))
    c_e_l1 = np.random.normal(0, 0.1, (T, B))
    c_p_l1 = np.array([8.0, -5.0, 10.0, -3.0, 7.0, -6.0, 9.0, -2.0, 
                       6.0, -4.0, 8.0, -1.0, 5.0, -3.0, 7.0, -2.0])
    
    print(f"Consensus variable c_p: {c_p_l1}")
    
    # Test different L2 regularization strengths
    l2_strengths = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    results = {}
    
    for strength in l2_strengths:
        print(f"\\nTesting L2 strength: {strength}")
        
        use_l2 = strength > 0
        result = solve_ev_with_agent_methods(
            ev_data, mi, gp, c_r_l1, c_e_l1, c_p_l1,
            T, B, dt_hr, rho_r, rho_e, rho_p, baseline_mask_t,
            use_l2=use_l2, rho_r_l2=strength, rho_e_l2=strength, rho_p_l2=strength,
            num_segments=10
        )
        
        if result:
            variance = np.var(result['p_ch'])
            total_charge = np.sum(result['p_ch'])
            final_soc = result['soc'][-1] * 100
            
            results[strength] = {
                'charge_pattern': result['p_ch'],
                'soc_pattern': [s * 100 for s in result['soc']],
                'variance': variance,
                'total_charge': total_charge,
                'final_soc': final_soc
            }
            
            print(f"  Variance: {variance:.4f}")
            print(f"  Total charge: {total_charge:.2f} kW")
            print(f"  Final SOC: {final_soc:.1f}%")
        else:
            print(f"  Optimization failed!")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot charge patterns for different L2 strengths
    ax1 = axes[0, 0]
    time_slots = np.arange(T)
    colors = plt.cm.viridis(np.linspace(0, 1, len(results)))
    
    for i, (strength, result) in enumerate(results.items()):
        ax1.plot(time_slots, result['charge_pattern'], 
                marker='o', label=f'L2={strength}', color=colors[i], linewidth=2)
    
    ax1.set_xlabel('Time slot (30-min intervals)')
    ax1.set_ylabel('Charging power (kW)')
    ax1.set_title('Charging Patterns with Different L2 Regularization')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot SOC evolution
    ax2 = axes[0, 1]
    for i, (strength, result) in enumerate(results.items()):
        ax2.plot(time_slots, result['soc_pattern'], 
                marker='s', label=f'L2={strength}', color=colors[i], linewidth=2)
    
    ax2.set_xlabel('Time slot (30-min intervals)')
    ax2.set_ylabel('State of Charge (%)')
    ax2.set_title('SOC Evolution with Different L2 Regularization')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot variance vs L2 strength
    ax3 = axes[1, 0]
    strengths = list(results.keys())
    variances = [results[s]['variance'] for s in strengths]
    
    ax3.plot(strengths, variances, marker='o', linewidth=2, color='red')
    ax3.set_xlabel('L2 Regularization Strength')
    ax3.set_ylabel('Charging Power Variance')
    ax3.set_title('Smoothing Effect of L2 Regularization')
    ax3.grid(True, alpha=0.3)
    
    # Plot final SOC vs L2 strength
    ax4 = axes[1, 1]
    final_socs = [results[s]['final_soc'] for s in strengths]
    
    ax4.plot(strengths, final_socs, marker='s', linewidth=2, color='blue')
    ax4.set_xlabel('L2 Regularization Strength')
    ax4.set_ylabel('Final SOC (%)')
    ax4.set_title('Impact on Target Achievement')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    import os
    output_dir = "/workspaces/admm_ev_aggregation/results"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/l2_regularization_analysis.png", dpi=300, bbox_inches='tight')
    print(f"\\nVisualization saved to {output_dir}/l2_regularization_analysis.png")
    
    # Summary statistics
    print("\\n" + "="*60)
    print("L2 REGULARIZATION SUMMARY")
    print("="*60)
    
    baseline_variance = results[0.0]['variance']
    print(f"Baseline variance (no L2): {baseline_variance:.4f}")
    
    for strength in [0.5, 1.0, 5.0]:
        if strength in results:
            reduction = (baseline_variance - results[strength]['variance']) / baseline_variance * 100
            print(f"L2 strength {strength}: {reduction:.1f}% variance reduction")
    
    return results

if __name__ == "__main__":
    results = comprehensive_l2_test()
