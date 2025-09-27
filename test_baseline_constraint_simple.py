"""
Simple test of baseline enforcement functionality
"""
import numpy as np
from models import EVAgent

def test_baseline_constraint_simple():
    """Simple test of baseline constraint implementation"""
    print("Testing baseline constraint implementation...")
    
    # Create a single EV
    ev = EVAgent(
        name="TestEV",
        e_kwh=50.0,
        soc0=0.4,  # 40% initial SOC
        soc_min=0.1,
        soc_max=0.9,
        p_charge_max=10.0,
        eta_c=0.9,
        available=[1] * 8,  # 8 time slots
        eta_d=0.9,
        can_discharge_to_grid=False
    )
    
    # Test parameters
    T = 8
    B = 2
    dt_hr = 0.25  # 15 min slots
    
    # Define baseline and enforcement
    P_base = [5.0, 5.0, 7.0, 7.0, 6.0, 6.0, 4.0, 4.0]
    baseline_mask_t = [1, 1, 1, 1, 0, 0, 0, 0]  # Enforce first 4 slots
    
    print(f"Baseline power: {P_base}")
    print(f"Enforcement mask: {baseline_mask_t}")
    
    # Create optimization model
    import pulp
    model = pulp.LpProblem("BaselineTest", pulp.LpMaximize)
    
    # Create variables
    ev.create_optimization_variables(T, B)
    
    # Add basic constraints
    ev.add_power_constraints(model, T)
    ev.add_soc_constraints(model, T, dt_hr)
    
    print("\\n=== Test without baseline constraint ===")
    
    # Simple objective: maximize charging in early slots
    variables = ev.variables
    objective1 = pulp.lpSum([variables['p_ch'][t] * (T-t) for t in range(T)])
    model += objective1
    
    # Solve
    solver = pulp.PULP_CBC_CMD(msg=0)
    model.solve(solver)
    
    if model.status == 1:  # Optimal
        result1 = [pulp.value(variables['p_ch'][t]) for t in range(T)]
        print(f"Charging pattern: {[f'{x:.1f}' for x in result1]}")
        
        # Check baseline deviations
        deviations = []
        for t in range(T):
            if baseline_mask_t[t] == 1:
                dev = abs(result1[t] - P_base[t])
                deviations.append(dev)
        print(f"Baseline deviations: {[f'{d:.1f}' for d in deviations]}")
        print(f"Max deviation: {max(deviations):.1f} kW")
    else:
        print("Optimization failed")
        return
    
    print("\\n=== Test with baseline constraint ===")
    
    # Add baseline constraints
    ev.add_baseline_constraints(model, P_base, baseline_mask_t, T, B)
    
    # Solve again
    model.solve(solver)
    
    if model.status == 1:  # Optimal
        result2 = [pulp.value(variables['p_ch'][t]) for t in range(T)]
        print(f"Charging pattern: {[f'{x:.1f}' for x in result2]}")
        
        # Check baseline deviations
        deviations = []
        for t in range(T):
            if baseline_mask_t[t] == 1:
                dev = abs(result2[t] - P_base[t])
                deviations.append(dev)
        print(f"Baseline deviations: {[f'{d:.1f}' for d in deviations]}")
        print(f"Max deviation: {max(deviations):.1f} kW")
        
        # Compare results
        print(f"\\n=== Comparison ===")
        print(f"Without constraint: {[f'{x:.1f}' for x in result1]}")
        print(f"With constraint: {[f'{x:.1f}' for x in result2]}")
        
        # Check if baseline is enforced
        enforced_exactly = all(
            abs(result2[t] - P_base[t]) < 0.001 
            for t in range(T) if baseline_mask_t[t] == 1
        )
        
        if enforced_exactly:
            print("✓ Baseline constraint working: Exact baseline matching achieved")
        else:
            print("✗ Baseline constraint not working properly")
    else:
        print("Optimization with baseline constraint failed")

if __name__ == "__main__":
    test_baseline_constraint_simple()
