"""
Test bidding decision optimization vs fixed bidding strategies
"""
import numpy as np
from models import MarketInputs
from bidding_optimizer import solve_ev_with_bidding_optimization, solve_ev_with_fixed_bidding
from utils import build_blocks

def test_bidding_optimization():
    """Test endogenous bidding decision optimization"""
    print("Testing bidding decision optimization...")
    
    # Create EV data
    ev_data = {
        'id': 'OptimizedEV',
        'arrival': 0,
        'departure': 12,
        'initial_soc': 25.0,  # %
        'target_soc': 85.0,   # %
        'battery_capacity': 60.0,  # kWh
        'charge_power_max': 12.0,  # kW
        'discharge_power_max': 10.0,
        'charge_efficiency': 0.9,
        'discharge_efficiency': 0.9,
        'can_discharge': False
    }
    
    # Time parameters
    T = 12  # 1 hour in 5-min slots
    B = 2   # 2 blocks
    dt_hr = 5.0/60.0  # 5-minute slots
    
    # Build blocks
    blocks = build_blocks(T, 5, 30)  # 30-minute blocks
    print(f"Blocks: {blocks}")
    
    # Market conditions
    mi = MarketInputs(
        energy_buy_price_per_kwh=[0.30] * T,  # ¥/kWh
        cap_price_cut_kw_per_block=[150.0, 100.0]  # High capacity prices
    )
    
    # System parameters
    R_max_per_block = [8.0, 6.0]  # Max reserve per block [kW]
    cap_prices = mi.cap_price_cut_kw_per_block
    
    # Baseline scenarios
    baseline_scenarios = [
        {
            'name': 'Low baseline (easy to follow)',
            'P_base': [3.0] * 6 + [4.0] * 6,
            'description': 'Charging needs align with baseline'
        },
        {
            'name': 'High baseline (hard to follow)', 
            'P_base': [15.0] * 6 + [10.0] * 6,
            'description': 'Baseline exceeds EV capability'
        },
        {
            'name': 'Variable baseline',
            'P_base': [2.0, 4.0, 6.0, 8.0, 6.0, 4.0, 10.0, 8.0, 6.0, 4.0, 2.0, 1.0],
            'description': 'Complex baseline pattern'
        }
    ]
    
    print(f"\\nEV: {ev_data['charge_power_max']}kW, {ev_data['battery_capacity']}kWh")
    print(f"SOC: {ev_data['initial_soc']}% → {ev_data['target_soc']}%")
    print(f"Capacity prices: {cap_prices} ¥/kW per block")
    print(f"Max reserve: {R_max_per_block} kW per block")
    
    for scenario in baseline_scenarios:
        print(f"\\n{'='*70}")
        print(f"Scenario: {scenario['name']}")
        print(f"Description: {scenario['description']}")
        print(f"Baseline: {scenario['P_base']}")
        print("="*70)
        
        P_base = scenario['P_base']
        
        # Test 1: Optimized bidding decisions
        print("\\n--- Optimized Bidding Decisions ---")
        result_optimized = solve_ev_with_bidding_optimization(
            ev_data, mi, None, blocks, T, B, dt_hr,
            R_max_per_block, cap_prices, P_base,
            time_limit_sec=10, msg=0
        )
        
        if result_optimized.get("model_status") == "Optimal":
            opt_bidding = result_optimized['bid_decision']
            opt_charging = result_optimized['p_ch']
            opt_reserve = result_optimized['r_cut']
            opt_metrics = result_optimized.get('bidding_metrics', {})
            
            print(f"Optimal bidding decisions: {opt_bidding}")
            print(f"Charging pattern: {[f'{x:.1f}' for x in opt_charging]}")
            print(f"Reserve provision: {[f'{x:.1f}' for x in opt_reserve]} kW")
            print(f"Final SOC: {result_optimized['soc'][-1]*100:.1f}%")
            print(f"Net profit: {opt_metrics.get('net_profit', 0):.1f} ¥")
            
            # Check baseline compliance in bidding blocks
            compliance_check = []
            for b in range(B):
                if opt_bidding[b] == 1 and b < len(blocks):
                    block_slots = blocks[b]
                    deviations = [abs(opt_charging[t] - P_base[t]) for t in block_slots if t < len(P_base)]
                    max_dev = max(deviations) if deviations else 0
                    compliance_check.append(max_dev < 0.01)
            
            if all(compliance_check):
                print("✓ Baseline compliance verified in all bidding blocks")
            else:
                print("⚠ Baseline compliance issues detected")
        else:
            print(f"Optimization failed: {result_optimized.get('model_status', 'Unknown')}")
            continue
        
        # Test 2: Fixed bidding strategies for comparison
        fixed_strategies = [
            ([1, 1], "Bid in both blocks"),
            ([1, 0], "Bid in first block only"),
            ([0, 1], "Bid in second block only"),
            ([0, 0], "No bidding")
        ]
        
        print("\\n--- Comparison with Fixed Bidding Strategies ---")
        for enforce_blocks, strategy_name in fixed_strategies:
            result_fixed = solve_ev_with_fixed_bidding(
                ev_data, mi, None, blocks, T, B, dt_hr,
                R_max_per_block, cap_prices, enforce_blocks, P_base,
                time_limit_sec=10, msg=0
            )
            
            if result_fixed.get("model_status") == "Optimal":
                fixed_objective = result_fixed.get('objective_value', 0)
                opt_objective = result_optimized.get('objective_value', 0)
                
                improvement = opt_objective - fixed_objective
                improvement_pct = (improvement / abs(fixed_objective) * 100) if fixed_objective != 0 else 0
                
                print(f"  {strategy_name}: Objective = {fixed_objective:.1f}, "
                      f"Improvement = {improvement:+.1f} ({improvement_pct:+.1f}%)")
        
        # Summary
        print(f"\\n--- Summary for {scenario['name']} ---")
        if result_optimized.get("model_status") == "Optimal":
            opt_blocks = sum(opt_bidding)
            total_reserve = sum(opt_reserve)
            
            print(f"Optimal strategy: Bid in {opt_blocks}/{B} blocks")
            print(f"Total reserve provision: {total_reserve:.1f} kW")
            
            if opt_blocks == 0:
                print("→ Market conditions favor not participating in reserve market")
            elif opt_blocks == B:
                print("→ Market conditions favor maximum reserve participation")
            else:
                print("→ Selective participation optimizes trade-offs")

if __name__ == "__main__":
    test_bidding_optimization()
