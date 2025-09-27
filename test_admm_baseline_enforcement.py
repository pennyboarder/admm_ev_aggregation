"""
Test baseline enforcement in full ADMM coordination
"""
import numpy as np
from models import EVAgent, GlobalParams, MarketInputs
from admm_coordinator import admm_cut_with_baseline_oop

def test_admm_baseline_enforcement():
    """Test baseline enforcement in full ADMM system"""
    print("Testing baseline enforcement in ADMM coordination...")
    
    # Create test EVs
    evs = []
    for i in range(3):  # 3 EVs for aggregation
        ev = EVAgent(
            name=f"EV_{i+1}",
            e_kwh=50.0,
            soc0=0.3,  # 30%
            soc_min=0.1,
            soc_max=0.9,  # 90%
            p_charge_max=10.0,
            eta_c=0.9,
            available=[1] * 24,  # Available for 24 time slots (2 hours)
            eta_d=0.9,
            can_discharge_to_grid=True
        )
        evs.append(ev)
    
    # Global parameters
    gp = GlobalParams(dt_min=5, sustain_min=30)
    
    # Market inputs - create pricing that incentivizes reserve
    T = 24  # 24 5-min slots = 2 hours
    mi = MarketInputs(
        energy_buy_price_per_kwh=[0.25] * T,  # ¥/kWh
        cap_price_cut_kw_per_block=[100.0] * 4  # High reserve price ¥/kW per block
    )
    
    # Reserve bid and baseline
    R_bid_blk = [15.0, 10.0, 8.0, 5.0]  # Target reserve per block [kW]
    
    # Create aggregate baseline for all EVs
    # Total needed charging ~ 3 EVs * (90%-30%) * 50kWh / 2hrs = 45 kWh/hr = 22.5 kW average
    P_base = [8.0] * 6 + [12.0] * 6 + [10.0] * 6 + [6.0] * 6  # Varying baseline
    
    # Enforce baseline in first 2 blocks only
    enforce_blocks = [1, 1, 0, 0]  # Enforce baseline in blocks 0,1 only
    
    print(f"Number of EVs: {len(evs)}")
    print(f"Time horizon: {T} slots ({T*5} minutes)")
    print(f"Reserve bids: {R_bid_blk} kW per block")
    print(f"Baseline enforcement: blocks {[i for i, e in enumerate(enforce_blocks) if e]}")
    print(f"Baseline power profile (first 12 slots): {P_base[:12]}")
    
    # Test 1: Without baseline enforcement
    print("\\n" + "="*60)
    print("Test 1: ADMM without baseline enforcement")
    print("="*60)
    
    result1 = admm_cut_with_baseline_oop(
        evs, gp, mi, R_bid_blk, P_base, enforce_blocks,
        post_slots=6, rho_r=6.0, rho_e=4.0, rho_p=3.0,
        steps=20, time_limit_sec_local=5, msg_local=False,
        num_threads=1, use_oop_solver=True,
        enforce_baseline_for_reserve=False
    )
    
    if 'x_r' in result1:
        print(f"ADMM completed with status: {result1.get('status', 'Unknown')}")
        
        # Analyze results
        total_reserve = [sum(result1['x_r'][i]) for i in range(len(evs))]
        total_charging = []
        baseline_deviations = []
        
        for i in range(len(evs)):
            charging = result1['x_p'][i]
            total_charging.append(sum(charging))
            
        aggregate_charging = [sum(result1['x_p'][i][t] for i in range(len(evs))) for t in range(T)]
        
        # Calculate baseline deviations in enforced slots
        enforced_slots = []
        for b in range(len(enforce_blocks)):
            if enforce_blocks[b] == 1:
                # Assuming 6 slots per block
                start_slot = b * 6
                enforced_slots.extend(range(start_slot, start_slot + 6))
        
        for t in enforced_slots:
            if t < len(aggregate_charging):
                deviation = abs(aggregate_charging[t] - P_base[t])
                baseline_deviations.append(deviation)
        
        print(f"Total reserve per EV: {[f'{r:.1f}' for r in total_reserve]} kW")
        print(f"Total charging per EV: {[f'{c:.1f}' for c in total_charging]} kWh")
        print(f"Aggregate reserve: {sum(total_reserve):.1f} kW")
        print(f"Max baseline deviation: {max(baseline_deviations) if baseline_deviations else 0:.1f} kW")
    
    # Test 2: With baseline enforcement  
    print("\\n" + "="*60)
    print("Test 2: ADMM with baseline enforcement")
    print("="*60)
    
    result2 = admm_cut_with_baseline_oop(
        evs, gp, mi, R_bid_blk, P_base, enforce_blocks,
        post_slots=6, rho_r=6.0, rho_e=4.0, rho_p=3.0,
        steps=20, time_limit_sec_local=5, msg_local=False,
        num_threads=1, use_oop_solver=True,
        enforce_baseline_for_reserve=True
    )
    
    if 'x_r' in result2:
        print(f"ADMM completed with status: {result2.get('status', 'Unknown')}")
        
        # Analyze results
        total_reserve = [sum(result2['x_r'][i]) for i in range(len(evs))]
        total_charging = []
        
        for i in range(len(evs)):
            charging = result2['x_p'][i]
            total_charging.append(sum(charging))
            
        aggregate_charging = [sum(result2['x_p'][i][t] for i in range(len(evs))) for t in range(T)]
        
        # Calculate baseline deviations in enforced slots
        baseline_deviations = []
        for t in enforced_slots:
            if t < len(aggregate_charging):
                deviation = abs(aggregate_charging[t] - P_base[t])
                baseline_deviations.append(deviation)
        
        print(f"Total reserve per EV: {[f'{r:.1f}' for r in total_reserve]} kW")
        print(f"Total charging per EV: {[f'{c:.1f}' for c in total_charging]} kWh")
        print(f"Aggregate reserve: {sum(total_reserve):.1f} kW")
        print(f"Max baseline deviation: {max(baseline_deviations) if baseline_deviations else 0:.1f} kW")
    
    # Comparison
    if 'x_r' in result1 and 'x_r' in result2:
        print("\\n" + "="*60)
        print("COMPARISON: Impact of baseline enforcement")
        print("="*60)
        
        reserve1 = sum([sum(result1['x_r'][i]) for i in range(len(evs))])
        reserve2 = sum([sum(result2['x_r'][i]) for i in range(len(evs))])
        
        # Calculate baseline adherence
        agg1 = [sum(result1['x_p'][i][t] for i in range(len(evs))) for t in range(T)]
        agg2 = [sum(result2['x_p'][i][t] for i in range(len(evs))) for t in range(T)]
        
        dev1 = [abs(agg1[t] - P_base[t]) for t in enforced_slots if t < len(agg1)]
        dev2 = [abs(agg2[t] - P_base[t]) for t in enforced_slots if t < len(agg2)]
        
        max_dev1 = max(dev1) if dev1 else 0
        max_dev2 = max(dev2) if dev2 else 0
        
        print(f"Total aggregate reserve:")
        print(f"  Without enforcement: {reserve1:.1f} kW")
        print(f"  With enforcement: {reserve2:.1f} kW")
        print(f"  Change: {reserve2 - reserve1:.1f} kW")
        
        print(f"\\nBaseline adherence (max deviation in enforced slots):")
        print(f"  Without enforcement: {max_dev1:.1f} kW") 
        print(f"  With enforcement: {max_dev2:.1f} kW")
        print(f"  Improvement: {max_dev1 - max_dev2:.1f} kW")
        
        if max_dev2 <= 0.1:
            print("\\n✓ Baseline enforcement successful: Perfect adherence achieved")
        elif max_dev2 < max_dev1:
            print("\\n✓ Baseline enforcement partially successful: Adherence improved")
        else:
            print("\\n? Baseline enforcement may need adjustment")

if __name__ == "__main__":
    test_admm_baseline_enforcement()
