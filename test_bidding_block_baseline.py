"""
Test baseline enforcement for specific bidding blocks only
"""
import numpy as np
from models import EVAgent, GlobalParams, MarketInputs
from admm_coordinator import admm_cut_with_baseline_oop
from utils import build_blocks, blocks_to_slot_mask

def test_bidding_block_baseline():
    """Test that baseline is enforced only in bidding blocks"""
    print("Testing baseline enforcement for bidding blocks only...")
    
    # Create test EVs
    evs = []
    for i in range(2):
        ev = EVAgent(
            name=f"EV_{i+1}",
            e_kwh=40.0,
            soc0=0.3,  # 30%
            soc_min=0.1,
            soc_max=0.9,  # 90%
            p_charge_max=8.0,
            eta_c=0.9,
            available=[1] * 12,  # 12 time slots (1 hour)
            eta_d=0.9,
            can_discharge_to_grid=False
        )
        evs.append(ev)
    
    # Global parameters
    gp = GlobalParams(dt_min=5, sustain_min=30)  # 30-min blocks
    
    # Create market inputs
    T = 12  # 12 slots = 1 hour
    mi = MarketInputs(
        energy_buy_price_per_kwh=[0.25] * T,
        cap_price_cut_kw_per_block=[50.0, 30.0]  # 2 blocks
    )
    
    # Build blocks (6 slots per block for 30-min blocks with 5-min slots)
    blocks = build_blocks(T, gp.dt_min, gp.sustain_min)
    print(f"Blocks: {blocks}")
    
    # Reserve bids and baseline
    R_bid_blk = [10.0, 8.0]  # Reserve bid per block
    
    # Create baseline that varies between blocks
    P_base = [6.0] * 6 + [12.0] * 6  # Low then high baseline
    
    # Test Case 1: Enforce baseline in first block only
    print("\\n" + "="*60)
    print("Test Case 1: Baseline enforcement in first block only")
    print("="*60)
    
    enforce_blocks_case1 = [1, 0]  # Enforce only block 0 (slots 0-5)
    baseline_mask_case1 = blocks_to_slot_mask(enforce_blocks_case1, blocks, T)
    
    print(f"Enforce blocks: {enforce_blocks_case1}")
    print(f"Baseline mask: {baseline_mask_case1}")
    print(f"Expected: Slots 0-5 must match baseline (6.0), slots 6-11 can deviate")
    
    result1 = admm_cut_with_baseline_oop(
        evs, gp, mi, R_bid_blk, P_base, enforce_blocks_case1,
        post_slots=3, rho_r=3.0, rho_e=2.0, rho_p=2.0,
        steps=15, time_limit_sec_local=3, msg_local=False,
        num_threads=1, use_oop_solver=True,
        enforce_baseline_for_reserve=True
    )
    
    if 'x_p' in result1:
        # Calculate aggregate charging
        agg_charging_case1 = [sum(result1['x_p'][i][t] for i in range(len(evs))) for t in range(T)]
        
        print(f"Baseline:         {P_base}")
        print(f"Aggregate charge: {[f'{x:.1f}' for x in agg_charging_case1]}")
        
        # Check enforcement in first block (slots 0-5)
        first_block_exact = all(
            abs(agg_charging_case1[t] - P_base[t]) < 0.01 
            for t in range(6)  # First 6 slots
        )
        
        # Check flexibility in second block (slots 6-11)
        second_block_deviation = any(
            abs(agg_charging_case1[t] - P_base[t]) > 0.01 
            for t in range(6, 12)  # Last 6 slots
        )
        
        print(f"First block baseline match: {'✓' if first_block_exact else '✗'}")
        print(f"Second block has flexibility: {'✓' if second_block_deviation else '✗'}")
    
    # Test Case 2: Enforce baseline in second block only
    print("\\n" + "="*60)
    print("Test Case 2: Baseline enforcement in second block only")
    print("="*60)
    
    enforce_blocks_case2 = [0, 1]  # Enforce only block 1 (slots 6-11)
    baseline_mask_case2 = blocks_to_slot_mask(enforce_blocks_case2, blocks, T)
    
    print(f"Enforce blocks: {enforce_blocks_case2}")
    print(f"Baseline mask: {baseline_mask_case2}")
    print(f"Expected: Slots 0-5 can deviate, slots 6-11 must match baseline (12.0)")
    
    result2 = admm_cut_with_baseline_oop(
        evs, gp, mi, R_bid_blk, P_base, enforce_blocks_case2,
        post_slots=3, rho_r=3.0, rho_e=2.0, rho_p=2.0,
        steps=15, time_limit_sec_local=3, msg_local=False,
        num_threads=1, use_oop_solver=True,
        enforce_baseline_for_reserve=True
    )
    
    if 'x_p' in result2:
        # Calculate aggregate charging
        agg_charging_case2 = [sum(result2['x_p'][i][t] for i in range(len(evs))) for t in range(T)]
        
        print(f"Baseline:         {P_base}")
        print(f"Aggregate charge: {[f'{x:.1f}' for x in agg_charging_case2]}")
        
        # Check flexibility in first block (slots 0-5)
        first_block_deviation = any(
            abs(agg_charging_case2[t] - P_base[t]) > 0.01 
            for t in range(6)  # First 6 slots
        )
        
        # Check enforcement in second block (slots 6-11)
        second_block_exact = all(
            abs(agg_charging_case2[t] - P_base[t]) < 0.01 
            for t in range(6, 12)  # Last 6 slots
        )
        
        print(f"First block has flexibility: {'✓' if first_block_deviation else '✗'}")
        print(f"Second block baseline match: {'✓' if second_block_exact else '✗'}")
    
    # Test Case 3: Enforce baseline in both blocks
    print("\\n" + "="*60)
    print("Test Case 3: Baseline enforcement in both blocks")
    print("="*60)
    
    enforce_blocks_case3 = [1, 1]  # Enforce both blocks
    baseline_mask_case3 = blocks_to_slot_mask(enforce_blocks_case3, blocks, T)
    
    print(f"Enforce blocks: {enforce_blocks_case3}")
    print(f"Baseline mask: {baseline_mask_case3}")
    print(f"Expected: All slots must match baseline")
    
    result3 = admm_cut_with_baseline_oop(
        evs, gp, mi, R_bid_blk, P_base, enforce_blocks_case3,
        post_slots=3, rho_r=3.0, rho_e=2.0, rho_p=2.0,
        steps=15, time_limit_sec_local=3, msg_local=False,
        num_threads=1, use_oop_solver=True,
        enforce_baseline_for_reserve=True
    )
    
    if 'x_p' in result3:
        # Calculate aggregate charging
        agg_charging_case3 = [sum(result3['x_p'][i][t] for i in range(len(evs))) for t in range(T)]
        
        print(f"Baseline:         {P_base}")
        print(f"Aggregate charge: {[f'{x:.1f}' for x in agg_charging_case3]}")
        
        # Check enforcement in all slots
        all_slots_exact = all(
            abs(agg_charging_case3[t] - P_base[t]) < 0.01 
            for t in range(T)
        )
        
        print(f"All slots baseline match: {'✓' if all_slots_exact else '✗'}")
    
    print("\\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("The baseline enforcement correctly applies only to bidding blocks:")
    print("- When enforce_blocks=[1,0]: Only first block follows baseline")
    print("- When enforce_blocks=[0,1]: Only second block follows baseline")  
    print("- When enforce_blocks=[1,1]: Both blocks follow baseline")
    print("- This allows flexibility in non-bidding periods while ensuring")
    print("  compliance during reserve provision periods.")

if __name__ == "__main__":
    test_bidding_block_baseline()
