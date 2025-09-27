# -*- coding: utf-8 -*-
"""
ADMM coordinator for EV aggregation with baseline tracking
"""

from typing import List, Dict
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from tqdm import tqdm
from models import EVAgent, GlobalParams, MarketInputs
from utils import build_blocks, blocks_to_slot_mask, build_post_windows
from local_solver import solve_local_ev_cut_with_baseline_L1, solve_local_ev_cut_with_baseline_enforcement_L1
from local_solver_oop import solve_ev_with_agent_methods, solve_single_ev_parallel_oop


def solve_single_ev_parallel(args):
    """Wrapper function for parallel EV optimization"""
    (i, ev, gp, mi, blocks, post_windows, c_r, rho_r, c_e, rho_e, 
     c_p, rho_p, baseline_mask_t, P_base, enforce_baseline_for_reserve,
     time_limit_sec_local, msg_local) = args
    
    if enforce_baseline_for_reserve:
        res = solve_local_ev_cut_with_baseline_enforcement_L1(
            ev, gp, mi, blocks, post_windows,
            c_r, rho_r, c_e, rho_e, c_p, rho_p,
            baseline_mask_t, P_base, enforce_baseline_for_reserve,
            time_limit_sec=time_limit_sec_local, msg=msg_local
        )
    else:
        res = solve_local_ev_cut_with_baseline_L1(
            ev, gp, mi, blocks, post_windows,
            c_r, rho_r, c_e, rho_e, c_p, rho_p,
            baseline_mask_t,
            time_limit_sec=time_limit_sec_local, msg=msg_local
        )
    
    return i, res


def admm_cut_with_baseline_oop(
    evs: List[EVAgent],
    gp: GlobalParams,
    mi: MarketInputs,
    R_bid_blk: List[float],     # target cut capacity per block [kW], only enforced where enforce_blocks[b]=1
    P_base: List[float],        # aggregate baseline [kW] per 5-min slot
    enforce_blocks: List[int],  # 0/1 per block: enable baseline + bid only in these blocks
    post_slots: int = 6,        # recovery window length in 5-min slots (6=30min)
    rho_r: float = 6.0,
    rho_e: float = 4.0,
    rho_p: float = 3.0,
    steps: int = 40,
    time_limit_sec_local: int = 6,
    msg_local: bool = False,
    over_relax: float = 1.2,
    num_threads: int = None,
    use_oop_solver: bool = True,
    enforce_baseline_for_reserve: bool = False,  # Enable baseline-reserve coupling
) -> Dict:
    """
    Object-oriented ADMM coordinator using EVAgent methods
    """
    # Same initialization as original function
    T = len(mi.energy_buy_price_per_kwh)
    blocks = build_blocks(T, gp.dt_min, gp.sustain_min)
    B, N = len(blocks), len(evs)
    post_windows = build_post_windows(blocks, T, post_slots)

    baseline_mask_t = blocks_to_slot_mask(enforce_blocks, blocks, T)
    R_bid_eff = [R_bid_blk[b] if enforce_blocks[b] else 0.0 for b in range(B)]

    # Initialize ADMM variables
    x_r = [[0.0]*B for _ in range(N)]
    x_e = [[0.0]*B for _ in range(N)]
    x_p = [[0.0]*T for _ in range(N)]
    x_d = [[0.0]*T for _ in range(N)]

    s_r = [[0.0]*B for _ in range(N)]; u_r = [[0.0]*B for _ in range(N)]
    s_e = [[0.0]*B for _ in range(N)]; u_e = [[0.0]*B for _ in range(N)]
    s_p = [[0.0]*T for _ in range(N)]; u_p = [[0.0]*T for _ in range(N)]

    hist = []
    best = {"r_norm": 1e18}

    # Determine number of threads
    if num_threads is None:
        import os
        num_threads = min(N, os.cpu_count() or 1)
    
    solver_type = "OOP" if use_oop_solver else "Original"
    pbar = tqdm(range(steps), desc=f"ADMM-{solver_type} ({num_threads} threads)", leave=True, position=0,
                bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}')
    
    for k in pbar:
        # --- x-update: local optimizations using EV agent methods ---
        if num_threads > 1 and N > 1:
            # Parallel execution
            tasks = []
            for i, ev in enumerate(evs):
                c_r = [s_r[i][b] - u_r[i][b] for b in range(B)]
                c_e = [s_e[i][b] - u_e[i][b] for b in range(B)]
                c_p = [s_p[i][t] - u_p[i][t] for t in range(T)]
                
                if use_oop_solver:
                    # Convert EVAgent to dict for OOP solver
                    ev_data = {
                        'id': ev.name,
                        'arrival': 0,
                        'departure': T,
                        'initial_soc': ev.soc0 * 100,  # Convert to %
                        'target_soc': ev.soc_max * 100,
                        'battery_capacity': ev.e_kwh,
                        'charge_power_max': ev.p_charge_max,
                        'discharge_power_max': ev.p_charge_max if ev.can_discharge_to_grid else 0,
                        'charge_efficiency': ev.eta_c,
                        'discharge_efficiency': ev.eta_d,
                        'can_discharge': ev.can_discharge_to_grid,
                        'available': ev.available  # 重要: 元のEVの利用可能性を渡す
                    }
                    
                    # 落札予定ブロックを決定
                    winning_blocks = [b for b in range(B) if enforce_blocks[b] == 1]
                    
                    # このEVのベースライン分担比率を計算
                    total_available_power = sum(
                        other_ev.p_charge_max for other_ev in evs if other_ev.available[0]
                    )
                    ev_contribution_ratio = ev.p_charge_max / total_available_power if total_available_power > 0 else 0.0
                    ev_baseline = [P_base[t] * ev_contribution_ratio for t in range(T)]
                    
                    task_args = (i, ev_data, mi, gp, c_r, c_e, c_p, T, B, gp.dt_min/60.0,
                               rho_r, rho_e, rho_p, baseline_mask_t, ev_baseline, 
                               enforce_baseline_for_reserve, winning_blocks, blocks,
                               time_limit_sec_local, msg_local)
                else:
                    # Use legacy solver
                    task_args = (i, ev, gp, mi, blocks, post_windows,
                               c_r, rho_r, c_e, rho_e, c_p, rho_p,
                               baseline_mask_t, P_base, enforce_baseline_for_reserve,
                               time_limit_sec_local, msg_local)
                
                tasks.append(task_args)
            
            # Execute tasks in parallel
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                future_to_ev_id = {}
                for i, task_args in enumerate(tasks):
                    if use_oop_solver:
                        future = executor.submit(solve_single_ev_parallel_oop, task_args)
                    else:
                        future = executor.submit(solve_local_ev_cut_with_baseline_L1, *task_args)
                    future_to_ev_id[future] = i
                
                if len(evs) > 10:
                    futures_list = list(future_to_ev_id.keys())
                    for future in tqdm(as_completed(futures_list), total=len(futures_list), 
                                     desc=f"  Solving EVs", leave=False, position=1):
                        i, res = future.result()
                        x_r[i] = res["r_cut"]
                        x_e[i] = res["e_rec"]
                        x_p[i] = res["p_ch"]
                        x_d[i] = res["p_dch"] if "p_dch" in res else [0.0]*T
                else:
                    for future in as_completed(future_to_ev_id):
                        i, res = future.result()
                        x_r[i] = res["r_cut"]
                        x_e[i] = res["e_rec"]
                        x_p[i] = res["p_ch"]
                        x_d[i] = res["p_dch"] if "p_dch" in res else [0.0]*T
        else:
            # Sequential execution
            ev_iter = enumerate(evs)
            if len(evs) > 10:
                ev_iter = tqdm(ev_iter, desc=f"  Solving EVs", leave=False, total=len(evs),
                              position=1, bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}')
            
            for i, ev in ev_iter:
                c_r = [s_r[i][b] - u_r[i][b] for b in range(B)]
                c_e = [s_e[i][b] - u_e[i][b] for b in range(B)]
                c_p = [s_p[i][t] - u_p[i][t] for t in range(T)]

                if use_oop_solver:
                    # Convert EVAgent to dict for OOP solver
                    ev_data = {
                        'id': ev.name,
                        'arrival': 0,
                        'departure': T,
                        'initial_soc': ev.soc0 * 100,  # Convert to %
                        'target_soc': ev.soc_max * 100,
                        'battery_capacity': ev.e_kwh,
                        'charge_power_max': ev.p_charge_max,
                        'discharge_power_max': ev.p_charge_max if ev.can_discharge_to_grid else 0,
                        'charge_efficiency': ev.eta_c,
                        'discharge_efficiency': ev.eta_d,
                        'can_discharge': ev.can_discharge_to_grid,
                        'available': ev.available  # 重要: 元のEVの利用可能性を渡す
                    }
                    
                    # 落札予定ブロックを決定（enforce_blocksの1が設定されているブロック）
                    winning_blocks = [b for b in range(B) if enforce_blocks[b] == 1]
                    
                    # このEVのベースライン分担比率を計算
                    total_available_power = sum(
                        other_ev.p_charge_max for other_ev in evs if other_ev.available[0]  # 簡略化: t=0での可用性で判定
                    )
                    ev_contribution_ratio = ev.p_charge_max / total_available_power if total_available_power > 0 else 0.0
                    
                    # このEV専用のベースライン目標を作成
                    ev_baseline = [P_base[t] * ev_contribution_ratio for t in range(T)]
                    
                    res = solve_ev_with_agent_methods(
                        ev_data, mi, gp, c_r, c_e, c_p, T, B, gp.dt_min/60.0,
                        rho_r, rho_e, rho_p, baseline_mask_t,
                        P_base=ev_baseline,  # EVごとの分担ベースラインを渡す
                        enforce_baseline_for_reserve=enforce_baseline_for_reserve,
                        winning_blocks=winning_blocks, blocks=blocks,  # 落札予定ブロック情報を渡す
                        time_limit_sec=time_limit_sec_local, msg=msg_local
                    )
                else:
                    if enforce_baseline_for_reserve:
                        # Use enhanced solver with baseline enforcement
                        res = solve_local_ev_cut_with_baseline_enforcement_L1(
                            ev, gp, mi, blocks, post_windows,
                            c_r, rho_r, c_e, rho_e, c_p, rho_p,
                            baseline_mask_t, P_base, enforce_baseline_for_reserve,
                            time_limit_sec=time_limit_sec_local, msg=msg_local
                        )
                    else:
                        # Use standard solver
                        res = solve_local_ev_cut_with_baseline_L1(
                            ev, gp, mi, blocks, post_windows,
                            c_r, rho_r, c_e, rho_e, c_p, rho_p,
                            baseline_mask_t,
                            time_limit_sec=time_limit_sec_local, msg=msg_local
                        )
                
                x_r[i] = res["r_cut"]
                x_e[i] = res["e_rec"]
                x_p[i] = res["p_ch"]
                x_d[i] = res["p_dch"] if "p_dch" in res else [0.0]*T

        # Rest of the ADMM algorithm remains the same...
        # (over-relaxation, s-update, u-update, residual calculation)
        # [The rest would be identical to the original function]
        
        # For now, break to show the structure
        if k >= 2:  # Just for testing
            break
    
    # Return results in same format as original
    return {
        "blocks": blocks,
        "sum_r_cut": [sum(x_r[i][b] for i in range(N)) for b in range(B)],
        "sum_e_rec": [sum(x_e[i][b] for i in range(N)) for b in range(B)],
        "sum_p_ch": [sum(x_p[i][t] for i in range(N)) for t in range(T)],
        "baseline_mask_t": baseline_mask_t,
        "individual_p_ch": x_p,
        "individual_p_dch": x_d,
        "history": hist,
        "status": "Test_OOP_Version"
    }


def solve_single_ev_parallel_oop(args):
    """Wrapper for parallel execution with OOP solver"""
    (i, ev_data, mi, gp, c_r, c_e, c_p, T, B, dt_hr, rho_r, rho_e, rho_p, 
     baseline_mask_t, P_base, enforce_baseline_for_reserve, winning_blocks, blocks,
     time_limit_sec_local, msg_local) = args
    
    res = solve_ev_with_agent_methods(
        ev_data, mi, gp, c_r, c_e, c_p, T, B, dt_hr,
        rho_r, rho_e, rho_p, baseline_mask_t,
        P_base=P_base, enforce_baseline_for_reserve=enforce_baseline_for_reserve,
        winning_blocks=winning_blocks, blocks=blocks,
        time_limit_sec=time_limit_sec_local, msg=msg_local
    )
    
    return i, res


def admm_cut_with_baseline(
    evs: List[EVAgent],
    gp: GlobalParams,
    mi: MarketInputs,
    R_bid_blk: List[float],     # target cut capacity per block [kW], only enforced where enforce_blocks[b]=1
    P_base: List[float],        # aggregate baseline [kW] per 5-min slot
    enforce_blocks: List[int],  # 0/1 per block: enable baseline + bid only in these blocks
    post_slots: int = 6,        # recovery window length in 5-min slots (6=30min)
    rho_r: float = 6.0,
    rho_e: float = 4.0,
    rho_p: float = 3.0,
    steps: int = 40,
    time_limit_sec_local: int = 6,
    msg_local: bool = False,
    over_relax: float = 1.2,
    num_threads: int = None,
    enforce_baseline_for_reserve: bool = False,  # Enable baseline-reserve coupling
) -> Dict:
    """
    ADMM coordinator with three sharings:
      Σ_i r_cut[i,b] = R_bid_eff[b]                      (only if enforce_blocks[b]=1 else =0)
      Σ_i e_rec[i,b]  = 0.5 * R_bid_eff[b]
      Σ_i p_ch[i,t]   = P_base[t]                        (only for t in enforced blocks)
    """
    T = len(mi.energy_buy_price_per_kwh)
    blocks = build_blocks(T, gp.dt_min, gp.sustain_min)
    B, N = len(blocks), len(evs)
    post_windows = build_post_windows(blocks, T, post_slots)

    baseline_mask_t = blocks_to_slot_mask(enforce_blocks, blocks, T)
    R_bid_eff = [R_bid_blk[b] if enforce_blocks[b] else 0.0 for b in range(B)]

    # variables per agent
    x_r = [[0.0]*B for _ in range(N)]  # local r_cut
    x_e = [[0.0]*B for _ in range(N)]  # local e_rec
    x_p = [[0.0]*T for _ in range(N)]  # local p_ch
    x_d = [[0.0]*T for _ in range(N)]  # local p_dch (discharge to grid)

    # s copies & duals
    s_r = [[0.0]*B for _ in range(N)]; u_r = [[0.0]*B for _ in range(N)]
    s_e = [[0.0]*B for _ in range(N)]; u_e = [[0.0]*B for _ in range(N)]
    s_p = [[0.0]*T for _ in range(N)]; u_p = [[0.0]*T for _ in range(N)]

    hist = []
    best = {"r_norm": 1e18}

    # Determine number of threads
    if num_threads is None:
        import os
        num_threads = min(N, os.cpu_count() or 1)
    
    # Progress bar for ADMM iterations
    pbar = tqdm(range(steps), desc=f"ADMM Optimization ({num_threads} threads)", leave=True, position=0,
                bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}')
    
    for k in pbar:
        # --- x-update: local MILPs (parallelizable) ---
        if num_threads > 1 and N > 1:
            # Parallel execution using ThreadPoolExecutor
            tasks = []
            for i, ev in enumerate(evs):
                c_r = [s_r[i][b] - u_r[i][b] for b in range(B)]
                c_e = [s_e[i][b] - u_e[i][b] for b in range(B)]
                c_p = [s_p[i][t] - u_p[i][t] for t in range(T)]
                
                task_args = (i, ev, gp, mi, blocks, post_windows,
                           c_r, rho_r, c_e, rho_e, c_p, rho_p,
                           baseline_mask_t, P_base, enforce_baseline_for_reserve,
                           time_limit_sec_local, msg_local)
                tasks.append(task_args)
            
            # Execute tasks in parallel
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                # Submit all tasks
                future_to_idx = {executor.submit(solve_single_ev_parallel, task): i for i, task in enumerate(tasks)}
                
                # Show progress if we have many EVs
                if len(evs) > 10:
                    futures_list = list(future_to_idx.keys())
                    for future in tqdm(as_completed(futures_list), total=len(futures_list), 
                                     desc=f"  Solving EVs", leave=False, position=1):
                        i, res = future.result()
                        x_r[i] = res["r_cut"]
                        x_e[i] = res["e_rec"]
                        x_p[i] = res["p_ch"]
                        x_d[i] = res["p_dch"] if "p_dch" in res else [0.0]*T
                else:
                    # No progress bar for few EVs
                    for future in as_completed(future_to_idx):
                        i, res = future.result()
                        x_r[i] = res["r_cut"]
                        x_e[i] = res["e_rec"]
                        x_p[i] = res["p_ch"]
                        x_d[i] = res["p_dch"] if "p_dch" in res else [0.0]*T
        else:
            # Sequential execution (fallback)
            ev_iter = enumerate(evs)
            if len(evs) > 10:
                ev_iter = tqdm(ev_iter, desc=f"  Solving EVs", leave=False, total=len(evs),
                              position=1, bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}')
            
            for i, ev in ev_iter:
                c_r = [s_r[i][b] - u_r[i][b] for b in range(B)]
                c_e = [s_e[i][b] - u_e[i][b] for b in range(B)]
                c_p = [s_p[i][t] - u_p[i][t] for t in range(T)]

                if enforce_baseline_for_reserve:
                    res = solve_local_ev_cut_with_baseline_enforcement_L1(
                        ev, gp, mi, blocks, post_windows,
                        c_r, rho_r, c_e, rho_e, c_p, rho_p,
                        baseline_mask_t, P_base, enforce_baseline_for_reserve,
                        time_limit_sec=time_limit_sec_local, msg=msg_local
                    )
                else:
                    res = solve_local_ev_cut_with_baseline_L1(
                        ev, gp, mi, blocks, post_windows,
                        c_r, rho_r, c_e, rho_e, c_p, rho_p,
                        baseline_mask_t,
                        time_limit_sec=time_limit_sec_local, msg=msg_local
                    )
                x_r[i] = res["r_cut"]
                x_e[i] = res["e_rec"]
                x_p[i] = res["p_ch"]
                x_d[i] = res["p_dch"] if "p_dch" in res else [0.0]*T

        # over-relax
        x_r_hat = [[over_relax*x_r[i][b] + (1-over_relax)*s_r[i][b] for b in range(B)] for i in range(N)]
        x_e_hat = [[over_relax*x_e[i][b] + (1-over_relax)*s_e[i][b] for b in range(B)] for i in range(N)]
        x_p_hat = [[over_relax*x_p[i][t] + (1-over_relax)*s_p[i][t] for t in range(T)] for i in range(N)]

        # --- s-update: projections ---
        # (1) Σ r_cut = R_bid_eff
        for b in range(B):
            total = sum(x_r_hat[i][b] + u_r[i][b] for i in range(N))
            delta = (R_bid_eff[b] - total) / N
            for i in range(N):
                s_r[i][b] = max(0.0, x_r_hat[i][b] + u_r[i][b] + delta)

        # (2) Σ e_rec = 0.5 * R_bid_eff
        for b in range(B):
            target = 0.5 * R_bid_eff[b]
            total = sum(x_e_hat[i][b] + u_e[i][b] for i in range(N))
            delta = (target - total) / N
            for i in range(N):
                s_e[i][b] = max(0.0, x_e_hat[i][b] + u_e[i][b] + delta)

        # (3) Baseline Σ p_ch = P_base  (only in enforced slots)
        for t in range(T):
            if baseline_mask_t[t] == 1:
                total = sum(x_p_hat[i][t] + u_p[i][t] for i in range(N))
                delta = (P_base[t] - total) / N
                for i in range(N):
                    s_p[i][t] = max(0.0, x_p_hat[i][t] + u_p[i][t] + delta)
            else:
                # no projection (keep nonnegative)
                for i in range(N):
                    s_p[i][t] = max(0.0, x_p_hat[i][t] + u_p[i][t])

        # --- u-update ---
        for i in range(N):
            for b in range(B):
                u_r[i][b] += x_r_hat[i][b] - s_r[i][b]
                u_e[i][b] += x_e_hat[i][b] - s_e[i][b]
            for t in range(T):
                u_p[i][t] += x_p_hat[i][t] - s_p[i][t]

        # --- residuals (primal) for logging ---
        sum_r = [sum(x_r[i][b] for i in range(N)) for b in range(B)]
        sum_e = [sum(x_e[i][b] for i in range(N)) for b in range(B)]
        sum_p = [sum(x_p[i][t] for i in range(N)) for t in range(T)]

        r1 = math.sqrt(sum((sum_r[b] - R_bid_eff[b])**2 for b in range(B)))
        r2 = math.sqrt(sum((sum_e[b] - 0.5*R_bid_eff[b])**2 for b in range(B)))
        r3 = math.sqrt(sum(((sum_p[t] - P_base[t]) if baseline_mask_t[t] else 0.0)**2 for t in range(T)))
        r_norm = r1 + r2 + r3  # simple aggregate

        hist.append({"iter": k, "r_cut_res": r1, "e_rec_res": r2, "base_res": r3, "r_norm": r_norm})
        if r_norm < best["r_norm"]:
            best = {"r_norm": r_norm, "sum_r": sum_r[:], "sum_e": sum_e[:], "sum_p": sum_p[:]}

        # Update progress bar with detailed residual information
        pbar.set_postfix_str(f"r_cut:{r1:.3f}, e_rec:{r2:.3f}, base:{r3:.3f}, total:{r_norm:.6f}")
        
        # early stop
        if r_norm < 1e-3:
            pbar.set_postfix_str(f"CONVERGED: total:{r_norm:.6f}")
            pbar.update(1)  # Ensure the final update is shown
            pbar.close()
            break
    else:
        # Loop completed without early termination
        pbar.close()

    return {
        "blocks": blocks,
        "sum_r_cut": [sum(x_r[i][b] for i in range(N)) for b in range(B)],
        "sum_e_rec": [sum(x_e[i][b] for i in range(N)) for b in range(B)],
        "sum_p_ch":  [sum(x_p[i][t] for i in range(N)) for t in range(T)],
        "baseline_mask_t": baseline_mask_t,
        "history": hist,
        "best": best,
        # Individual results
        "individual_r_cut": [[x_r[i][b] for b in range(B)] for i in range(N)],
        "individual_e_rec": [[x_e[i][b] for b in range(B)] for i in range(N)],
        "individual_p_ch":  [[x_p[i][t] for t in range(T)] for i in range(N)],
        "individual_p_dch": [[x_d[i][t] for t in range(T)] for i in range(N)],
    }
