# -*- coding: utf-8 -*-
# ADMM for "provide up-reserve by STOPPING CHARGING" with a baseline
# - 5-min resolution scheduling
# - 30-min block-constant "cut" capacity r_cut[b] (kW): can always be delivered by stopping planned charging
# - Baseline tracking (aggregate) is enforced ONLY in bidding blocks
# - Recovery energy after the block (fleet can recharge later): aggregate equality Σ e_rec_i,b = 0.5 * R_bid[b]
# - ADMM uses L1 penalties (linear-friendly for PuLP/CBC)

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import math, random, pulp


# ---------------------------
# Data models
# ---------------------------
@dataclass
class EVAgent:
    name: str
    e_kwh: float
    soc0: float
    soc_min: float
    soc_max: float
    p_charge_max: float
    eta_c: float
    available: List[int]  # 0/1 per 5-min slot


@dataclass
class GlobalParams:
    dt_min: int = 5
    sustain_min: int = 30
    degr_cost_per_kwh: float = 1.0


@dataclass
class MarketInputs:
    # energy price [¥/kWh] per 5-min
    energy_buy_price_per_kwh: List[float]
    # capacity price for "charging stop up-reserve" [¥/kW per 30-min block]
    cap_price_cut_kw_per_block: List[float]


# ---------------------------
# Helpers
# ---------------------------
def build_blocks(T: int, dt_min: int, sustain_min: int = 30) -> List[List[int]]:
    """Non-overlapping blocks of sustain_min minutes."""
    W = (sustain_min + dt_min - 1) // dt_min
    blocks, t = [], 0
    while t < T:
        blocks.append(list(range(t, min(T, t + W))))
        t += W
    return blocks  # dt=5 -> 6 slots per 30-min block


def blocks_to_slot_mask(enforce_blocks: List[int], blocks: List[List[int]], T: int) -> List[int]:
    """Make a 0/1 mask over 5-min slots from 0/1 blocks."""
    mask = [0] * T
    for b, slots in enumerate(blocks):
        if enforce_blocks[b]:
            for t in slots:
                mask[t] = 1
    return mask


def build_post_windows(blocks: List[List[int]], T: int, post_slots: int) -> List[List[int]]:
    """For each block, return indices of the post-recovery window of length post_slots (clamped)."""
    post = []
    for slots in blocks:
        end = slots[-1]
        start = end + 1
        stop = min(T, start + post_slots)
        post.append(list(range(start, stop)))
    return post


# ---------------------------
# Local MILP (per EV): r_cut, e_rec, p_ch, SOC with baseline L1 and ADMM centers
# ---------------------------
def solve_local_ev_cut_with_baseline_L1(
    ev: EVAgent,
    gp: GlobalParams,
    mi: MarketInputs,
    blocks: List[List[int]],
    post_windows: List[List[int]],
    c_r: List[float],   # ADMM center for r_cut: s_r - u_r
    rho_r: float,
    c_e: List[float],   # ADMM center for e_rec: s_e - u_e
    rho_e: float,
    c_p: List[float],   # ADMM center for p_ch: s_p - u_p
    rho_p: float,
    baseline_mask_t: List[int],  # 0/1 mask (only in bidding blocks)
    time_limit_sec: int = 8,
    msg: bool = False,
) -> Dict:
    """
    Decision vars:
      p_ch[t] >=0  (charging power), SOC[t] ∈ [0,1]
      r_cut[b] >=0 (block-constant "cut" capacity by stopping charge)
      e_rec[b] >=0 (battery-side recovery energy in kWh after the block)
      L1 auxiliaries v_r[b] >= |r_cut[b] - c_r[b]|, v_e[b] >= |e_rec[b] - c_e[b]|,
                   w_p[t] >= |p_ch[t] - c_p[t]| * mask
    Constraints:
      deliverability in-block:   r_cut[b] <= p_ch[t]                   ∀t∈block b
      recovery headroom (post):  e_rec[b] <= Σ_{τ∈post_b} ( (pmax*avail - p_ch[τ]) * dt_hr * eta_c )
                                 e_rec[b] <= (soc_max - soc[end_b]) * E_kWh
    Objective:
      + cap_price[b]*r_cut[b] - energy_buy_cost - degradation
      - rho_r Σ v_r - rho_e Σ v_e - rho_p Σ mask_t * w_p
    """
    T = len(ev.available)
    dt_hr = gp.dt_min / 60.0
    B = len(blocks)

    m = pulp.LpProblem(f"EV_{ev.name}_cut_ADMM_L1", pulp.LpMaximize)

    # Variables
    p_ch = pulp.LpVariable.dicts("p_ch", range(T), lowBound=0.0)
    soc  = pulp.LpVariable.dicts("soc",  range(T), lowBound=0.0, upBound=1.0)

    r_cut = pulp.LpVariable.dicts("r_cut", range(B), lowBound=0.0)
    e_rec = pulp.LpVariable.dicts("e_rec", range(B), lowBound=0.0)

    # L1 auxiliaries
    v_r = pulp.LpVariable.dicts("v_r", range(B), lowBound=0.0)
    v_e = pulp.LpVariable.dicts("v_e", range(B), lowBound=0.0)
    w_p = pulp.LpVariable.dicts("w_p", range(T), lowBound=0.0)

    # Availability & per-slot charge limit
    for t in range(T):
        m += p_ch[t] <= ev.p_charge_max * ev.available[t]

    # SOC dynamics (no discharge in this minimal variant; easy to add if needed)
    m += soc[0] == ev.soc0 + (p_ch[0] * ev.eta_c * dt_hr / ev.e_kwh)
    m += soc[0] >= ev.soc_min
    m += soc[0] <= ev.soc_max
    for t in range(1, T):
        m += soc[t] == soc[t-1] + (p_ch[t] * ev.eta_c * dt_hr / ev.e_kwh)
        m += soc[t] >= ev.soc_min
        m += soc[t] <= ev.soc_max

    # Deliverability in block: can always stop at least r_cut[b] kW
    for b, slots in enumerate(blocks):
        for t in slots:
            m += r_cut[b] <= p_ch[t]

    # Recovery headroom after block + SOC ceiling at block end
    for b, slots in enumerate(blocks):
        end_t = slots[-1]
        post = post_windows[b]

        if len(post) > 0:
            # battery-side extra energy you can still push in post window
            post_energy_headroom = pulp.lpSum(
                (ev.p_charge_max * ev.available[tt] - p_ch[tt]) * dt_hr * ev.eta_c for tt in post
            )
            m += e_rec[b] <= post_energy_headroom
        # SOC ceiling driven limit (battery-side)
        m += e_rec[b] <= (ev.soc_max - soc[end_t]) * ev.e_kwh

        # L1 |r - c_r| <= v_r ; |e - c_e| <= v_e
        m +=  r_cut[b] - c_r[b] <= v_r[b]
        m +=  c_r[b] - r_cut[b] <= v_r[b]
        m +=  e_rec[b] - c_e[b] <= v_e[b]
        m +=  c_e[b] - e_rec[b] <= v_e[b]

    # Baseline L1 term per-slot (masked)
    for t in range(T):
        m +=  p_ch[t] - c_p[t] <= w_p[t]
        m +=  c_p[t] - p_ch[t] <= w_p[t]

    # Objective
    cap = mi.cap_price_cut_kw_per_block
    buy = mi.energy_buy_price_per_kwh
    energy_buy_cost = pulp.lpSum(buy[t] * (p_ch[t] * dt_hr) for t in range(T))
    degr_cost = gp.degr_cost_per_kwh * pulp.lpSum((p_ch[t] * dt_hr) for t in range(T))

    cap_rev = pulp.lpSum(cap[b] * r_cut[b] for b in range(B))

    l1_pen = (
        rho_r * pulp.lpSum(v_r[b] for b in range(B)) +
        rho_e * pulp.lpSum(v_e[b] for b in range(B)) +
        pulp.lpSum(rho_p * baseline_mask_t[t] * w_p[t] for t in range(T))
    )

    m += cap_rev - energy_buy_cost - degr_cost - l1_pen

    solver = pulp.PULP_CBC_CMD(msg=msg, timeLimit=time_limit_sec)
    m.solve(solver)

    return {
        "status": pulp.LpStatus[m.status],
        "r_cut": [max(0.0, pulp.value(r_cut[b]) or 0.0) for b in range(B)],
        "e_rec": [max(0.0, pulp.value(e_rec[b]) or 0.0) for b in range(B)],
        "p_ch":  [max(0.0, pulp.value(p_ch[t])  or 0.0) for t in range(T)],
    }


# ---------------------------
# ADMM coordinator (three sharings): r_cut sum, e_rec sum, baseline per-slot
# ---------------------------
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
) -> Dict:
    """
    Sharings:
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

    # s copies & duals
    s_r = [[0.0]*B for _ in range(N)]; u_r = [[0.0]*B for _ in range(N)]
    s_e = [[0.0]*B for _ in range(N)]; u_e = [[0.0]*B for _ in range(N)]
    s_p = [[0.0]*T for _ in range(N)]; u_p = [[0.0]*T for _ in range(N)]

    hist = []
    best = {"r_norm": 1e18}

    for k in range(steps):
        # --- x-update: local MILPs (parallelizable) ---
        for i, ev in enumerate(evs):
            c_r = [s_r[i][b] - u_r[i][b] for b in range(B)]
            c_e = [s_e[i][b] - u_e[i][b] for b in range(B)]
            c_p = [s_p[i][t] - u_p[i][t] for t in range(T)]

            res = solve_local_ev_cut_with_baseline_L1(
                ev, gp, mi, blocks, post_windows,
                c_r, rho_r, c_e, rho_e, c_p, rho_p,
                baseline_mask_t,
                time_limit_sec=time_limit_sec_local, msg=msg_local
            )
            x_r[i] = res["r_cut"]
            x_e[i] = res["e_rec"]
            x_p[i] = res["p_ch"]

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

        # early stop
        if r_norm < 1e-3:
            break

    return {
        "blocks": blocks,
        "sum_r_cut": [sum(x_r[i][b] for i in range(N)) for b in range(B)],
        "sum_e_rec": [sum(x_e[i][b] for i in range(N)) for b in range(B)],
        "sum_p_ch":  [sum(x_p[i][t] for i in range(N)) for t in range(T)],
        "baseline_mask_t": baseline_mask_t,
        "history": hist,
        "best": best,
    }


# ---------------------------
# Demo builder
# ---------------------------
def build_demo(n_evs=10, hours=4, dt_min=5, seed=7) -> Tuple[List[EVAgent], GlobalParams, MarketInputs, List[float], List[int], List[float]]:
    random.seed(seed)
    T = int(hours * 60 / dt_min)

    evs = []
    for i in range(n_evs):
        e_kwh = random.uniform(35, 55)
        soc0  = random.uniform(0.5, 0.8)
        eta   = 0.95
        pmax  = random.uniform(3.0, 7.0)

        avail = [0]*T
        for t in range(T):
            hh = (t*dt_min // 60) % 24
            # half day / half night availability for variety
            if (i % 2 == 0 and 8 <= hh < 18) or (i % 2 == 1 and (hh >= 18 or hh < 8)):
                avail[t] = 1

        evs.append(EVAgent(
            name=f"EV{i:02d}",
            e_kwh=e_kwh, soc0=soc0, soc_min=0.2, soc_max=0.9,
            p_charge_max=pmax, eta_c=eta, available=avail
        ))

    gp = GlobalParams(dt_min=dt_min, sustain_min=30, degr_cost_per_kwh=1.0)

    # energy price: cheap night, expensive day
    buy = []
    for t in range(T):
        hh = (t*dt_min // 60) % 24
        buy.append(20.0 if 0 <= hh < 7 else 28.0)

    blocks = build_blocks(T, dt_min, 30)
    B = len(blocks)
    cap_cut = [40.0]*B
    enforce_blocks = [0]*B
    # set a couple of "bidding" blocks in the evening
    for b, slots in enumerate(blocks):
        hh = (slots[0]*dt_min // 60) % 24
        if 19 <= hh < 21:
            cap_cut[b] = 120.0
            enforce_blocks[b] = 1

    mi = MarketInputs(energy_buy_price_per_kwh=buy, cap_price_cut_kw_per_block=cap_cut)

    # R_bid only for enforced blocks
    R_bid = [0.0]*B
    for b in range(B):
        if enforce_blocks[b]:
            R_bid[b] = 25.0  # try to cut 25 kW by stopping charge

    # Baseline [kW] only matters during enforced blocks; simple flat baseline here
    P_base = [10.0]*T
    for b, slots in enumerate(blocks):
        if enforce_blocks[b]:
            for t in slots:
                P_base[t] = 35.0  # aggregate planned charging; can be slashed on activation

    return evs, gp, mi, R_bid, enforce_blocks, P_base


# ---------------------------
# Main (demo)
# ---------------------------
if __name__ == "__main__":
    evs, gp, mi, R_bid, enforce_blocks, P_base = build_demo(n_evs=12, hours=4, dt_min=5, seed=3)
    out = admm_cut_with_baseline(
        evs, gp, mi, R_bid, P_base, enforce_blocks,
        post_slots=6, rho_r=6.0, rho_e=4.0, rho_p=3.0,
        steps=30, time_limit_sec_local=6, over_relax=1.25
    )
    print("Enforced blocks:", enforce_blocks)
    print("Sum r_cut [kW]: ", [round(x,1) for x in out["sum_r_cut"]])
    print("Sum e_rec [kWh]:", [round(x,1) for x in out["sum_e_rec"]])
    # show first 12 slots of aggregate baseline tracking
    agg_p = out["sum_p_ch"]
    mask  = out["baseline_mask_t"]
    first_masked = [i for i,m in enumerate(mask) if m==1][:12]
    print("First masked slots (t idx):", first_masked)
    if first_masked:
        t0 = first_masked[0]
        print("Agg p_ch @ masked:", [round(agg_p[t],1) for t in range(t0, min(len(agg_p), t0+12))])
