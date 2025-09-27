# -*- coding: utf-8 -*-
"""
Local optimization problem solver for individual EVs
"""

from typing import List, Dict
import pulp
from models import EVAgent, GlobalParams, MarketInputs


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
    p_dch = pulp.LpVariable.dicts("p_dch", range(T), lowBound=0.0) if ev.can_discharge_to_grid else None
    soc  = pulp.LpVariable.dicts("soc",  range(T), lowBound=0.0, upBound=1.0)

    # Binary variables for mutual exclusion constraint (charge vs discharge)
    y_ch = pulp.LpVariable.dicts("y_ch", range(T), cat='Binary') if ev.can_discharge_to_grid else None
    y_dch = pulp.LpVariable.dicts("y_dch", range(T), cat='Binary') if ev.can_discharge_to_grid else None

    r_cut = pulp.LpVariable.dicts("r_cut", range(B), lowBound=0.0)
    e_rec = pulp.LpVariable.dicts("e_rec", range(B), lowBound=0.0)

    # L1 auxiliaries
    v_r = pulp.LpVariable.dicts("v_r", range(B), lowBound=0.0)
    v_e = pulp.LpVariable.dicts("v_e", range(B), lowBound=0.0)
    w_p = pulp.LpVariable.dicts("w_p", range(T), lowBound=0.0)

    # Availability & per-slot charge/discharge limits
    for t in range(T):
        if p_dch is not None:  # V2G capability - use binary variables for mutual exclusion
            # Power limits with availability
            m += p_ch[t] <= ev.p_charge_max * ev.available[t]
            m += p_dch[t] <= ev.p_charge_max * ev.available[t]
            
            # Mutual exclusion constraint: cannot charge and discharge simultaneously
            m += y_ch[t] + y_dch[t] <= 1  # At most one can be active
            
            # Link binary variables to power variables
            # If y_ch[t] = 0, then p_ch[t] must be 0
            # If y_dch[t] = 0, then p_dch[t] must be 0
            M = ev.p_charge_max  # Big-M value
            if ev.available[t] > 0:
                m += p_ch[t] <= M * y_ch[t]
                m += p_dch[t] <= M * y_dch[t]
            else:
                # If not available, force both to zero
                m += p_ch[t] == 0
                m += p_dch[t] == 0
                m += y_ch[t] == 0
                m += y_dch[t] == 0
        else:
            # Charge-only EV - standard constraint
            m += p_ch[t] <= ev.p_charge_max * ev.available[t]

    # SOC dynamics with charging, discharging, and external use
    # Initial SOC
    charge_energy_0 = p_ch[0] * ev.eta_c * dt_hr / ev.e_kwh
    discharge_energy_0 = (p_dch[0] / ev.eta_d * dt_hr / ev.e_kwh) if p_dch else 0
    external_use_0 = (ev.discharge_schedule[0] / ev.e_kwh) if ev.discharge_schedule else 0
    
    m += soc[0] == ev.soc0 + charge_energy_0 - discharge_energy_0 - external_use_0
    m += soc[0] >= ev.soc_min
    m += soc[0] <= ev.soc_max
    
    # SOC evolution
    for t in range(1, T):
        charge_energy = p_ch[t] * ev.eta_c * dt_hr / ev.e_kwh
        discharge_energy = (p_dch[t] / ev.eta_d * dt_hr / ev.e_kwh) if p_dch else 0
        external_use = (ev.discharge_schedule[t] / ev.e_kwh) if ev.discharge_schedule and t < len(ev.discharge_schedule) else 0
        
        m += soc[t] == soc[t-1] + charge_energy - discharge_energy - external_use
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
    
    # Energy costs and revenues
    energy_buy_cost = pulp.lpSum(buy[t] * (p_ch[t] * dt_hr) for t in range(T))
    energy_sell_revenue = pulp.lpSum(buy[t] * (p_dch[t] * dt_hr) for t in range(T)) if p_dch else 0
    
    # Degradation costs (both charging and discharging cause degradation)
    total_throughput = pulp.lpSum((p_ch[t] + (p_dch[t] if p_dch else 0)) * dt_hr for t in range(T))
    degr_cost = gp.degr_cost_per_kwh * total_throughput

    cap_rev = pulp.lpSum(cap[b] * r_cut[b] for b in range(B))

    l1_pen = (
        rho_r * pulp.lpSum(v_r[b] for b in range(B)) +
        rho_e * pulp.lpSum(v_e[b] for b in range(B)) +
        pulp.lpSum(rho_p * baseline_mask_t[t] * w_p[t] for t in range(T))
    )

    m += cap_rev + energy_sell_revenue - energy_buy_cost - degr_cost - l1_pen

    # solver = pulp.PULP_CBC_CMD(msg=msg, timeLimit=time_limit_sec)
    solver = pulp.SCIP_CMD(msg=msg, timeLimit=time_limit_sec)
    m.solve(solver)

    return {
        "status": pulp.LpStatus[m.status],
        "r_cut": [max(0.0, pulp.value(r_cut[b]) or 0.0) for b in range(B)],
        "e_rec": [max(0.0, pulp.value(e_rec[b]) or 0.0) for b in range(B)],
        "p_ch":  [max(0.0, pulp.value(p_ch[t])  or 0.0) for t in range(T)],
        "p_dch": [max(0.0, pulp.value(p_dch[t]) or 0.0) for t in range(T)] if p_dch else [0.0] * T,
    }


def solve_local_ev_cut_with_baseline_enforcement_L1(
    ev: 'EVAgent',
    gp: 'GlobalParams',
    mi: 'MarketInputs',
    blocks: List[List[int]],
    post_windows: List[List[int]], 
    c_r: List[float],
    rho_r: float,
    c_e: List[float],
    rho_e: float,
    c_p: List[float],
    rho_p: float,
    baseline_mask_t: List[int],
    P_base: List[float],
    enforce_baseline_for_reserve: bool = True,
    time_limit_sec: int = 6,
    msg: bool = False
) -> Dict:
    """
    Enhanced EV optimization with baseline enforcement for reserve provision
    
    If enforce_baseline_for_reserve is True:
    - Reserve capacity can only be provided if aggregate charging follows baseline
    - Any deviation from baseline in enforced slots prevents reserve provision
    
    Args:
        ev: EV agent
        gp: Global parameters  
        mi: Market inputs
        blocks: Power blocks
        post_windows: Recovery windows
        c_r, c_e, c_p: ADMM consensus variables
        rho_r, rho_e, rho_p: ADMM penalty parameters
        baseline_mask_t: Baseline enforcement mask (0/1 per slot)
        P_base: Baseline power per slot [kW]
        enforce_baseline_for_reserve: Enable baseline-reserve coupling
        time_limit_sec: Solver time limit
        msg: Solver verbosity
        
    Returns:
        Dictionary with optimization results
    """
    # Use the standard solver but add baseline constraints
    result = solve_local_ev_cut_with_baseline_L1(
        ev, gp, mi, blocks, post_windows,
        c_r, rho_r, c_e, rho_e, c_p, rho_p,
        baseline_mask_t, time_limit_sec, msg
    )
    
    # If baseline enforcement is disabled, return standard result
    if not enforce_baseline_for_reserve:
        return result
        
    # Check baseline adherence in enforced slots
    T = len(ev.available)
    baseline_violation = False
    
    for t in range(T):
        if baseline_mask_t[t] == 1:  # Enforced slot
            if abs(result['p_ch'][t] - P_base[t]) > 1e-6:  # Tolerance for numerical errors
                baseline_violation = True
                break
    
    # If baseline is violated, force all reserve cuts to zero
    if baseline_violation:
        result['r_cut'] = [0.0] * len(result['r_cut'])
        result['baseline_violation'] = True
    else:
        result['baseline_violation'] = False
        
    return result
