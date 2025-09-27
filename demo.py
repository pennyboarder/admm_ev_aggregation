# -*- coding: utf-8 -*-
"""
Demo data generator for EV aggregation system
"""

from typing import List, Tuple
import random
from models import EVAgent, GlobalParams, MarketInputs
from utils import build_blocks


def build_demo(n_evs=10, hours=4, dt_min=5, seed=7) -> Tuple[List[EVAgent], GlobalParams, MarketInputs, List[float], List[int], List[float]]:
    """
    Build a demo scenario with random EVs and market conditions
    
    Returns:
        evs: List of EV agents
        gp: Global parameters
        mi: Market inputs
        R_bid: Bid capacity per block
        enforce_blocks: Which blocks have baseline enforcement
        P_base: Baseline power per time slot
    """
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
            p_charge_max=pmax, eta_c=eta, available=avail,
            discharge_schedule=None, eta_d=0.95, can_discharge_to_grid=False
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
