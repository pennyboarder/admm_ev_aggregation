# -*- coding: utf-8 -*-
"""
Enhanced demo with EV discharge (external use) scenarios
"""

from typing import List, Tuple
import random
import math
from ..core.models import EVAgent, GlobalParams, MarketInputs
from .utils import build_blocks


def build_demo_with_discharge(n_evs=10, hours=4, dt_min=5, seed=7, 
                             include_v2g=True, discharge_intensity=0.3) -> Tuple[List[EVAgent], GlobalParams, MarketInputs, List[float], List[int], List[float]]:
    """
    Build a demo scenario with EV discharge (external use) patterns
    
    Args:
        n_evs: Number of EVs
        hours: Simulation duration
        dt_min: Time resolution in minutes
        seed: Random seed
        include_v2g: Whether some EVs have V2G capability
        discharge_intensity: Intensity of external discharge events (0-1)
    """
    random.seed(seed)
    T = int(hours * 60 / dt_min)

    evs = []
    for i in range(n_evs):
        e_kwh = random.uniform(35, 55)
        soc0  = random.uniform(0.6, 0.9)  # Start with higher SOC for discharge scenarios
        eta   = 0.95
        pmax  = random.uniform(3.0, 7.0)

        # Availability pattern (still 0/1 for grid connection)
        avail = [0]*T
        for t in range(T):
            hh = (t*dt_min // 60) % 24
            # Day/night availability patterns
            if (i % 2 == 0 and 8 <= hh < 18) or (i % 2 == 1 and (hh >= 18 or hh < 8)):
                avail[t] = 1

        # Generate external discharge schedule (EV usage patterns)
        discharge_schedule = generate_discharge_schedule(T, dt_min, e_kwh, discharge_intensity, seed + i)
        
        # V2G capability (some EVs can sell back to grid)
        v2g_capable = include_v2g and (i % 3 == 0)  # Every 3rd EV has V2G

        evs.append(EVAgent(
            name=f"EV{i:02d}",
            e_kwh=e_kwh, soc0=soc0, soc_min=0.2, soc_max=0.9,
            p_charge_max=pmax, eta_c=eta, available=avail,
            discharge_schedule=discharge_schedule,
            eta_d=0.93,  # Slightly lower discharge efficiency
            can_discharge_to_grid=v2g_capable
        ))

    gp = GlobalParams(dt_min=dt_min, sustain_min=30, degr_cost_per_kwh=1.0)

    # Energy price: more dynamic pricing for V2G scenarios
    buy = []
    for t in range(T):
        hh = (t*dt_min // 60) % 24
        if 0 <= hh < 7:
            price = 20.0  # Night: cheap
        elif 7 <= hh < 10 or 17 <= hh < 20:
            price = 35.0  # Peak hours: expensive
        else:
            price = 28.0  # Day: moderate
        buy.append(price)

    blocks = build_blocks(T, dt_min, 30)
    B = len(blocks)
    cap_cut = [40.0]*B
    enforce_blocks = [0]*B
    
    # Set bidding blocks during peak hours
    for b, slots in enumerate(blocks):
        hh = (slots[0]*dt_min // 60) % 24
        if (7 <= hh < 10) or (17 <= hh < 20):  # Morning and evening peaks
            cap_cut[b] = 120.0
            enforce_blocks[b] = 1

    mi = MarketInputs(energy_buy_price_per_kwh=buy, cap_price_cut_kw_per_block=cap_cut)

    # R_bid for enforced blocks
    R_bid = [0.0]*B
    for b in range(B):
        if enforce_blocks[b]:
            R_bid[b] = 25.0  # Reserve capacity target

    # Baseline - higher during non-peak hours to allow for discharge
    P_base = [15.0]*T
    for b, slots in enumerate(blocks):
        if enforce_blocks[b]:
            for t in slots:
                P_base[t] = 40.0  # Higher baseline during peak hours

    return evs, gp, mi, R_bid, enforce_blocks, P_base


def generate_discharge_schedule(T: int, dt_min: int, e_kwh: float, intensity: float, seed: int) -> List[float]:
    """
    Generate realistic EV usage (discharge) patterns
    
    Args:
        T: Number of time slots
        dt_min: Time resolution in minutes
        e_kwh: EV battery capacity
        intensity: Discharge intensity factor (0-1)
        seed: Random seed for this EV
    
    Returns:
        List of discharge energy per time slot in kWh
    """
    random.seed(seed)
    dt_hr = dt_min / 60.0
    discharge = [0.0] * T
    
    if intensity <= 0:
        return discharge
    
    # Define typical usage patterns
    usage_patterns = [
        # Morning commute: 7-9 AM
        {"start_hour": 7, "duration_hours": 2, "intensity": 0.8, "energy_rate": 15.0},  # kWh/hr
        # Lunch trip: 12-13 PM  
        {"start_hour": 12, "duration_hours": 1, "intensity": 0.3, "energy_rate": 8.0},
        # Evening commute: 17-19 PM
        {"start_hour": 17, "duration_hours": 2, "intensity": 0.9, "energy_rate": 18.0},
        # Evening errands: 20-21 PM
        {"start_hour": 20, "duration_hours": 1, "intensity": 0.4, "energy_rate": 10.0},
    ]
    
    # Apply usage patterns with probability based on intensity
    for pattern in usage_patterns:
        if random.random() < intensity * pattern["intensity"]:
            start_slot = int(pattern["start_hour"] * 60 / dt_min)
            duration_slots = int(pattern["duration_hours"] * 60 / dt_min)
            
            for slot_offset in range(duration_slots):
                t = start_slot + slot_offset
                if t < T:
                    # Energy consumption during this slot
                    base_consumption = pattern["energy_rate"] * dt_hr
                    # Add some randomness
                    consumption = base_consumption * random.uniform(0.7, 1.3)
                    # Limit to reasonable values
                    max_consumption = e_kwh * 0.1  # Max 10% of battery per slot
                    discharge[t] = min(consumption, max_consumption)
    
    return discharge


# Enhanced version of original build_demo function
def build_demo(n_evs=10, hours=4, dt_min=5, seed=7) -> Tuple[List[EVAgent], GlobalParams, MarketInputs, List[float], List[int], List[float]]:
    """Original demo function enhanced with minimal discharge"""
    return build_demo_with_discharge(n_evs, hours, dt_min, seed, include_v2g=False, discharge_intensity=0.2)
