# Check EV capabilities
from demo import build_demo

def check_ev_capabilities():
    evs, gp, mi, R_bid, enforce_blocks, P_base = build_demo(n_evs=3, hours=4, dt_min=5, seed=1)
    
    print("EV Capabilities:")
    total_max_power = 0
    for i, ev in enumerate(evs):
        print(f"EV{i}: max_power={ev.p_charge_max:.1f}kW, capacity={ev.e_kwh:.1f}kWh, soc0={ev.soc0:.2f}")
        total_max_power += ev.p_charge_max
    
    print(f"\nTotal maximum charging power: {total_max_power:.1f}kW")
    print(f"Baseline requirement: {P_base[0]:.1f}kW")
    print(f"Can meet baseline? {'Yes' if total_max_power >= P_base[0] else 'No'}")
    
    # Check availability
    available_power_t0 = sum(ev.p_charge_max for ev in evs if ev.available[0])
    print(f"Available power at t=0: {available_power_t0:.1f}kW")

if __name__ == "__main__":
    check_ev_capabilities()
