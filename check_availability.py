# Check EV availability patterns
from demo import build_demo

def check_ev_availability():
    evs, gp, mi, R_bid, enforce_blocks, P_base = build_demo(n_evs=3, hours=4, dt_min=5, seed=1)
    
    print("EV Availability Patterns:")
    for i, ev in enumerate(evs):
        available_slots = sum(ev.available)
        print(f"EV{i}: max_power={ev.p_charge_max:.1f}kW, available_slots={available_slots}/{len(ev.available)}")
        # Show first 24 slots
        avail_str = ''.join(['1' if av else '0' for av in ev.available[:24]])
        print(f"  First 24 slots: {avail_str}")
    
    # Calculate available power per time slot
    print("\nAvailable power per time slot (first 12):")
    for t in range(12):
        available_power = sum(ev.p_charge_max for ev in evs if ev.available[t])
        print(f"  t={t}: {available_power:.1f}kW (baseline: {P_base[t]:.1f}kW)")
        if available_power < P_base[t]:
            print(f"    ⚠️  INSUFFICIENT POWER!")

if __name__ == "__main__":
    check_ev_availability()
