# -*- coding: utf-8 -*-
"""
ADMM EV Aggregation Package

A distributed optimization system for electric vehicle charging coordination
with ancillary service provision using ADMM (Alternating Direction Method of Multipliers).
"""

from .models import EVAgent, GlobalParams, MarketInputs
from .admm_coordinator import admm_cut_with_baseline
from .local_solver import solve_local_ev_cut_with_baseline_L1
from .utils import build_blocks, blocks_to_slot_mask, build_post_windows
from .demo import build_demo
from .demo_discharge import build_demo_with_discharge
from .visualization import (
    plot_ev_charging_schedule,
    plot_soc_evolution, 
    plot_aggregate_analysis,
    plot_convergence_history,
    plot_ev_discharge_patterns,
    save_all_plots
)
from .csv_export import (
    export_time_series_csv,
    export_all_csv_data
)

__all__ = [
    'EVAgent',
    'GlobalParams', 
    'MarketInputs',
    'admm_cut_with_baseline',
    'solve_local_ev_cut_with_baseline_L1',
    'build_blocks',
    'blocks_to_slot_mask',
    'build_post_windows',
    'build_demo',
    'build_demo_with_discharge',
    'plot_ev_charging_schedule',
    'plot_soc_evolution',
    'plot_aggregate_analysis',
    'plot_convergence_history',
    'plot_ev_discharge_patterns',
    'save_all_plots',
    'export_time_series_csv',
    'export_all_csv_data'
]
