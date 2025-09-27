# ADMM EV Aggregation Package
"""
ADMM-based Electric Vehicle Aggregation System

A comprehensive system for optimizing electric vehicle charging/discharging
using Alternating Direction Method of Multipliers (ADMM) coordination.
"""

__version__ = "1.0.0"
__author__ = "ADMM EV Team"

# Core exports
from .core.models import EVAgent, GlobalParams, MarketInputs
from .solvers.admm_coordinator import admm_cut_with_baseline, admm_cut_with_baseline_oop
from .utils.utils import build_blocks, blocks_to_slot_mask, build_post_windows

__all__ = [
    "EVAgent", 
    "GlobalParams", 
    "MarketInputs",
    "admm_cut_with_baseline",
    "admm_cut_with_baseline_oop", 
    "build_blocks",
    "blocks_to_slot_mask",
    "build_post_windows"
]
