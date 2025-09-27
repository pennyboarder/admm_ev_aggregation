# -*- coding: utf-8 -*-
"""
Utility functions for EV aggregation system
"""

from typing import List


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
