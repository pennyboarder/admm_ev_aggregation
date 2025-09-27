# -*- coding: utf-8 -*-
"""
Performance comparison between sequential and parallel ADMM processing
"""

import time
import os
from demo_discharge import build_demo_with_discharge
from admm_coordinator import admm_cut_with_baseline


def benchmark_parallel_performance():
    """Compare performance between sequential and parallel processing"""
    print("ADMM Parallel Processing Performance Benchmark")
    print("=" * 60)
    
    # Generate test data
    evs, gp, mi, R_bid, enforce_blocks, P_base = build_demo_with_discharge(
        n_evs=30, hours=8, dt_min=10, seed=42, include_v2g=True
    )
    
    print(f"Test dataset: {len(evs)} EVs, {8} hours, {gp.dt_min} min resolution")
    print(f"V2G capable EVs: {sum(1 for ev in evs if ev.can_discharge_to_grid)}")
    print(f"Available CPU cores: {os.cpu_count()}")
    print()
    
    # Test parameters
    test_params = {
        'post_slots': 3,
        'rho_r': 3.0,
        'rho_e': 3.0, 
        'rho_p': 2.0,
        'steps': 10,  # Fewer steps for faster testing
        'time_limit_sec_local': 2,
        'over_relax': 1.0
    }
    
    # Test 1: Sequential processing
    print("Test 1: Sequential Processing")
    print("-" * 30)
    start_time = time.time()
    
    result_seq = admm_cut_with_baseline(
        evs, gp, mi, R_bid, P_base, enforce_blocks,
        num_threads=1,  # Force sequential
        **test_params
    )
    
    seq_time = time.time() - start_time
    print(f"Sequential execution time: {seq_time:.2f} seconds")
    
    if result_seq["history"]:
        seq_residual = result_seq["history"][-1]["r_norm"]
        print(f"Final residual: {seq_residual:.6f}")
    print()
    
    # Test 2: Parallel processing (auto-detect cores)
    print("Test 2: Parallel Processing (Auto-detect)")
    print("-" * 40)
    start_time = time.time()
    
    result_par = admm_cut_with_baseline(
        evs, gp, mi, R_bid, P_base, enforce_blocks,
        num_threads=None,  # Auto-detect
        **test_params
    )
    
    par_time = time.time() - start_time
    print(f"Parallel execution time: {par_time:.2f} seconds")
    
    if result_par["history"]:
        par_residual = result_par["history"][-1]["r_norm"]
        print(f"Final residual: {par_residual:.6f}")
    print()
    
    # Test 3: Different thread counts
    thread_counts = [2, 4, 8]
    thread_results = []
    
    for threads in thread_counts:
        if threads <= os.cpu_count():
            print(f"Test 3.{threads}: {threads} Threads")
            print(f"-" * 20)
            start_time = time.time()
            
            result = admm_cut_with_baseline(
                evs, gp, mi, R_bid, P_base, enforce_blocks,
                num_threads=threads,
                **test_params
            )
            
            exec_time = time.time() - start_time
            residual = result["history"][-1]["r_norm"] if result["history"] else 0.0
            
            thread_results.append((threads, exec_time, residual))
            print(f"{threads}-thread execution time: {exec_time:.2f} seconds")
            print(f"Final residual: {residual:.6f}")
            print()
    
    # Performance summary
    print("PERFORMANCE SUMMARY")
    print("=" * 60)
    print(f"Sequential (1 thread):  {seq_time:.2f}s")
    print(f"Parallel (auto):        {par_time:.2f}s")
    
    speedup = seq_time / par_time if par_time > 0 else 0
    print(f"Speedup:                {speedup:.2f}x")
    
    if speedup > 1.0:
        efficiency = (speedup / os.cpu_count()) * 100
        print(f"Parallel efficiency:    {efficiency:.1f}%")
    
    print()
    print("Thread Count Comparison:")
    print("Threads | Time (s) | Speedup | Efficiency")
    print("-" * 40)
    print(f"   1    |  {seq_time:6.2f}  |   1.00x |    100.0%")
    
    for threads, exec_time, residual in thread_results:
        thread_speedup = seq_time / exec_time if exec_time > 0 else 0
        thread_efficiency = (thread_speedup / threads) * 100
        print(f"   {threads}    |  {exec_time:6.2f}  |  {thread_speedup:5.2f}x |   {thread_efficiency:5.1f}%")
    
    print()
    print("Optimization Quality (Final Residuals):")
    print(f"Sequential: {seq_residual:.6f}")
    print(f"Parallel:   {par_residual:.6f}")
    residual_diff = abs(seq_residual - par_residual) / seq_residual * 100
    print(f"Difference: {residual_diff:.3f}%")


if __name__ == "__main__":
    benchmark_parallel_performance()
