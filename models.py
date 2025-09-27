# -*- coding: utf-8 -*-
"""
Data models for EV aggregation system with optimization capabilities
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import pulp
import math
import numpy as np


class EVAgent:
    """Electric Vehicle agent with charging characteristics and optimization capabilities"""
    
    def __init__(self, 
                 name: str,
                 e_kwh: float,
                 soc0: float,
                 soc_min: float,
                 soc_max: float,
                 p_charge_max: float,
                 eta_c: float,
                 available: List[int],
                 discharge_schedule: List[float] = None,
                 eta_d: float = 0.95,
                 can_discharge_to_grid: bool = False):
        """
        Initialize EV Agent
        
        Args:
            name: EV identifier
            e_kwh: Battery capacity [kWh]
            soc0: Initial State of Charge [0-1]
            soc_min: Minimum SOC constraint [0-1]
            soc_max: Maximum SOC constraint [0-1]
            p_charge_max: Maximum charging power [kW]
            eta_c: Charging efficiency [0-1]
            available: Availability schedule (0/1 per time slot)
            discharge_schedule: External discharge schedule [kWh per slot]
            eta_d: Discharging efficiency [0-1]
            can_discharge_to_grid: V2G capability flag
        """
        self.name = name
        self.e_kwh = e_kwh
        self.soc0 = soc0
        self.soc_min = soc_min
        self.soc_max = soc_max
        self.p_charge_max = p_charge_max
        self.eta_c = eta_c
        self.available = available
        self.discharge_schedule = discharge_schedule or [0.0] * len(available)
        self.eta_d = eta_d
        self.can_discharge_to_grid = can_discharge_to_grid
        
        # Optimization variables (will be set during optimization)
        self.variables: Dict = {}
        self.constraints: List = []
        
    def create_optimization_variables(self, T: int, B: int) -> Dict:
        """
        Create optimization variables for this EV
        
        Args:
            T: Number of time slots
            B: Number of blocks
            
        Returns:
            Dictionary of PuLP variables
        """
        variables = {}
        
        # Power variables
        variables['p_ch'] = pulp.LpVariable.dicts(f"{self.name}_p_ch", range(T), lowBound=0.0)
        if self.can_discharge_to_grid:
            variables['p_dch'] = pulp.LpVariable.dicts(f"{self.name}_p_dch", range(T), lowBound=0.0)
            # Binary variables for mutual exclusion
            variables['y_ch'] = pulp.LpVariable.dicts(f"{self.name}_y_ch", range(T), cat='Binary')
            variables['y_dch'] = pulp.LpVariable.dicts(f"{self.name}_y_dch", range(T), cat='Binary')
        
        # SOC variables
        variables['soc'] = pulp.LpVariable.dicts(f"{self.name}_soc", range(T), lowBound=0.0, upBound=1.0)
        
        # Reserve and recovery variables
        variables['r_cut'] = pulp.LpVariable.dicts(f"{self.name}_r_cut", range(B), lowBound=0.0)
        variables['e_rec'] = pulp.LpVariable.dicts(f"{self.name}_e_rec", range(B), lowBound=0.0)
        
        # L1 auxiliary variables
        variables['v_r'] = pulp.LpVariable.dicts(f"{self.name}_v_r", range(B), lowBound=0.0)
        variables['v_e'] = pulp.LpVariable.dicts(f"{self.name}_v_e", range(B), lowBound=0.0)
        variables['w_p'] = pulp.LpVariable.dicts(f"{self.name}_w_p", range(T), lowBound=0.0)
        
        self.variables = variables
        return variables
    
    def add_bidding_decision_variables(self, B: int) -> None:
        """
        Add binary variables for bidding decisions
        
        Args:
            B: Number of blocks
        """
        if not hasattr(self, 'variables') or not self.variables:
            raise ValueError("Must call create_optimization_variables first")
            
        variables = self.variables
        
        # Binary variables for bidding decisions (1 = bid in this block, 0 = don't bid)
        variables['bid_decision'] = pulp.LpVariable.dicts(
            f"{self.name}_bid_decision", range(B), cat='Binary'
        )
        
        return variables['bid_decision']
    
    def add_power_constraints(self, model: pulp.LpProblem, T: int) -> List:
        """
        Add power limit and mutual exclusion constraints
        
        Args:
            model: PuLP optimization model
            T: Number of time slots
            
        Returns:
            List of constraints added
        """
        constraints = []
        variables = self.variables
        
        for t in range(T):
            if self.can_discharge_to_grid:
                # V2G capable EV with mutual exclusion constraints
                # Power limits
                constraints.append(
                    variables['p_ch'][t] <= self.p_charge_max * self.available[t]
                )
                constraints.append(
                    variables['p_dch'][t] <= self.p_charge_max * self.available[t]
                )
                
                # Mutual exclusion constraint
                constraints.append(
                    variables['y_ch'][t] + variables['y_dch'][t] <= 1
                )
                
                # Link binary variables to power variables
                if self.available[t] > 0:
                    M = self.p_charge_max  # Big-M value
                    constraints.append(variables['p_ch'][t] <= M * variables['y_ch'][t])
                    constraints.append(variables['p_dch'][t] <= M * variables['y_dch'][t])
                else:
                    # If not available, force all to zero
                    constraints.append(variables['p_ch'][t] == 0)
                    constraints.append(variables['p_dch'][t] == 0)
                    constraints.append(variables['y_ch'][t] == 0)
                    constraints.append(variables['y_dch'][t] == 0)
            else:
                # Charge-only EV
                constraints.append(
                    variables['p_ch'][t] <= self.p_charge_max * self.available[t]
                )
        
        # Add constraints to model
        for constraint in constraints:
            model += constraint
            
        self.constraints.extend(constraints)
        return constraints
    
    def add_soc_constraints(self, model: pulp.LpProblem, T: int, dt_hr: float) -> List:
        """
        Add State of Charge dynamics and limit constraints
        
        Args:
            model: PuLP optimization model
            T: Number of time slots
            dt_hr: Time step in hours
            
        Returns:
            List of constraints added
        """
        constraints = []
        variables = self.variables
        
        # Initial SOC constraint
        charge_energy_0 = variables['p_ch'][0] * self.eta_c * dt_hr / self.e_kwh
        discharge_energy_0 = 0.0
        external_use_0 = 0.0
        
        if self.can_discharge_to_grid:
            discharge_energy_0 = variables['p_dch'][0] / self.eta_d * dt_hr / self.e_kwh
        
        if self.discharge_schedule:
            external_use_0 = self.discharge_schedule[0] / self.e_kwh
        
        constraints.append(
            variables['soc'][0] == self.soc0 + charge_energy_0 - discharge_energy_0 - external_use_0
        )
        constraints.append(variables['soc'][0] >= self.soc_min)
        constraints.append(variables['soc'][0] <= self.soc_max)
        
        # SOC evolution constraints
        for t in range(1, T):
            charge_energy = variables['p_ch'][t] * self.eta_c * dt_hr / self.e_kwh
            discharge_energy = 0.0
            external_use = 0.0
            
            if self.can_discharge_to_grid:
                discharge_energy = variables['p_dch'][t] / self.eta_d * dt_hr / self.e_kwh
            
            if self.discharge_schedule and t < len(self.discharge_schedule):
                external_use = self.discharge_schedule[t] / self.e_kwh
            
            constraints.append(
                variables['soc'][t] == variables['soc'][t-1] + charge_energy - discharge_energy - external_use
            )
            constraints.append(variables['soc'][t] >= self.soc_min)
            constraints.append(variables['soc'][t] <= self.soc_max)
        
        # Add constraints to model
        for constraint in constraints:
            model += constraint
            
        self.constraints.extend(constraints)
        return constraints
    
    def add_reserve_constraints(self, model: pulp.LpProblem, blocks: List[List[int]], 
                              post_windows: List[List[int]], dt_hr: float) -> List:
        """
        Add reserve capacity and recovery constraints
        
        Args:
            model: PuLP optimization model
            blocks: Time slot blocks
            post_windows: Post-event recovery windows
            dt_hr: Time step in hours
            
        Returns:
            List of constraints added
        """
        constraints = []
        variables = self.variables
        B = len(blocks)
        
        # Deliverability constraints
        for b, slots in enumerate(blocks):
            for t in slots:
                constraints.append(variables['r_cut'][b] <= variables['p_ch'][t])
        
        # Recovery headroom constraints
        for b, slots in enumerate(blocks):
            end_t = slots[-1]
            post = post_windows[b]
            
            if len(post) > 0:
                # Battery-side extra energy that can still be pushed in post window
                post_energy_headroom = pulp.lpSum(
                    (self.p_charge_max * self.available[tt] - variables['p_ch'][tt]) * dt_hr * self.eta_c 
                    for tt in post
                )
                constraints.append(variables['e_rec'][b] <= post_energy_headroom)
            
            # SOC ceiling driven limit (battery-side)
            constraints.append(
                variables['e_rec'][b] <= (self.soc_max - variables['soc'][end_t]) * self.e_kwh
            )
        
        # Add constraints to model
        for constraint in constraints:
            model += constraint
            
        self.constraints.extend(constraints)
        return constraints
    
    def add_l1_constraints(self, model: pulp.LpProblem, c_r: List[float], c_e: List[float], 
                          c_p: List[float], T: int, B: int) -> List:
        """
        Add L1 regularization constraints for ADMM
        
        Args:
            model: PuLP optimization model
            c_r, c_e, c_p: ADMM consensus variables
            T: Number of time slots
            B: Number of blocks
            
        Returns:
            List of constraints added
        """
        constraints = []
        variables = self.variables
        
        # L1 constraints for reserve and recovery
        for b in range(B):
            constraints.append(variables['r_cut'][b] - c_r[b] <= variables['v_r'][b])
            constraints.append(c_r[b] - variables['r_cut'][b] <= variables['v_r'][b])
            constraints.append(variables['e_rec'][b] - c_e[b] <= variables['v_e'][b])
            constraints.append(c_e[b] - variables['e_rec'][b] <= variables['v_e'][b])
        
        # L1 constraints for baseline tracking
        for t in range(T):
            constraints.append(variables['p_ch'][t] - c_p[t] <= variables['w_p'][t])
            constraints.append(c_p[t] - variables['p_ch'][t] <= variables['w_p'][t])
        
        # Add constraints to model
        for constraint in constraints:
            model += constraint
            
        self.constraints.extend(constraints)
        return constraints
    
    def add_baseline_constraints(self, model: pulp.LpProblem, 
                               P_base: List[float], 
                               baseline_mask_t: List[int],
                               T: int, B: int) -> List:
        """
        Add baseline constraints that link reserve provision to baseline adherence
        
        Args:
            model: PuLP optimization model
            P_base: Baseline power per time slot [kW]
            baseline_mask_t: Mask indicating which slots enforce baseline (0/1)
            T: Number of time slots
            B: Number of blocks
            
        Returns:
            List of constraints added
        """
        constraints = []
        variables = self.variables
        
        # No need for binary variables - we'll force exact baseline matching
        
        # For each enforced time slot, detect if power deviates from baseline
        epsilon = 0.001  # Small tolerance for exact matching
        for t in range(T):
            if baseline_mask_t[t] == 1:  # Only in enforced slots
                # Force exact baseline matching or set deviation flag
                M = max(self.p_charge_max, P_base[t]) + 1.0  # Big-M constant
                
                # Option 1: Force exact baseline matching
                constraints.append(variables['p_ch'][t] == P_base[t])
                
                # Option 2: Alternative - use auxiliary variables for deviation detection
                # Uncomment below if you want deviation detection instead of forced matching
                
                # # Detect positive deviation: p_ch[t] - P_base[t] <= M * baseline_dev[t] 
                # constraints.append(
                #     variables['p_ch'][t] - P_base[t] <= M * variables['baseline_dev'][t]
                # )
                # 
                # # Detect negative deviation: P_base[t] - p_ch[t] <= M * baseline_dev[t]
                # constraints.append(
                #     P_base[t] - variables['p_ch'][t] <= M * variables['baseline_dev'][t]
                # )
                # 
                # # If no deviation, baseline_dev[t] can be 0: 
                # # |p_ch[t] - P_base[t]| >= epsilon * baseline_dev[t]
                # constraints.append(
                #     variables['p_ch'][t] - P_base[t] >= -epsilon + epsilon * variables['baseline_dev'][t]
                # )
                # constraints.append(
                #     variables['p_ch'][t] - P_base[t] <= epsilon - epsilon * variables['baseline_dev'][t]
                # )
        
        # Since we force exact baseline matching in enforced slots,
        # no additional reserve restrictions are needed
        
        # Add constraints to model
        for constraint in constraints:
            model += constraint
            
        self.constraints.extend(constraints)
        return constraints
    
    def add_conditional_baseline_constraints(self, model: pulp.LpProblem, 
                                           P_base: List[float], 
                                           blocks: List[List[int]], 
                                           T: int, B: int) -> List:
        """
        Add baseline constraints that are activated only when bidding in a block
        
        Args:
            model: PuLP optimization model
            P_base: Baseline power per time slot [kW]
            blocks: Time slot groupings per block
            T: Number of time slots
            B: Number of blocks
            
        Returns:
            List of constraints added
        """
        constraints = []
        variables = self.variables
        
        if 'bid_decision' not in variables:
            raise ValueError("Must call add_bidding_decision_variables first")
        
        # For each block, if we decide to bid, then baseline must be followed
        for b in range(B):
            if b < len(blocks):
                block_slots = blocks[b]
                
                # Big-M constraint: if bid_decision[b] = 1, then p_ch[t] = P_base[t] for all t in block
                M = max(self.p_charge_max, max(P_base)) + 1.0  # Big-M constant
                
                for t in block_slots:
                    if t < T:
                        # If bid_decision[b] = 1, then |p_ch[t] - P_base[t]| ≤ 0 (i.e., equality)
                        # If bid_decision[b] = 0, then |p_ch[t] - P_base[t]| ≤ M (i.e., no constraint)
                        
                        constraints.append(
                            variables['p_ch'][t] - P_base[t] <= M * (1 - variables['bid_decision'][b])
                        )
                        constraints.append(
                            P_base[t] - variables['p_ch'][t] <= M * (1 - variables['bid_decision'][b])
                        )
        
        # Add constraints to model
        for constraint in constraints:
            model += constraint
            
        self.constraints.extend(constraints)
        return constraints
    
    def add_targeted_baseline_constraints(self, model: pulp.LpProblem,
                                         P_base: List[float],
                                         winning_blocks: List[int],
                                         blocks: List[List[int]],
                                         T: int) -> List:
        """
        Add baseline constraints focused on specific winning blocks from coordinator
        This forces the EV to concentrate charging to match baseline in targeted time periods
        
        Args:
            model: PuLP optimization model
            P_base: Baseline power per time slot [kW]
            winning_blocks: List of block indices where coordinator expects to win bids
            blocks: Time slot groupings per block  
            T: Number of time slots
            
        Returns:
            List of constraints added
        """
        constraints = []
        variables = self.variables
        
        # print(f"🎯 {self.name}: Adding targeted baseline constraints for winning blocks: {winning_blocks}")
        
        # For each winning block, enforce exact baseline matching
        for b in winning_blocks:
            if b < len(blocks):
                block_slots = blocks[b]
                
                for t in block_slots:
                    if t < T and t < len(P_base):
                        # Strong baseline matching constraint
                        # P_baseは既にこのEVの分担分が設定されている（coordinatorから渡される）
                        target_power = P_base[t]
                        
                        # Only set targets for available time slots
                        if self.available[t] == 0:
                            continue
                            
                        # Ensure target doesn't exceed EV capability
                        target_power = min(target_power, self.p_charge_max)
                        
                        # Add soft constraint with penalty for deviation
                        if target_power > 0.1:  # Only if meaningful target
                            constraints.append(
                                variables['p_ch'][t] >= target_power * 0.8  # At least 80% of target
                            )
                            constraints.append(
                                variables['p_ch'][t] <= target_power * 1.2  # At most 120% of target
                            )
                            
                            # print(f"   Slot {t}: targeting {target_power:.1f}kW (baseline: {P_base[t]:.1f}kW)")
        
        # Add constraints to model
        for constraint in constraints:
            model += constraint
            
        self.constraints.extend(constraints)
        return constraints
    
    def add_conditional_reserve_constraints(self, model: pulp.LpProblem, 
                                          R_max_per_block: List[float],
                                          B: int) -> List:
        """
        Add reserve constraints that are activated only when bidding
        
        Args:
            model: PuLP optimization model
            R_max_per_block: Maximum reserve capacity per block [kW]
            B: Number of blocks
            
        Returns:
            List of constraints added
        """
        constraints = []
        variables = self.variables
        
        if 'bid_decision' not in variables:
            raise ValueError("Must call add_bidding_decision_variables first")
        
        for b in range(B):
            # Reserve can only be provided if we decide to bid in this block
            # r_cut[b] ≤ R_max_per_block[b] * bid_decision[b]
            constraints.append(
                variables['r_cut'][b] <= R_max_per_block[b] * variables['bid_decision'][b]
            )
            
            # Recovery energy is also constrained by bidding decision
            constraints.append(
                variables['e_rec'][b] <= R_max_per_block[b] * 0.5 * variables['bid_decision'][b]
            )
        
        # Add constraints to model
        for constraint in constraints:
            model += constraint
            
        self.constraints.extend(constraints)
        return constraints
    
    def add_l2_approximation_variables(self, T: int, B: int, num_segments: int = 5) -> None:
        """
        Add variables for L2 norm approximation using piecewise linear approximation
        
        Args:
            T: Number of time slots
            B: Number of blocks  
            num_segments: Number of segments for piecewise linear approximation
        """
        if not hasattr(self, 'variables') or not self.variables:
            raise ValueError("Must create optimization variables first")
            
        variables = self.variables
        
        # L2 approximation variables for reserve capacity deviations
        variables['z_r'] = pulp.LpVariable.dicts(f"{self.name}_z_r", range(B), lowBound=0.0)
        variables['lambda_r'] = {}
        for b in range(B):
            variables['lambda_r'][b] = pulp.LpVariable.dicts(
                f"{self.name}_lambda_r_{b}", range(num_segments + 1), 
                lowBound=0.0, upBound=1.0
            )
        
        # L2 approximation variables for recovery energy deviations  
        variables['z_e'] = pulp.LpVariable.dicts(f"{self.name}_z_e", range(B), lowBound=0.0)
        variables['lambda_e'] = {}
        for b in range(B):
            variables['lambda_e'][b] = pulp.LpVariable.dicts(
                f"{self.name}_lambda_e_{b}", range(num_segments + 1),
                lowBound=0.0, upBound=1.0
            )
        
        # L2 approximation variables for power deviations
        variables['z_p'] = pulp.LpVariable.dicts(f"{self.name}_z_p", range(T), lowBound=0.0)
        variables['lambda_p'] = {}
        for t in range(T):
            variables['lambda_p'][t] = pulp.LpVariable.dicts(
                f"{self.name}_lambda_p_{t}", range(num_segments + 1),
                lowBound=0.0, upBound=1.0
            )
    
    def add_l2_approximation_constraints(self, model: pulp.LpProblem, c_r: List[float], 
                                       c_e: List[float], c_p: List[float], T: int, B: int,
                                       num_segments: int = 5, max_deviation: float = 50.0) -> List:
        """
        Add L2 norm approximation constraints using piecewise linear approximation
        
        Args:
            model: PuLP optimization model
            c_r, c_e, c_p: ADMM consensus variables
            T: Number of time slots
            B: Number of blocks
            num_segments: Number of segments for approximation
            max_deviation: Maximum expected deviation for scaling
            
        Returns:
            List of constraints added
        """
        constraints = []
        variables = self.variables
        
        # Create breakpoints for piecewise linear approximation of x^2
        # Estimate maximum deviation based on power and consensus variable ranges
        import numpy as np
        
        c_r_array = np.array(c_r) if not isinstance(c_r, np.ndarray) else c_r
        c_e_array = np.array(c_e) if not isinstance(c_e, np.ndarray) else c_e
        c_p_array = np.array(c_p) if not isinstance(c_p, np.ndarray) else c_p
        
        max_deviation = max(
            abs(self.p_charge_max),
            abs(np.max(c_p_array)) if c_p_array.size > 0 else 0,
            abs(np.max(c_r_array)) if c_r_array.size > 0 else 0,
            abs(np.max(c_e_array)) if c_e_array.size > 0 else 0,
            abs(np.min(c_p_array)) if c_p_array.size > 0 else 0,
            abs(np.min(c_r_array)) if c_r_array.size > 0 else 0,
            abs(np.min(c_e_array)) if c_e_array.size > 0 else 0
        )
        max_deviation = max(max_deviation, 1.0)  # Ensure at least 1.0
        
        breakpoints = [i * max_deviation / num_segments for i in range(num_segments + 1)]
        values = [x**2 for x in breakpoints]  # f(x) = x^2
        
        # L2 approximation for reserve capacity
        for b in range(B):
            # Absolute value constraint: z_r[b] >= |r_cut[b] - c_r[b]|
            constraints.append(variables['z_r'][b] >= variables['r_cut'][b] - c_r[b])
            constraints.append(variables['z_r'][b] >= c_r[b] - variables['r_cut'][b])
            
            # Piecewise linear approximation: z_r[b] = sum(lambda * breakpoint)
            constraints.append(
                variables['z_r'][b] == pulp.lpSum(
                    variables['lambda_r'][b][i] * breakpoints[i] 
                    for i in range(num_segments + 1)
                )
            )
            
            # SOS2 constraint (sum of lambda = 1, at most 2 adjacent non-zero)
            constraints.append(
                pulp.lpSum(variables['lambda_r'][b][i] for i in range(num_segments + 1)) == 1
            )
        
        # L2 approximation for recovery energy (similar structure)
        for b in range(B):
            constraints.append(variables['z_e'][b] >= variables['e_rec'][b] - c_e[b])
            constraints.append(variables['z_e'][b] >= c_e[b] - variables['e_rec'][b])
            
            constraints.append(
                variables['z_e'][b] == pulp.lpSum(
                    variables['lambda_e'][b][i] * breakpoints[i]
                    for i in range(num_segments + 1)
                )
            )
            
            constraints.append(
                pulp.lpSum(variables['lambda_e'][b][i] for i in range(num_segments + 1)) == 1
            )
        
        # L2 approximation for power baseline tracking
        for t in range(T):
            constraints.append(variables['z_p'][t] >= variables['p_ch'][t] - c_p[t])
            constraints.append(variables['z_p'][t] >= c_p[t] - variables['p_ch'][t])
            
            constraints.append(
                variables['z_p'][t] == pulp.lpSum(
                    variables['lambda_p'][t][i] * breakpoints[i]
                    for i in range(num_segments + 1)
                )
            )
            
            constraints.append(
                pulp.lpSum(variables['lambda_p'][t][i] for i in range(num_segments + 1)) == 1
            )
        
        # Add all constraints to model
        for constraint in constraints:
            model += constraint
            
        self.constraints.extend(constraints)
        return constraints
    
    def create_objective(self, mi, gp, T: int, B: int, dt_hr: float,
                        rho_r: float, rho_e: float, rho_p: float,
                        baseline_mask_t: List[int],
                        use_l2: bool = False, rho_r_l2: float = 0.0, 
                        rho_e_l2: float = 0.0, rho_p_l2: float = 0.0,
                        num_segments: int = 5) -> pulp.LpAffineExpression:
        """
        Create objective function for this EV
        
        Args:
            mi: Market inputs
            gp: Global parameters
            T: Number of time slots
            B: Number of blocks
            dt_hr: Time step in hours
            rho_r, rho_e, rho_p: ADMM penalty parameters
            baseline_mask_t: Baseline enforcement mask
            
        Returns:
            PuLP objective expression
        """
        variables = self.variables
        
        # Energy costs and revenues
        energy_buy_cost = pulp.lpSum(
            mi.energy_buy_price_per_kwh[t] * (variables['p_ch'][t] * dt_hr) 
            for t in range(T)
        )
        
        energy_sell_revenue = 0.0
        if self.can_discharge_to_grid:
            energy_sell_revenue = pulp.lpSum(
                mi.energy_buy_price_per_kwh[t] * (variables['p_dch'][t] * dt_hr) 
                for t in range(T)
            )
        
        # Degradation costs
        total_throughput = pulp.lpSum(variables['p_ch'][t] * dt_hr for t in range(T))
        if self.can_discharge_to_grid:
            total_throughput += pulp.lpSum(variables['p_dch'][t] * dt_hr for t in range(T))
        degr_cost = gp.degr_cost_per_kwh * total_throughput
        
        # Reserve capacity revenue
        cap_rev = pulp.lpSum(
            mi.cap_price_cut_kw_per_block[b] * variables['r_cut'][b] 
            for b in range(B)
        )
        
        # Bidding decision revenue/cost (if bidding variables exist)
        bidding_benefit = 0.0
        if 'bid_decision' in variables:
            # Revenue from successful bidding (capacity payment)
            bidding_benefit += pulp.lpSum(
                mi.cap_price_cut_kw_per_block[b] * variables['bid_decision'][b] * 0.1  # Small fixed benefit for bidding
                for b in range(B)
            )
            
            # Cost of baseline compliance (flexibility cost)
            baseline_compliance_cost = pulp.lpSum(
                variables['bid_decision'][b] * 1.0  # Small cost per block for compliance
                for b in range(B)
            )
        
        # ADMM L1 penalties - use first element if rho is array, otherwise scalar
        rho_r_val = rho_r[0] if isinstance(rho_r, (list, np.ndarray)) else rho_r
        rho_e_val = rho_e[0] if isinstance(rho_e, (list, np.ndarray)) else rho_e
        rho_p_val = rho_p[0] if isinstance(rho_p, (list, np.ndarray)) else rho_p
        
        l1_pen = (
            rho_r_val * pulp.lpSum(variables['v_r'][b] for b in range(B)) +
            rho_e_val * pulp.lpSum(variables['v_e'][b] for b in range(B)) +
            pulp.lpSum(rho_p_val * baseline_mask_t[t] * variables['w_p'][t] for t in range(T))
        )
        
        # ADMM L2 penalties (if enabled)
        l2_pen = 0.0
        if use_l2 and 'z_r' in variables:
            # Piecewise linear approximation of L2 norm
            breakpoints = [i * 50.0 / num_segments for i in range(num_segments + 1)]  # max_deviation = 50.0
            values = [x**2 for x in breakpoints]
            
            # L2 penalty for reserve capacity
            l2_pen += rho_r_l2 * pulp.lpSum(
                pulp.lpSum(variables['lambda_r'][b][i] * values[i] for i in range(num_segments + 1))
                for b in range(B)
            )
            
            # L2 penalty for recovery energy
            l2_pen += rho_e_l2 * pulp.lpSum(
                pulp.lpSum(variables['lambda_e'][b][i] * values[i] for i in range(num_segments + 1))
                for b in range(B)
            )
            
            # L2 penalty for power baseline
            l2_pen += rho_p_l2 * pulp.lpSum(
                pulp.lpSum(variables['lambda_p'][t][i] * values[i] * baseline_mask_t[t] 
                          for i in range(num_segments + 1))
                for t in range(T)
            )
        
        # Objective: maximize revenue - costs - L1 penalties - L2 penalties + bidding benefits
        if 'bid_decision' in variables:
            objective = cap_rev + energy_sell_revenue - energy_buy_cost - degr_cost - l1_pen - l2_pen + bidding_benefit - baseline_compliance_cost
        else:
            objective = cap_rev + energy_sell_revenue - energy_buy_cost - degr_cost - l1_pen - l2_pen
        
        return objective
    
    def extract_results(self) -> Dict:
        """
        Extract optimization results from solved variables
        
        Returns:
            Dictionary containing optimization results
        """
        if not self.variables:
            return {"status": "No variables"}
        
        variables = self.variables
        T = len(variables['p_ch'])
        B = len(variables['r_cut'])
        
        results = {
            "status": "Optimal",
            "r_cut": [max(0.0, pulp.value(variables['r_cut'][b]) or 0.0) for b in range(B)],
            "e_rec": [max(0.0, pulp.value(variables['e_rec'][b]) or 0.0) for b in range(B)],
            "p_ch": [max(0.0, pulp.value(variables['p_ch'][t]) or 0.0) for t in range(T)],
            "soc": [max(0.0, pulp.value(variables['soc'][t]) or 0.0) for t in range(T)]
        }
        
        if self.can_discharge_to_grid and 'p_dch' in variables:
            results["p_dch"] = [max(0.0, pulp.value(variables['p_dch'][t]) or 0.0) for t in range(T)]
        else:
            results["p_dch"] = [0.0] * T
            
        # Add bidding decision results if they exist
        if 'bid_decision' in variables:
            results["bid_decision"] = [
                int(pulp.value(variables['bid_decision'][b]) or 0) for b in range(B)
            ]
        
        return results
    
    def __str__(self) -> str:
        """String representation of EV agent"""
        return (f"EVAgent(name={self.name}, capacity={self.e_kwh}kWh, "
                f"max_power={self.p_charge_max}kW, V2G={self.can_discharge_to_grid})")
    
    def __repr__(self) -> str:
        """Detailed representation of EV agent"""
        return self.__str__()


@dataclass
class GlobalParams:
    """Global system parameters"""
    dt_min: int = 5
    sustain_min: int = 30
    degr_cost_per_kwh: float = 1.0


@dataclass
class MarketInputs:
    """Market pricing information"""
    # energy price [¥/kWh] per 5-min
    energy_buy_price_per_kwh: List[float]
    # capacity price for "charging stop up-reserve" [¥/kW per 30-min block]
    cap_price_cut_kw_per_block: List[float]
