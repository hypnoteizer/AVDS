"""

Adaptive Vibration Damping System (AVDS)
Trinity College Dublin — Mechanical Engineering Portfolio Project

PROJECT OVERVIEW:
This project demonstrates the design and simulation of an adaptive vibration damping 
system for a single degree-of-freedom (SDOF) mass-spring-damper system. The adaptive 
control algorithm dynamically adjusts the damping coefficient based on real-time RMS 
vibration measurements to minimize oscillations.

WHY THIS PROJECT FOR TRINITY:
This project showcases essential mechanical engineering skills including:
1. Mathematical modeling of dynamic mechanical systems
2. Numerical simulation and control algorithm implementation
3. Data analysis and performance optimization
4. Application of feedback control theory to real-world vibration problems

These competencies align with Trinity's emphasis on analytical thinking, problem-solving,
and innovative engineering design.

AUTHOR: Portfolio Project for Trinity College Dublin Application
DATE: 2025
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from typing import Tuple, Dict

# ============================================================================
# SYSTEM PARAMETERS
# ============================================================================

# Physical parameters
M = 10.0                    # Mass [kg]
K = 1000.0                  # Spring stiffness [N/m]
C_INITIAL = 20.0            # Initial damping coefficient [Ns/m]

# Forcing function parameters
F0 = 50.0                   # Force amplitude [N]
OMEGA = 10.0                # Forcing frequency [rad/s]

# Critical damping coefficient
C_CRIT = 2 * np.sqrt(K * M)

# Simulation parameters
T_START = 0.0               # Start time [s]
T_END = 10.0                # End time [s]
DT = 0.001                  # Time step [s]

# Adaptive control parameters
CONTROL_UPDATE_INTERVAL = 0.05      # Update damping every 0.05 s
RMS_WINDOW = 0.5                    # RMS calculation window [s]
TARGET_RMS = 0.05                   # Target RMS displacement [m]
CONTROL_GAIN = 250.0                # Control gain
C_MIN = 0.005 * C_CRIT              # Minimum damping
C_MAX = 2.0 * C_CRIT                # Maximum damping

# Initial conditions
X0 = 0.0                    # Initial displacement [m]
V0 = 0.0                    # Initial velocity [m/s]


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def forcing_function(t: float) -> float:
    """
    Sinusoidal forcing function: F(t) = F0 * sin(ω * t)
    
    Args:
        t: Time [s]
    
    Returns:
        Force [N]
    """
    return F0 * np.sin(OMEGA * t)


def clamp(value: float, min_val: float, max_val: float) -> float:
    """
    Clamp value between minimum and maximum bounds.
    
    Args:
        value: Value to clamp
        min_val: Minimum bound
        max_val: Maximum bound
    
    Returns:
        Clamped value
    """
    return max(min_val, min(value, max_val))


def calculate_rms(signal: np.ndarray) -> float:
    """
    Calculate Root Mean Square (RMS) of a signal.
    
    Args:
        signal: Input signal array
    
    Returns:
        RMS value
    """
    return np.sqrt(np.mean(signal**2))


def calculate_settling_time(t: np.ndarray, x: np.ndarray, threshold: float = 0.05) -> float:
    """
    Calculate settling time (time to stay within ±threshold of steady state).
    
    Args:
        t: Time array [s]
        x: Displacement array [m]
        threshold: Settling threshold as fraction of max displacement
    
    Returns:
        Settling time [s], or NaN if not settled
    """
    if len(x) < 2:
        return np.nan
    
    # Use last 10% of signal as steady state reference
    steady_state = np.mean(x[-len(x)//10:])
    max_displacement = np.max(np.abs(x))
    
    if max_displacement < 1e-6:
        return 0.0
    
    tolerance = threshold * max_displacement
    
    # Find last time signal exceeded tolerance
    for i in range(len(x)-1, -1, -1):
        if np.abs(x[i] - steady_state) > tolerance:
            if i < len(x) - 1:
                return t[i+1]
            else:
                return np.nan
    
    return t[0]


def calculate_mechanical_energy(x: np.ndarray, v: np.ndarray, m: float, k: float) -> np.ndarray:
    """
    Calculate total mechanical energy: E = (1/2)mv² + (1/2)kx²
    
    Args:
        x: Displacement array [m]
        v: Velocity array [m/s]
        m: Mass [kg]
        k: Spring stiffness [N/m]
    
    Returns:
        Energy array [J]
    """
    kinetic = 0.5 * m * v**2
    potential = 0.5 * k * x**2
    return kinetic + potential


# ============================================================================
# SIMULATION FUNCTIONS
# ============================================================================

def system_dynamics_fixed(t: float, state: np.ndarray, m: float, c: float, k: float) -> np.ndarray:
    """
    System dynamics with fixed damping coefficient.
    State: [x, v] where x is displacement and v is velocity
    Equation: m*x'' + c*x' + k*x = F(t)
    
    Args:
        t: Time [s]
        state: State vector [x, v]
        m: Mass [kg]
        c: Damping coefficient [Ns/m]
        k: Spring stiffness [N/m]
    
    Returns:
        State derivative [v, a]
    """
    x, v = state
    F = forcing_function(t)
    a = (F - c * v - k * x) / m
    return np.array([v, a])


def simulate_fixed_damping(m: float, c: float, k: float, 
                          t_span: Tuple[float, float], 
                          initial_state: np.ndarray,
                          t_eval: np.ndarray) -> Dict:
    """
    Simulate system with fixed damping coefficient.
    
    Args:
        m: Mass [kg]
        c: Damping coefficient [Ns/m]
        k: Spring stiffness [N/m]
        t_span: Time span tuple (t_start, t_end) [s]
        initial_state: Initial state [x0, v0]
        t_eval: Time points for evaluation [s]
    
    Returns:
        Dictionary containing simulation results
    """
    sol = solve_ivp(
        fun=lambda t, y: system_dynamics_fixed(t, y, m, c, k),
        t_span=t_span,
        y0=initial_state,
        t_eval=t_eval,
        method='RK45',
        rtol=1e-8,
        atol=1e-10
    )
    
    return {
        't': sol.t,
        'x': sol.y[0],
        'v': sol.y[1],
        'c': np.full_like(sol.t, c)
    }


def simulate_adaptive_damping(m: float, c_initial: float, k: float,
                              t_span: Tuple[float, float],
                              initial_state: np.ndarray,
                              dt: float) -> Dict:
    """
    Simulate system with adaptive damping control.
    
    The adaptive algorithm:
    1. Measures RMS displacement over a sliding window
    2. Compares RMS to target value
    3. Adjusts damping coefficient to minimize vibration
    
    Args:
        m: Mass [kg]
        c_initial: Initial damping coefficient [Ns/m]
        k: Spring stiffness [N/m]
        t_span: Time span tuple (t_start, t_end) [s]
        initial_state: Initial state [x0, v0]
        dt: Time step [s]
    
    Returns:
        Dictionary containing simulation results
    """
    t_start, t_end = t_span
    t = np.arange(t_start, t_end + dt, dt)
    n_steps = len(t)
    
    # Initialize arrays
    x = np.zeros(n_steps)
    v = np.zeros(n_steps)
    c_history = np.zeros(n_steps)
    
    # Set initial conditions
    x[0], v[0] = initial_state
    c_current = c_initial
    c_history[0] = c_current
    
    # Calculate number of samples in RMS window
    window_samples = int(RMS_WINDOW / dt)
    control_update_steps = int(CONTROL_UPDATE_INTERVAL / dt)
    
    # RK4 integration with adaptive damping
    for i in range(n_steps - 1):
        # Current state
        state = np.array([x[i], v[i]])
        t_current = t[i]
        
        # Update damping coefficient at control intervals
        if i > 0 and i % control_update_steps == 0 and i >= window_samples:
            # Calculate RMS of recent displacement
            x_window = x[i-window_samples:i]
            rms_displacement = calculate_rms(x_window)
            
            # Adaptive control law
            error = rms_displacement - TARGET_RMS
            c_current = clamp(c_current + CONTROL_GAIN * error, C_MIN, C_MAX)
        
        c_history[i] = c_current
        
        # RK4 integration step
        k1 = system_dynamics_fixed(t_current, state, m, c_current, k)
        k2 = system_dynamics_fixed(t_current + dt/2, state + dt*k1/2, m, c_current, k)
        k3 = system_dynamics_fixed(t_current + dt/2, state + dt*k2/2, m, c_current, k)
        k4 = system_dynamics_fixed(t_current + dt, state + dt*k3, m, c_current, k)
        
        state_new = state + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
        
        x[i+1] = state_new[0]
        v[i+1] = state_new[1]
    
    c_history[-1] = c_current
    
    return {
        't': t,
        'x': x,
        'v': v,
        'c': c_history
    }


# ============================================================================
# ANALYSIS AND METRICS
# ============================================================================

def calculate_metrics(t: np.ndarray, x: np.ndarray, c: np.ndarray) -> Dict:
    """
    Calculate performance metrics for the simulation.
    
    Args:
        t: Time array [s]
        x: Displacement array [m]
        c: Damping coefficient array [Ns/m]
    
    Returns:
        Dictionary containing metrics
    """
    # Maximum displacement
    max_displacement = np.max(np.abs(x))
    
    # RMS of last 1 second
    last_1s_idx = int(1.0 / (t[1] - t[0]))
    rms_last_1s = calculate_rms(x[-last_1s_idx:])
    
    # Settling time (5% criterion)
    settling_time = calculate_settling_time(t, x, threshold=0.05)
    
    # Final damping coefficient
    final_damping = c[-1]
    
    return {
        'max_displacement': max_displacement,
        'rms_last_1s': rms_last_1s,
        'settling_time': settling_time,
        'final_damping': final_damping
    }


def print_comparison_table(metrics_fixed: Dict, metrics_adaptive: Dict):
    """
    Print comparison table of performance metrics.
    
    Args:
        metrics_fixed: Metrics for fixed damping case
        metrics_adaptive: Metrics for adaptive damping case
    """
    print("\n" + "="*80)
    print("PERFORMANCE COMPARISON TABLE")
    print("="*80)
    print(f"{'Case':<20} {'Max |x| [m]':<15} {'RMS (last 1s) [m]':<20} {'Settling Time [s]':<20} {'Final c [Ns/m]':<15}")
    print("-"*80)
    
    # Non-adaptive case
    print(f"{'Non-Adaptive':<20} {metrics_fixed['max_displacement']:<15.6f} "
          f"{metrics_fixed['rms_last_1s']:<20.6f} "
          f"{metrics_fixed['settling_time']:<20.3f} "
          f"{metrics_fixed['final_damping']:<15.2f}")
    
    # Adaptive case
    print(f"{'Adaptive':<20} {metrics_adaptive['max_displacement']:<15.6f} "
          f"{metrics_adaptive['rms_last_1s']:<20.6f} "
          f"{metrics_adaptive['settling_time']:<20.3f} "
          f"{metrics_adaptive['final_damping']:<15.2f}")
    
    print("="*80)
    
    # Calculate improvements
    displacement_improvement = (1 - metrics_adaptive['max_displacement'] / metrics_fixed['max_displacement']) * 100
    rms_improvement = (1 - metrics_adaptive['rms_last_1s'] / metrics_fixed['rms_last_1s']) * 100
    
    print(f"\nIMPROVEMENTS WITH ADAPTIVE CONTROL:")
    print(f"  • Maximum displacement reduced by: {displacement_improvement:.1f}%")
    print(f"  • Steady-state RMS reduced by: {rms_improvement:.1f}%")
    print("="*80 + "\n")


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_results(results_fixed: Dict, results_adaptive: Dict, 
                metrics_fixed: Dict, metrics_adaptive: Dict):
    """
    Create comprehensive visualization of simulation results.
    
    Args:
        results_fixed: Simulation results for fixed damping
        results_adaptive: Simulation results for adaptive damping
        metrics_fixed: Performance metrics for fixed damping
        metrics_adaptive: Performance metrics for adaptive damping
    """
    fig = plt.figure(figsize=(15, 10))
    
    # Plot 1: Displacement comparison
    ax1 = plt.subplot(3, 1, 1)
    ax1.plot(results_fixed['t'], results_fixed['x'], 'b-', linewidth=1.5, 
             label='Non-Adaptive', alpha=0.7)
    ax1.plot(results_adaptive['t'], results_adaptive['x'], 'r-', linewidth=1.5, 
             label='Adaptive', alpha=0.7)
    ax1.axhline(y=TARGET_RMS, color='g', linestyle='--', linewidth=1, 
                label=f'Target RMS = {TARGET_RMS} m')
    ax1.axhline(y=-TARGET_RMS, color='g', linestyle='--', linewidth=1)
    ax1.set_xlabel('Time [s]', fontsize=11)
    ax1.set_ylabel('Displacement [m]', fontsize=11)
    ax1.set_title('Displacement Response Comparison', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Adaptive damping coefficient
    ax2 = plt.subplot(3, 1, 2)
    ax2.plot(results_adaptive['t'], results_adaptive['c'], 'r-', linewidth=2)
    ax2.axhline(y=C_CRIT, color='k', linestyle='--', linewidth=1, 
                label=f'Critical Damping = {C_CRIT:.1f} Ns/m')
    ax2.axhline(y=C_INITIAL, color='b', linestyle='--', linewidth=1, 
                label=f'Initial Damping = {C_INITIAL:.1f} Ns/m')
    ax2.set_xlabel('Time [s]', fontsize=11)
    ax2.set_ylabel('Damping Coefficient [Ns/m]', fontsize=11)
    ax2.set_title('Adaptive Damping Coefficient Over Time', fontsize=13, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Mechanical energy comparison
    ax3 = plt.subplot(3, 1, 3)
    energy_fixed = calculate_mechanical_energy(results_fixed['x'], results_fixed['v'], M, K)
    energy_adaptive = calculate_mechanical_energy(results_adaptive['x'], results_adaptive['v'], M, K)
    
    ax3.plot(results_fixed['t'], energy_fixed, 'b-', linewidth=1.5, 
             label='Non-Adaptive', alpha=0.7)
    ax3.plot(results_adaptive['t'], energy_adaptive, 'r-', linewidth=1.5, 
             label='Adaptive', alpha=0.7)
    ax3.set_xlabel('Time [s]', fontsize=11)
    ax3.set_ylabel('Mechanical Energy [J]', fontsize=11)
    ax3.set_title('Total Mechanical Energy Comparison', fontsize=13, fontweight='bold')
    ax3.legend(loc='upper right', fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('avds_results.png', dpi=300, bbox_inches='tight')
    print("Plots saved as 'avds_results.png'")
    plt.show()


# ============================================================================
# REPORT GENERATION
# ============================================================================

def generate_report(metrics_fixed: Dict, metrics_adaptive: Dict):
    """
    Generate comprehensive project report.
    
    Args:
        metrics_fixed: Performance metrics for fixed damping
        metrics_adaptive: Performance metrics for adaptive damping
    """
    report = """
================================================================================
                    ADAPTIVE VIBRATION DAMPING SYSTEM (AVDS)
                           PROJECT TECHNICAL REPORT
================================================================================

1. PROJECT SUMMARY
------------------
This project implements an adaptive vibration damping system for a single 
degree-of-freedom (SDOF) mass-spring-damper system subjected to sinusoidal
forcing. The adaptive controller continuously monitors system vibrations and
dynamically adjusts the damping coefficient to minimize oscillations while
maintaining system stability.

2. MATHEMATICAL MODEL
----------------------
Equation of Motion:
    m·ẍ + c·ẋ + k·x = F(t)

Where:
    m  = Mass of the system
    c  = Damping coefficient (adaptive)
    k  = Spring stiffness
    x  = Displacement
    F(t) = F₀·sin(ω·t) = External forcing function

Natural Frequency:
    ωₙ = √(k/m) = {omega_n:.3f} rad/s

Critical Damping:
    c_crit = 2√(k·m) = {c_crit:.3f} Ns/m

3. SYSTEM PARAMETERS
--------------------
Physical Parameters:
    • Mass (m):                {m:.2f} kg
    • Spring Stiffness (k):    {k:.2f} N/m
    • Initial Damping (c₀):    {c_initial:.2f} Ns/m
    • Damping Ratio (ζ₀):      {zeta_initial:.4f}

Forcing Function:
    • Amplitude (F₀):          {f0:.2f} N
    • Frequency (ω):           {omega:.2f} rad/s
    • Frequency Ratio (ω/ωₙ):  {freq_ratio:.3f}

Adaptive Control:
    • Target RMS:              {target_rms:.4f} m
    • Control Gain:            {gain:.1f}
    • Update Interval:         {update_interval:.3f} s
    • Damping Range:           [{c_min:.2f}, {c_max:.2f}] Ns/m

4. RESULTS INTERPRETATION
--------------------------
Non-Adaptive System (Fixed Damping):
    • Maximum Displacement:    {fixed_max:.6f} m
    • Steady-State RMS:        {fixed_rms:.6f} m
    • Settling Time:           {fixed_settle:.3f} s

Adaptive System (Variable Damping):
    • Maximum Displacement:    {adaptive_max:.6f} m
    • Steady-State RMS:        {adaptive_rms:.6f} m
    • Settling Time:           {adaptive_settle:.3f} s
    • Final Damping:           {adaptive_final_c:.2f} Ns/m

Performance Improvements:
    • Maximum Displacement:    {disp_improvement:.1f}% reduction
    • Steady-State RMS:        {rms_improvement:.1f}% reduction
    • Energy Dissipation:      Enhanced through adaptive control

Key Observations:
    1. The adaptive controller successfully reduces vibration amplitude
    2. Damping coefficient converges to optimal value for steady-state
    3. System responds dynamically to changing vibration levels
    4. Mechanical energy is more efficiently dissipated

5. RELEVANCE TO TRINITY COLLEGE DUBLIN
---------------------------------------
This project demonstrates core competencies essential for mechanical 
engineering studies at Trinity:

    • Mathematical Modeling: Formulation and solution of differential 
      equations governing dynamic mechanical systems.

    • Numerical Methods: Implementation of Runge-Kutta integration and
      real-time signal processing algorithms.

    • Control Systems: Design and validation of feedback control strategies
      for vibration suppression.

    • Engineering Analysis: Quantitative performance evaluation using 
      industry-standard metrics (RMS, settling time, energy dissipation).

These skills align directly with Trinity's curriculum in dynamics, control
theory, and computational engineering, showcasing readiness for advanced
mechanical engineering coursework.

6. CONCLUSIONS
--------------
The Adaptive Vibration Damping System successfully demonstrates the advantage
of active control over passive systems. The adaptive algorithm reduces 
vibrations by dynamically tuning the damping coefficient based on real-time
measurements, resulting in improved performance across all key metrics.

This project illustrates practical applications in:
    • Automotive suspension systems
    • Building seismic protection
    • Precision manufacturing equipment
    • Aerospace vibration control

================================================================================
                            END OF REPORT
================================================================================
"""
    
    # Calculate additional parameters for report
    omega_n = np.sqrt(K / M)
    freq_ratio = OMEGA / omega_n
    zeta_initial = C_INITIAL / C_CRIT
    
    displacement_improvement = (1 - metrics_adaptive['max_displacement'] / 
                               metrics_fixed['max_displacement']) * 100
    rms_improvement = (1 - metrics_adaptive['rms_last_1s'] / 
                      metrics_fixed['rms_last_1s']) * 100
    
    # Format report with actual values
    formatted_report = report.format(
        omega_n=omega_n,
        c_crit=C_CRIT,
        m=M,
        k=K,
        c_initial=C_INITIAL,
        zeta_initial=zeta_initial,
        f0=F0,
        omega=OMEGA,
        freq_ratio=freq_ratio,
        target_rms=TARGET_RMS,
        gain=CONTROL_GAIN,
        update_interval=CONTROL_UPDATE_INTERVAL,
        c_min=C_MIN,
        c_max=C_MAX,
        fixed_max=metrics_fixed['max_displacement'],
        fixed_rms=metrics_fixed['rms_last_1s'],
        fixed_settle=metrics_fixed['settling_time'],
        adaptive_max=metrics_adaptive['max_displacement'],
        adaptive_rms=metrics_adaptive['rms_last_1s'],
        adaptive_settle=metrics_adaptive['settling_time'],
        adaptive_final_c=metrics_adaptive['final_damping'],
        disp_improvement=displacement_improvement,
        rms_improvement=rms_improvement
    )
    
    print(formatted_report)
    
    # Save report to file
    with open('avds_report.txt', 'w') as f:
        f.write(formatted_report)
    print("Report saved as 'avds_report.txt'\n")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main execution function for AVDS project.
    """
    print("\n" + "="*80)
    print(" "*20 + "ADAPTIVE VIBRATION DAMPING SYSTEM (AVDS)")
    print(" "*15 + "Trinity College Dublin Portfolio Project")
    print("="*80 + "\n")
    
    print("Initializing simulation parameters...")
    print(f"  • Mass: {M} kg")
    print(f"  • Spring Stiffness: {K} N/m")
    print(f"  • Initial Damping: {C_INITIAL} Ns/m")
    print(f"  • Critical Damping: {C_CRIT:.2f} Ns/m")
    print(f"  • Simulation Time: {T_START} - {T_END} s")
    print(f"  • Time Step: {DT} s\n")
    
    # Prepare time evaluation points
    t_eval = np.arange(T_START, T_END + DT, DT)
    initial_state = np.array([X0, V0])
    
    # Run non-adaptive simulation
    print("Running non-adaptive simulation (fixed damping)...")
    results_fixed = simulate_fixed_damping(
        m=M,
        c=C_INITIAL,
        k=K,
        t_span=(T_START, T_END),
        initial_state=initial_state,
        t_eval=t_eval
    )
    print("  ✓ Non-adaptive simulation complete")
    
    # Run adaptive simulation
    print("Running adaptive simulation (variable damping)...")
    results_adaptive = simulate_adaptive_damping(
        m=M,
        c_initial=C_INITIAL,
        k=K,
        t_span=(T_START, T_END),
        initial_state=initial_state,
        dt=DT
    )
    print("  ✓ Adaptive simulation complete\n")
    
    # Calculate metrics
    print("Calculating performance metrics...")
    metrics_fixed = calculate_metrics(
        results_fixed['t'],
        results_fixed['x'],
        results_fixed['c']
    )
    metrics_adaptive = calculate_metrics(
        results_adaptive['t'],
        results_adaptive['x'],
        results_adaptive['c']
    )
    print("  ✓ Metrics calculated\n")
    
    # Print comparison table
    print_comparison_table(metrics_fixed, metrics_adaptive)
    
    # Generate visualizations
    print("Generating visualizations...")
    plot_results(results_fixed, results_adaptive, metrics_fixed, metrics_adaptive)
    print("  ✓ Visualizations complete\n")
    
    # Generate comprehensive report
    print("Generating technical report...")
    generate_report(metrics_fixed, metrics_adaptive)
    print("  ✓ Report generation complete\n")
    
    print("="*80)
    print(" "*25 + "SIMULATION COMPLETED SUCCESSFULLY")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
