"""
Festival OR Model - Multi-Zone Dynamic Simulation Toolkit
Extends or_des_toolkit_week6.py (ORSL Week 6) with:
  - A ZoneSpec/Zone framework so new zones can be instantiated as needed
  - A weather Markov chain w(t) (Section 22.4 of the model document)
  - Inter-zone crowd movement m_z(t) (Section 22, "Inter-Zone Movement")
  - Severity-weighted incident sampling (Section 17 / 22.9)
  - Violation/noise accumulation V(t), nu(t) (Section 22.10)
  - Aggregation into u_O, u_G, u_A (Section 24)

Each zone reuses the arrival/service primitives from or_des_toolkit_week6
(generate_arrivals, sample_service_times) but the whole festival steps
together on a shared 15-minute clock so zones can exchange crowd.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from or_des_toolkit_week6 import arrival_rate_at_time, QueueScenario  # reuse arrival shape helper


# ============================================================
# WEATHER PROCESS  (Section 22.4)
# ============================================================

WEATHER_STATES = ["clear", "rain", "heat"]

# T_w: rows/cols = clear, rain, heat (from -> to). Stationary dist ~ (0.45, 0.40, 0.15)
T_WEATHER = np.array([
    [0.95, 0.04, 0.01],
    [0.08, 0.91, 0.01],
    [0.05, 0.02, 0.93],
])

D_MAX_BY_WEATHER = {"clear": 2.0, "rain": 1.7, "heat": 1.5}      # people/m^2
PHI_BY_WEATHER = {"clear": 82.0, "rain": 70.0, "heat": 75.0}     # exit flow, people/m/min
COST_MULT_BY_WEATHER = {"clear": 1.00, "rain": 1.15, "heat": 1.40}
NOSHOW_BY_WEATHER = {"clear": 0.10, "rain": 0.20, "heat": 0.12}
INCIDENT_MULT = {  # (minor, moderate, critical) multipliers, Section 17.3
    "clear": (1.0, 1.0, 1.0),
    "rain":  (1.3, 1.5, 1.2),
    "heat":  (2.0, 2.5, 3.0),
}


def draw_initial_weather(rng: np.random.Generator) -> str:
    # stationary distribution of T_WEATHER, hardcoded from Section 14.2 prior
    return rng.choice(WEATHER_STATES, p=[0.45, 0.40, 0.15])


def step_weather(w: str, rng: np.random.Generator) -> str:
    idx = WEATHER_STATES.index(w)
    return rng.choice(WEATHER_STATES, p=T_WEATHER[idx])


# ============================================================
# ZONE FRAMEWORK  (Section 22.1)
# ============================================================

@dataclass(frozen=True)
class ZoneSpec:
    """Static definition of a zone. Instantiate one of these per zone."""
    name: str
    area_m2: float
    n_gates: int
    gate_base_throughput_per_hr: float = 400.0   # g_i^base, Section 22.12
    exit_width_m: float = 0.0                    # W_l, used for egress A_max
    arrival_share: float = 0.0                   # fraction of festival-wide a_z allocation
    v_z: int = 1                                 # vendor stalls (Section 21)
    adjacent: Tuple[str, ...] = field(default_factory=tuple)  # names of adjacent zones


# Default 4-zone topology from Section 22.1
DEFAULT_ZONES: Dict[str, ZoneSpec] = {
    "main_stage":   ZoneSpec("main_stage",   14000, n_gates=2, exit_width_m=35, arrival_share=0.431,
                              v_z=112, adjacent=("second_stage", "food_court", "camping")),
    "second_stage": ZoneSpec("second_stage",  8000, n_gates=1, exit_width_m=15, arrival_share=0.246,
                              v_z=64,  adjacent=("main_stage", "food_court")),
    "food_court":   ZoneSpec("food_court",    6000, n_gates=1, exit_width_m=12, arrival_share=0.185,
                              v_z=49,  adjacent=("main_stage", "second_stage", "camping")),
    "camping":      ZoneSpec("camping",       4500, n_gates=1, exit_width_m=18, arrival_share=0.138,
                              v_z=36,  adjacent=("main_stage", "food_court")),
}


# ============================================================
# SEVERITY-WEIGHTED INCIDENTS  (Section 17 / 22.9)
# ============================================================

RATE_PER_1000 = {"minor": 10.0, "moderate": 0.57, "critical": 0.003}
OMEGA = {"minor": 1.0, "moderate": 10.0, "critical": 1000.0}


def sample_incidents(a_z: float, weather: str, dt_hours: float, rng: np.random.Generator) -> Dict[str, int]:
    mult = dict(zip(["minor", "moderate", "critical"], INCIDENT_MULT[weather]))
    out = {}
    for tier, rate1000 in RATE_PER_1000.items():
        lam = (rate1000 / 1000.0) * mult[tier] * a_z * (dt_hours / 16.0)
        out[tier] = rng.poisson(max(lam, 0.0))
    return out


# ============================================================
# ZONE RUNTIME STATE
# ============================================================

@dataclass
class ZoneState:
    spec: ZoneSpec
    a: float = 0.0          # current occupancy a_z(t)
    q_gate: float = 0.0     # gate queue q_i(t), aggregated across this zone's gates
    incidents_cum: Dict[str, int] = field(default_factory=lambda: {"minor": 0, "moderate": 0, "critical": 0})

    @property
    def density(self) -> float:
        return self.a / self.spec.area_m2


# ============================================================
# FESTIVAL SIMULATION
# ============================================================

@dataclass(frozen=True)
class FestivalScenario:
    name: str
    a_total: int               # intended total attendance (a^total)
    t_evac_min: int            # T_evac, 8 (strict) or 10 (lenient)
    ticket_price: float = 365.0
    horizon_steps: int = 64    # T, 15-min steps over 16 hours
    dt_hours: float = 0.25
    kappa_m: float = 0.05      # inter-zone equilibration rate, Section 22 "Inter-Zone Movement"
    v_max_violation: int = 3   # V_max threshold for full evacuation (Section 26)
    seed: Optional[int] = None


@dataclass(frozen=True)
class AlternativeA(FestivalScenario):
    """Convenience constructors for A1-A4 from Section 16.1 / 25."""
    pass


def make_alternative(label: str) -> FestivalScenario:
    table = {
        "A1": dict(a_total=45000, t_evac_min=8),
        "A2": dict(a_total=45000, t_evac_min=10),
        "A3": dict(a_total=55000, t_evac_min=10),
        "A4": dict(a_total=65000, t_evac_min=10),
    }
    if label not in table:
        raise ValueError(f"Unknown alternative {label!r}, expected A1-A4")
    return FestivalScenario(name=label, **table[label])


def egress_capacity(zones: Dict[str, ZoneSpec], weather: str, t_evac_min: int) -> float:
    total_width = sum(z.exit_width_m for z in zones.values())
    return total_width * PHI_BY_WEATHER[weather] * t_evac_min


def holding_capacity(zones: Dict[str, ZoneSpec], weather: str, d_min: float = 0.5) -> float:
    total_area = sum(z.area_m2 for z in zones.values())
    return total_area / d_min


def a_max(zones: Dict[str, ZoneSpec], weather: str, t_evac_min: int) -> float:
    return min(holding_capacity(zones, weather), egress_capacity(zones, weather, t_evac_min))


def run_festival_once(
    scenario: FestivalScenario,
    zones: Dict[str, ZoneSpec] = DEFAULT_ZONES,
    seed: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run one stochastic replication of the festival.
    Returns (timeline, zone_timeline):
      timeline: per-step festival-wide variables (weather, V, nu, A_max, total occupancy)
      zone_timeline: per-step per-zone variables (a_z, density, queue, incidents)
    """
    rng = np.random.default_rng(seed if seed is not None else scenario.seed)

    states = {name: ZoneState(spec) for name, spec in zones.items()}
    weather = draw_initial_weather(rng)
    no_show = NOSHOW_BY_WEATHER[weather]
    a_eff_total = scenario.a_total * (1 - no_show)

    V = 0
    nu = 0
    evacuated = False

    fest_rows = []
    zone_rows = []

    # Arrival shape: reuse the morning_peak helper from week6 toolkit as a
    # stand-in for the log-normal 70/30 arrival profile (Section 22.5)
    arrival_scn = QueueScenario(horizon_minutes=scenario.horizon_steps, arrival_pattern="morning_peak")
    raw_shape = np.array([arrival_rate_at_time(t, arrival_scn) for t in range(scenario.horizon_steps)])
    arrival_fraction = raw_shape / raw_shape.sum()  # f(t), normalised so sum = 1

    for t in range(scenario.horizon_steps):
        d_max_w = D_MAX_BY_WEATHER[weather]
        amax_now = a_max(zones, weather, scenario.t_evac_min)

        if evacuated:
            for st in states.values():
                st.a = 0.0
        else:
            # --- gate arrivals into each zone, proportional to arrival_share ---
            step_inflow_total = a_eff_total * arrival_fraction[t]
            for name, st in states.items():
                inflow = step_inflow_total * st.spec.arrival_share
                # policy rule: halt entry if zone density >= theta_warn * D_max (Section 11)
                if st.density >= 0.85 * d_max_w:
                    st.q_gate += inflow
                else:
                    admit = min(inflow + st.q_gate, max(0.0, 0.85 * d_max_w * st.spec.area_m2 - st.a))
                    st.q_gate = max(0.0, inflow + st.q_gate - admit)
                    st.a += admit

            # --- inter-zone movement m_z(t): density-gradient flow ---
            moves = {name: 0.0 for name in states}
            for name, st in states.items():
                for nbr in st.spec.adjacent:
                    nbr_st = states[nbr]
                    grad = st.density - nbr_st.density
                    if grad > 0:
                        flow = scenario.kappa_m * grad * min(st.spec.area_m2, nbr_st.spec.area_m2)
                        moves[name] -= flow
                        moves[nbr] += flow
            for name, st in states.items():
                st.a = max(0.0, st.a + moves[name])

            # --- incidents (Section 17 / 22.9) ---
            for st in states.values():
                inc = sample_incidents(st.a, weather, scenario.dt_hours, rng)
                for k, v in inc.items():
                    st.incidents_cum[k] += v

            # --- violations: density breach or A_max breach ---
            total_a = sum(st.a for st in states.values())
            density_breach = any(st.density > D_MAX_BY_WEATHER[weather] for st in states.values())
            amax_breach = total_a > amax_now
            V += int(density_breach) + int(amax_breach)

            # noise complaints, sampled hourly (every 4 steps)
            if t % 4 == 0:
                chi_nu = 0.8 if weather == "rain" else 1.0
                nu += rng.poisson(0.3 * (total_a / 65000.0) * chi_nu)

            if V > scenario.v_max_violation:
                evacuated = True

        # --- log per-zone state ---
        for name, st in states.items():
            zone_rows.append({
                "t": t, "zone": name, "a_z": st.a, "density": st.density,
                "q_gate": st.q_gate,
                "minor": st.incidents_cum["minor"],
                "moderate": st.incidents_cum["moderate"],
                "critical": st.incidents_cum["critical"],
            })

        fest_rows.append({
            "t": t, "weather": weather, "V": V, "nu": nu,
            "A_max": amax_now, "total_a": sum(st.a for st in states.values()),
            "evacuated": evacuated,
        })

        # --- step weather for next period ---
        weather = step_weather(weather, rng)

    return pd.DataFrame(fest_rows), pd.DataFrame(zone_rows)


# ============================================================
# AGGREGATION INTO UTILITIES  (Section 24)
# ============================================================

def aggregate_run(
    timeline: pd.DataFrame,
    zone_timeline: pd.DataFrame,
    scenario: FestivalScenario,
    cost_per_attendee: float = 80.0,
    omega_z_total: float = 391090.0,  # sum_z b_z baseline, Section 25
) -> Dict[str, float]:
    p = scenario.ticket_price

    R = p * timeline["total_a"].iloc[-1] / 1000.0  # (EUR thousands)
    C = (timeline["weather"].map(COST_MULT_BY_WEATHER) * (omega_z_total / 1000.0)).mean()
    D = zone_timeline["density"].max()
    Q = (zone_timeline["q_gate"] / zone_timeline["zone"].map(lambda z: DEFAULT_ZONES[z].v_z)).max()

    minor = zone_timeline.groupby("zone")["minor"].max().sum()
    moderate = zone_timeline.groupby("zone")["moderate"].max().sum()
    critical = zone_timeline.groupby("zone")["critical"].max().sum()
    phi = OMEGA["minor"] * minor + OMEGA["moderate"] * moderate + OMEGA["critical"] * (critical ** 2)

    V_final = timeline["V"].iloc[-1]
    nu_final = timeline["nu"].iloc[-1]

    u_O = 3 * R - 2 * C - 1 * D - phi
    # government utility (Section 6.2): economic benefit vs noise/violations/evac
    E_econ = R
    u_G = 2 * E_econ - 1 * nu_final - 3 * V_final * 10 - 2 * scenario.t_evac_min

    feasible = V_final == 0 and not timeline["evacuated"].any()

    return {
        "scenario": scenario.name,
        "u_O": u_O, "u_G": u_G, "D": D, "Q": Q,
        "V": V_final, "nu": nu_final, "Phi": phi,
        "minor": minor, "moderate": moderate, "critical": critical,
        "final_attendance": timeline["total_a"].iloc[-1],
        "feasible": feasible,
    }


def monte_carlo(scenario: FestivalScenario, n_runs: int = 100, seed: int = 2026,
                 zones: Dict[str, ZoneSpec] = DEFAULT_ZONES) -> pd.DataFrame:
    rows = []
    for run in range(n_runs):
        timeline, zone_timeline = run_festival_once(scenario, zones=zones, seed=seed + 17 * run)
        rows.append(aggregate_run(timeline, zone_timeline, scenario))
    return pd.DataFrame(rows)


def summarize_alternative(df: pd.DataFrame) -> Dict[str, float]:
    return {
        "E_uO": df["u_O"].mean(),
        "E_uG": df["u_G"].mean(),
        "minimax_uO": df["u_O"].min(),
        "frac_feasible": df["feasible"].mean(),
        "mean_D": df["D"].mean(),
        "mean_critical": df["critical"].mean(),
    }


if __name__ == "__main__":
    for label in ["A1", "A2", "A3", "A4"]:
        scn = make_alternative(label)
        results = monte_carlo(scn, n_runs=50)
        summary = summarize_alternative(results)
        print(label, summary)