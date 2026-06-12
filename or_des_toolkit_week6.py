
"""
OR Simulation Lab - Week 6
Dynamic Simulation and Discrete Event Queueing Toolkit

Reusable for:
- airport security queues
- museum visitor flow
- festival entrance gates
- hospital triage
- port truck gates
- call centres
- evacuation bottlenecks

Week 5: many possible futures for a one-period decision.
Week 6: many possible futures for a system evolving over time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Optional, Tuple, Dict

import numpy as np
import pandas as pd


ArrivalPattern = Literal["constant", "morning_peak", "two_peaks"]


@dataclass(frozen=True)
class QueueScenario:
    horizon_minutes: int = 360
    base_arrival_rate_per_min: float = 1.8
    arrival_pattern: ArrivalPattern = "morning_peak"
    mean_service_minutes: float = 2.2
    service_cv: float = 0.45
    disruption_probability: float = 0.10
    disruption_multiplier: float = 1.35
    seed: Optional[int] = None


@dataclass(frozen=True)
class QueuePolicy:
    name: str
    n_servers: int
    dynamic_extra_server: bool = False
    extra_server_start: int = 90
    extra_server_end: int = 230
    max_acceptable_wait: float = 15.0


def arrival_rate_at_time(t: float, scenario: QueueScenario) -> float:
    base = scenario.base_arrival_rate_per_min
    horizon = scenario.horizon_minutes

    if scenario.arrival_pattern == "constant":
        return base

    if scenario.arrival_pattern == "morning_peak":
        peak = 1.8 * np.exp(-0.5 * ((t - 0.42 * horizon) / 55.0) ** 2)
        return base * (0.75 + peak)

    if scenario.arrival_pattern == "two_peaks":
        peak1 = 1.3 * np.exp(-0.5 * ((t - 0.30 * horizon) / 45.0) ** 2)
        peak2 = 1.5 * np.exp(-0.5 * ((t - 0.72 * horizon) / 50.0) ** 2)
        return base * (0.65 + peak1 + peak2)

    raise ValueError(f"Unknown arrival pattern: {scenario.arrival_pattern}")


def generate_arrivals(scenario: QueueScenario, rng: np.random.Generator) -> np.ndarray:
    arrivals = []
    for minute in range(scenario.horizon_minutes):
        rate = arrival_rate_at_time(minute, scenario)
        n_arrivals = rng.poisson(rate)
        if n_arrivals:
            arrivals.extend(minute + rng.random(n_arrivals))
    return np.array(sorted(arrivals), dtype=float)


def sample_service_times(n: int, scenario: QueueScenario, rng: np.random.Generator) -> np.ndarray:
    cv = scenario.service_cv
    shape = 1 / (cv ** 2)
    scale = scenario.mean_service_minutes / shape
    service = rng.gamma(shape=shape, scale=scale, size=n)
    if rng.random() < scenario.disruption_probability:
        service = service * scenario.disruption_multiplier
    return service


def servers_available_at_time(t: float, policy: QueuePolicy) -> int:
    servers = policy.n_servers
    if policy.dynamic_extra_server and policy.extra_server_start <= t <= policy.extra_server_end:
        servers += 1
    return servers


def simulate_queue_once(
    policy: QueuePolicy,
    scenario: QueueScenario,
    run_id: int = 0,
    seed: Optional[int] = None,
    return_timeline: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed if seed is not None else scenario.seed)
    arrivals = generate_arrivals(scenario, rng)
    n = len(arrivals)
    service_times = sample_service_times(n, scenario, rng)

    max_servers = policy.n_servers + (1 if policy.dynamic_extra_server else 0)
    server_available = np.zeros(max_servers)

    starts = np.zeros(n)
    departures = np.zeros(n)
    waits = np.zeros(n)
    servers_used = np.zeros(n, dtype=int)
    active_servers_at_arrival = np.zeros(n, dtype=int)

    for i, arrival in enumerate(arrivals):
        active_servers = servers_available_at_time(arrival, policy)
        active_servers_at_arrival[i] = active_servers
        chosen = int(np.argmin(server_available[:active_servers]))
        start = max(arrival, server_available[chosen])
        departure = start + service_times[i]

        starts[i] = start
        departures[i] = departure
        waits[i] = start - arrival
        servers_used[i] = chosen
        server_available[chosen] = departure

    customers = pd.DataFrame({
        "run_id": run_id,
        "policy": policy.name,
        "customer_id": np.arange(n),
        "arrival_time": arrivals,
        "service_start": starts,
        "service_time": service_times,
        "departure_time": departures,
        "wait_time": waits,
        "time_in_system": departures - arrivals,
        "server": servers_used,
        "servers_available_at_arrival": active_servers_at_arrival,
        "overload": waits > policy.max_acceptable_wait,
    })

    if not return_timeline:
        return customers, pd.DataFrame()

    # Efficient minute-level timeline.
    minutes = np.arange(scenario.horizon_minutes)
    arrived_cum = np.searchsorted(arrivals, minutes, side="right")
    departed_cum = np.searchsorted(np.sort(departures), minutes, side="right")
    service_started_cum = np.searchsorted(np.sort(starts), minutes, side="right")
    queue_length = arrived_cum - service_started_cum
    in_service = service_started_cum - departed_cum
    servers_available = np.array([servers_available_at_time(m, policy) for m in minutes])

    timeline = pd.DataFrame({
        "run_id": run_id,
        "policy": policy.name,
        "minute": minutes,
        "arrived_cumulative": arrived_cum,
        "departed_cumulative": departed_cum,
        "queue_length": np.maximum(0, queue_length),
        "in_service": np.maximum(0, in_service),
        "servers_available": servers_available,
        "in_system": arrived_cum - departed_cum,
    })

    return customers, timeline


def summarize_queue_run(customers: pd.DataFrame, timeline: pd.DataFrame, policy: QueuePolicy, scenario: QueueScenario) -> Dict[str, float]:
    if customers.empty:
        return {
            "policy": policy.name,
            "n_customers": 0.0,
            "mean_wait": 0.0,
            "p95_wait": 0.0,
            "max_wait": 0.0,
            "prob_overload": 0.0,
            "mean_time_in_system": 0.0,
            "max_queue_length": 0.0,
            "mean_queue_length": 0.0,
            "server_utilisation_proxy": 0.0,
        }

    if timeline.empty:
        # Approximate utilisation without timeline.
        avg_servers = policy.n_servers + (1 if policy.dynamic_extra_server else 0) * (
            max(0, policy.extra_server_end - policy.extra_server_start) / scenario.horizon_minutes
        )
        available_server_minutes = avg_servers * scenario.horizon_minutes
        max_queue = float(customers["overload"].sum())
        mean_queue = max_queue / 3
    else:
        available_server_minutes = timeline["servers_available"].sum()
        max_queue = float(timeline["queue_length"].max())
        mean_queue = float(timeline["queue_length"].mean())

    utilisation = customers["service_time"].sum() / max(1, available_server_minutes)

    return {
        "policy": policy.name,
        "n_customers": float(len(customers)),
        "mean_wait": float(customers["wait_time"].mean()),
        "p95_wait": float(customers["wait_time"].quantile(0.95)),
        "max_wait": float(customers["wait_time"].max()),
        "prob_overload": float(customers["overload"].mean()),
        "mean_time_in_system": float(customers["time_in_system"].mean()),
        "max_queue_length": max_queue,
        "mean_queue_length": mean_queue,
        "server_utilisation_proxy": float(utilisation),
    }


def simulate_policy_many_runs(
    policy: QueuePolicy,
    scenario: QueueScenario,
    n_runs: int = 100,
    seed: int = 42
) -> pd.DataFrame:
    rows = []
    for run in range(n_runs):
        customers, timeline = simulate_queue_once(
            policy=policy,
            scenario=scenario,
            run_id=run,
            seed=seed + 17 * run,
            return_timeline=True,
        )
        rows.append(summarize_queue_run(customers, timeline, policy, scenario))
    return pd.DataFrame(rows)


def compare_queue_policies(
    policies: Iterable[QueuePolicy],
    scenario: QueueScenario,
    n_runs: int = 100,
    seed: int = 42
) -> pd.DataFrame:
    parts = []
    for i, policy in enumerate(policies):
        result = simulate_policy_many_runs(policy, scenario, n_runs, seed + 1000 * i)
        parts.append(result)
    return pd.concat(parts, ignore_index=True)


def aggregate_policy_summary(run_summary: pd.DataFrame) -> pd.DataFrame:
    return run_summary.groupby("policy").agg(
        mean_wait=("mean_wait", "mean"),
        p95_wait_mean=("p95_wait", "mean"),
        p95_wait_p95=("p95_wait", lambda x: float(np.quantile(x, 0.95))),
        max_queue_mean=("max_queue_length", "mean"),
        overload_prob_mean=("prob_overload", "mean"),
        utilisation_mean=("server_utilisation_proxy", "mean"),
        customers_mean=("n_customers", "mean"),
    ).reset_index()


def project_des_template() -> Dict[str, str]:
    return {
        "entity": "What moves through the system? passengers, patients, visitors, trucks, orders",
        "arrival_event": "When does an entity enter the system?",
        "queue_rule": "Who waits, where, and in what order?",
        "resource": "What limited capacity serves the entity? lanes, staff, gates, doctors, machines",
        "service_time": "How long does service take?",
        "departure_event": "When does the entity leave the system?",
        "kpis": "Waiting time, queue length, utilisation, overload, fairness, cost, safety",
    }
