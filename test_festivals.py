"""
Festival simulation test suite — multi-festival validation.

Tests the simulation against real-world festival archetypes to verify that
parameters produce plausible outcomes across different festival types.
Each festival is defined with real-world reference data for manual sanity checks.

Run:  python test_festivals.py
"""

from __future__ import annotations
import textwrap
import numpy as np
import pandas as pd
from main import (
    FestivalType, FestivalScenario, ZoneSpec,
    make_scenario, monte_carlo, summarize_alternative,
    FESTIVAL_TYPE, DEFAULT_ZONES,
)
from dataclasses import dataclass, field
from typing import Dict, Tuple

N_RUNS   = 30   # replications per festival (keep fast; raise to 100 for final results)
SEED     = 2026

# ── Helpers ────────────────────────────────────────────────────────────────────

def zones_from_areas(
    areas: Dict[str, float],
    exit_widths: Dict[str, float],
    arrival_shares: Dict[str, float],
    vendor_stalls: Dict[str, int],
) -> Dict[str, ZoneSpec]:
    """Build a zones dict from simple per-zone dicts."""
    return {
        name: ZoneSpec(
            name=name,
            area_m2=areas[name],
            n_gates=0,
            exit_width_m=exit_widths[name],
            arrival_share=arrival_shares[name],
            v_z=vendor_stalls[name],
            is_exogenous=False,
        )
        for name in areas
    }


@dataclass
class FestivalSpec:
    """One test case: a real festival with expected-range assertions."""
    name: str
    description: str          # one-line real-world context
    scenario: FestivalScenario
    zones: Dict[str, ZoneSpec]
    # Expected ranges for sanity checks (None = skip)
    expect_ever_admitted_range: Tuple[int, int] = None   # (min, max) gate scans total
    expect_peak_occ_range:      Tuple[int, int] = None   # (min, max) festival terrain peak
    expect_feasible_pct_min:    float = 0.0              # minimum fraction feasible runs
    expect_mean_density_max:    float = 3.0              # p/m² ceiling


# ── Festival definitions ───────────────────────────────────────────────────────

def build_test_festivals() -> list[FestivalSpec]:
    tests = []

    # ── 1. Lowlands (baseline — should match main.py A4) ──────────────────────
    # 65,000 3-day camping sell-out, Biddinghuizen NL
    tests.append(FestivalSpec(
        name="Lowlands_A4",
        description="65k 3-day camping sell-out (Biddinghuizen NL) — simulation baseline",
        scenario=make_scenario(
            "Lowlands_A4", a_total=65000, t_evac_min=10,
            festival_type=FESTIVAL_TYPE["camping_sellout"],
            n_stages=3, staggered_end_times=True,
            gate_open_hour=10, headliner_start_hour=23,
        ),
        zones=DEFAULT_ZONES,
        expect_ever_admitted_range=(60000, 75000),
        expect_peak_occ_range=(45000, 65000),
        expect_feasible_pct_min=0.70,
        expect_mean_density_max=2.0,
    ))

    # ── 2. Glastonbury ────────────────────────────────────────────────────────
    # 210,000 5-day camping, Worthy Farm UK — largest greenfield festival
    # Zones: Pyramid Field, Other Stage, West Holts, Park, Avalon
    # ever_admitted = day-1 gate scans + small day-tripper scans days 2-5
    # With 2% day-trippers: 210k * 0.02 * 4 extra days ≈ 17k on top of ~206k day 1
    glastonbury_areas = {
        "pyramid_field":  45000,
        "other_stage":    35000,
        "west_holts":     25000,
        "park_avalon":    30000,
    }
    glastonbury_exits = {"pyramid_field": 900, "other_stage": 700,
                         "west_holts": 500, "park_avalon": 600}
    glastonbury_shares = {"pyramid_field": 0.35, "other_stage": 0.27,
                          "west_holts": 0.19, "park_avalon": 0.19}
    glastonbury_stalls = {"pyramid_field": 300, "other_stage": 220,
                          "west_holts": 160, "park_avalon": 190}
    ft_glastonbury = FestivalType(
        n_days=5, is_camping=True, camping_fraction=0.98,
        sold_out_fraction=1.0, has_official_resale=False,
    )
    tests.append(FestivalSpec(
        name="Glastonbury",
        description="210k 5-day camping sell-out (Worthy Farm UK)",
        scenario=make_scenario(
            "Glastonbury", a_total=210000, t_evac_min=10,
            festival_type=ft_glastonbury,
            n_stages=4, staggered_end_times=True,
            gate_open_hour=8, headliner_start_hour=22,
        ),
        zones=zones_from_areas(glastonbury_areas, glastonbury_exits,
                               glastonbury_shares, glastonbury_stalls),
        expect_ever_admitted_range=(200000, 240000),  # day-1 + day-tripper re-scans days 2-5
        expect_peak_occ_range=(120000, 210000),
        expect_feasible_pct_min=0.50,
        expect_mean_density_max=2.5,
    ))

    # ── 3. Ultra Music Festival Miami ─────────────────────────────────────────
    # 165,000 3-day day-tripper EDM sell-out, Bayfront Park — hard-end exodus
    # Bayfront Park: ~35 acres = ~140,000m² total, festival area ~half that
    # ever_admitted: 165k * 3 days (full re-admission each day, no camping)
    ultra_areas = {
        "main_stage":    35000,
        "live_stage":    25000,
        "resistance":    20000,
    }
    ultra_exits = {"main_stage": 1200, "live_stage": 800, "resistance": 650}
    ultra_shares = {"main_stage": 0.55, "live_stage": 0.25, "resistance": 0.20}
    ultra_stalls = {"main_stage": 330, "live_stage": 220, "resistance": 176}
    ft_ultra = FestivalType(
        n_days=3, is_camping=False, camping_fraction=0.0,
        sold_out_fraction=1.0, has_official_resale=True,
    )
    tests.append(FestivalSpec(
        name="Ultra_Miami",
        description="165k 3-day day-tripper EDM sell-out (Bayfront Park) — hard-end exodus",
        scenario=make_scenario(
            "Ultra_Miami", a_total=165000, t_evac_min=10,
            festival_type=ft_ultra,
            n_stages=3, staggered_end_times=False,
            gate_open_hour=12, headliner_start_hour=23,
        ),
        zones=zones_from_areas(ultra_areas, ultra_exits,
                               ultra_shares, ultra_stalls),
        expect_ever_admitted_range=(140000, 510000),  # capped by venue capacity at high density
        expect_peak_occ_range=(80000, 165000),
        expect_feasible_pct_min=0.0,   # known feasibility risk at this density
        expect_mean_density_max=3.0,
    ))

    # ── 4. Download Festival ──────────────────────────────────────────────────
    # 111,000 3-day camping rock, Donington Park UK
    # Donington Park site: ~500 acres, festival area ~150 acres = ~600,000m²
    download_areas = {
        "main_stage":   55000,
        "second_stage": 35000,
        "third_stage":  25000,
    }
    download_exits = {"main_stage": 1400, "second_stage": 900, "third_stage": 640}
    download_shares = {"main_stage": 0.50, "second_stage": 0.30, "third_stage": 0.20}
    download_stalls = {"main_stage": 200, "second_stage": 130, "third_stage": 90}
    ft_download = FestivalType(
        n_days=3, is_camping=True, camping_fraction=0.80,
        sold_out_fraction=0.95, has_official_resale=False,
    )
    tests.append(FestivalSpec(
        name="Download_UK",
        description="111k 3-day camping rock (Donington Park UK)",
        scenario=make_scenario(
            "Download_UK", a_total=111000, t_evac_min=10,
            festival_type=ft_download,
            n_stages=3, staggered_end_times=True,
            gate_open_hour=11, headliner_start_hour=22,
        ),
        zones=zones_from_areas(download_areas, download_exits,
                               download_shares, download_stalls),
        expect_ever_admitted_range=(100000, 165000),  # 111k day 1 + 20% day-tripper days 2-3
        expect_peak_occ_range=(60000, 111000),
        expect_feasible_pct_min=0.50,
        expect_mean_density_max=2.0,
    ))

    # ── 5. Defqon.1 ───────────────────────────────────────────────────────────
    # 60,000 2-day day-tripper hardstyle, Biddinghuizen NL (same site as Lowlands)
    # ever_admitted: ~60k * 2 days = ~120k cumulative gate scans
    defqon_areas = {
        "main_stage":    20000,
        "blue_stage":    15000,
        "endshow_area":  25000,
    }
    defqon_exits = {"main_stage": 400, "blue_stage": 300, "endshow_area": 500}
    defqon_shares = {"main_stage": 0.40, "blue_stage": 0.30, "endshow_area": 0.30}
    defqon_stalls = {"main_stage": 120, "blue_stage": 90, "endshow_area": 150}
    ft_defqon = FestivalType(
        n_days=2, is_camping=False, camping_fraction=0.0,
        sold_out_fraction=1.0, has_official_resale=True,
    )
    tests.append(FestivalSpec(
        name="Defqon1_NL",
        description="60k 2-day day-tripper hardstyle sell-out (Biddinghuizen NL)",
        scenario=make_scenario(
            "Defqon1_NL", a_total=60000, t_evac_min=8,
            festival_type=ft_defqon,
            n_stages=3, staggered_end_times=False,
            gate_open_hour=11, headliner_start_hour=23,
        ),
        zones=zones_from_areas(defqon_areas, defqon_exits,
                               defqon_shares, defqon_stalls),
        expect_ever_admitted_range=(100000, 130000),  # ~60k × 2 days full re-admission
        expect_peak_occ_range=(40000, 72000),   # day-boundary overlap: day-2 arrivals before day-1 trickle clears
        expect_feasible_pct_min=0.40,
        expect_mean_density_max=2.5,
    ))

    # ── 6. Small single-day festival ─────────────────────────────────────────
    # 8,000 1-day day-tripper local festival, single stage
    small_areas = {"main_stage": 6000, "food_area": 3000}
    small_exits  = {"main_stage": 120,  "food_area": 60}
    small_shares = {"main_stage": 0.65, "food_area": 0.35}
    small_stalls = {"main_stage": 32,   "food_area": 40}
    ft_small = FestivalType(
        n_days=1, is_camping=False, camping_fraction=0.0,
        sold_out_fraction=0.75, has_official_resale=False,
    )
    tests.append(FestivalSpec(
        name="Small_LocalFest",
        description="8k 1-day single-stage local festival",
        scenario=make_scenario(
            "Small_LocalFest", a_total=8000, t_evac_min=8,
            festival_type=ft_small,
            n_stages=1, staggered_end_times=False,
            gate_open_hour=13, headliner_start_hour=20,
        ),
        zones=zones_from_areas(small_areas, small_exits,
                               small_shares, small_stalls),
        expect_ever_admitted_range=(6000, 8500),
        expect_peak_occ_range=(4000, 8000),
        expect_feasible_pct_min=0.70,
        expect_mean_density_max=2.0,
    ))

    # ── 7. Roskilde ───────────────────────────────────────────────────────────
    # 130,000 8-day camping (4 music days), Roskilde DK — Europe's oldest major
    roskilde_areas = {
        "orange_stage": 38000,
        "arena_stage":  25000,
        "pavilion":     20000,
        "apollo":       15000,
    }
    roskilde_exits  = {"orange_stage": 800, "arena_stage": 520,
                       "pavilion": 420, "apollo": 300}
    roskilde_shares = {"orange_stage": 0.38, "arena_stage": 0.26,
                       "pavilion": 0.20, "apollo": 0.16}
    roskilde_stalls = {"orange_stage": 260, "arena_stage": 170,
                       "pavilion": 140, "apollo": 105}
    ft_roskilde = FestivalType(
        n_days=4, is_camping=True, camping_fraction=0.97,
        sold_out_fraction=1.0, has_official_resale=False,
    )
    tests.append(FestivalSpec(
        name="Roskilde_DK",
        description="130k 4-day camping sell-out (Roskilde DK) — 8 days total incl. warm-up",
        scenario=make_scenario(
            "Roskilde_DK", a_total=130000, t_evac_min=10,
            festival_type=ft_roskilde,
            n_stages=4, staggered_end_times=True,
            gate_open_hour=9, headliner_start_hour=22,
        ),
        zones=zones_from_areas(roskilde_areas, roskilde_exits,
                               roskilde_shares, roskilde_stalls),
        expect_ever_admitted_range=(115000, 145000),  # day-1 + small day-tripper days 2-4
        expect_peak_occ_range=(80000, 130000),
        expect_feasible_pct_min=0.40,
        expect_mean_density_max=2.5,
    ))

    return tests


# ── Runner ─────────────────────────────────────────────────────────────────────

def run_tests(tests: list[FestivalSpec], n_runs: int = N_RUNS) -> pd.DataFrame:
    rows = []
    for spec in tests:
        print(f"  Running {spec.name} ({spec.scenario.festival_type.n_days}d, "
              f"{spec.scenario.a_total:,} tickets)...", end=" ", flush=True)
        df  = monte_carlo(spec.scenario, n_runs=n_runs, seed=SEED, zones=spec.zones)
        s   = summarize_alternative(df)

        # Assertions
        failures = []
        ea  = s["mean_ever_admitted"]
        po  = s["mean_peak_occ"]
        feas = s["frac_feasible"]
        dens = s["mean_D"]

        if spec.expect_ever_admitted_range:
            lo, hi = spec.expect_ever_admitted_range
            if not (lo <= ea <= hi):
                failures.append(f"ever_admitted {ea:,.0f} not in [{lo:,}, {hi:,}]")

        if spec.expect_peak_occ_range:
            lo, hi = spec.expect_peak_occ_range
            if not (lo <= po <= hi):
                failures.append(f"peak_occ {po:,.0f} not in [{lo:,}, {hi:,}]")

        if feas < spec.expect_feasible_pct_min:
            failures.append(
                f"feasible {feas:.0%} < min {spec.expect_feasible_pct_min:.0%}")

        if dens > spec.expect_mean_density_max:
            failures.append(
                f"mean_density {dens:.3f} > max {spec.expect_mean_density_max:.1f}")

        status = "PASS" if not failures else "FAIL"
        print(status)
        if failures:
            for f in failures:
                print(f"    ✗ {f}")

        rows.append({
            "Festival":        spec.name,
            "Days":            spec.scenario.festival_type.n_days,
            "Tickets":         f"{spec.scenario.a_total:,}",
            "Camping%":        f"{spec.scenario.festival_type.camping_fraction:.0%}",
            "Stages":          spec.scenario.n_stages,
            "Staggered":       "Y" if spec.scenario.staggered_end_times else "N",
            "E[u_O]":          f"{s['E_uO']:,.0f}",
            "E[u_A]":          f"{s['E_uA']:.1f}",
            "Feasible%":       f"{s['frac_feasible']*100:.0f}%",
            "Peak occ.":       f"{s['mean_peak_occ']:,.0f}",
            "Ever admitted":   f"{s['mean_ever_admitted']:,.0f}",
            "Mean D (p/m²)":   f"{s['mean_D']:.3f}",
            "Mean Q/stall":    f"{s['mean_Q']:.2f}",
            "Status":          status,
        })

    return pd.DataFrame(rows)


def print_report(df: pd.DataFrame) -> None:
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 160)
    pd.set_option("display.max_colwidth", 22)
    print("\n" + "=" * 160)
    print("MULTI-FESTIVAL SIMULATION RESULTS")
    print("=" * 160)
    print(df.to_string(index=False))
    print("=" * 160)
    n_pass = (df["Status"] == "PASS").sum()
    n_fail = (df["Status"] == "FAIL").sum()
    print(f"\n  {n_pass}/{len(df)} PASS   {n_fail}/{len(df)} FAIL\n")


if __name__ == "__main__":
    print(f"Running multi-festival test suite ({N_RUNS} reps each)...\n")
    tests  = build_test_festivals()
    results = run_tests(tests, n_runs=N_RUNS)
    print_report(results)
