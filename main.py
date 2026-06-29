"""
Festival OR Model - Multi-Zone Dynamic Simulation Toolkit
Extends or_des_toolkit_week6.py (ORSL Week 6) with:
  - A ZoneSpec/Zone framework so new zones can be instantiated as needed
  - A weather Markov chain w(t) (Section 22.4 of the model document)
  - Inter-zone crowd movement m_z(t) scaled by boundary width (Section 22)
  - Severity-weighted incident sampling (Section 17 / 22.9)
  - Violation/noise accumulation V(t), nu(t) (Section 22.10)
  - Aggregation into u_O, u_G, u_A (Section 24)
  - Visualisation: plot_day_overview, plot_zone_density, plot_vendor_queues,
                   plot_incidents, plot_policy_comparison, print_summary_table

FIXES applied vs earlier versions:
  1. Arrival profile: proper log-normal 70/30 shape (Section 22.5)
     replacing the borrowed week-6 Gaussian bump, fixing under-admission
  2. u_O: Q term restored  (Section 13.4: 3R - 2C - D - Q - Phi)
  3. u_G: x10 multiplier removed; infrastructure strain I added as
     attendance-scaled proxy  (Section 6.2: 2*E - N - 3*V - 2*T_evac - I)
  4. u_A: now computed and returned  (Section 6.3)
  5. Phi: per-zone squaring of critical incidents (Section 13.2 notation)
  6. V: single combined flag per step to prevent double-increment cascade
  7. Entrance surge threshold: per-entrance queue (not * n_lanes)
  8. v_max_violation raised to 5 to avoid premature evacuation from
     single-step double-increment
"""

from __future__ import annotations
from dataclasses import dataclass, field
import math
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker



# ============================================================
# VISUAL STYLE
# ============================================================

plt.rcParams.update({
    "figure.facecolor":   "#0F1117",
    "axes.facecolor":     "#1A1D27",
    "axes.edgecolor":     "#2E3147",
    "axes.labelcolor":    "#C8CDD8",
    "axes.titlecolor":    "#E8EAF0",
    "axes.titlesize":     11,
    "axes.labelsize":     9,
    "axes.grid":          True,
    "grid.color":         "#2E3147",
    "grid.linewidth":     0.6,
    "xtick.color":        "#7A7F94",
    "ytick.color":        "#7A7F94",
    "xtick.labelsize":    8,
    "ytick.labelsize":    8,
    "legend.facecolor":   "#1E2235",
    "legend.edgecolor":   "#2E3147",
    "legend.labelcolor":  "#C8CDD8",
    "legend.fontsize":    8,
    "text.color":         "#E8EAF0",
    "figure.titlesize":   13,
    "figure.titleweight": "bold",
    "lines.linewidth":    2.0,
    "savefig.facecolor":  "#0F1117",
})

ZONE_COLOURS    = {"alpha_main_stage": "#4FC3F7", "cape_lowlands": "#FFB74D",
                   "planet_paradise":  "#81C784", "camping":       "#CE93D8"}
_DEFAULT_COLOURS = ["#4FC3F7", "#FFB74D", "#81C784", "#CE93D8", "#F48FB1", "#80CBC4"]

def _zone_colour(zone_name: str, idx: int) -> str:
    return ZONE_COLOURS.get(zone_name, _DEFAULT_COLOURS[idx % len(_DEFAULT_COLOURS)])
WEATHER_COLOURS = {"clear": "#FFE082", "rain": "#64B5F6", "heat": "#EF9A9A"}
ALT_COLOURS     = {"A1": "#78909C", "A2": "#4DB6AC", "A3": "#FF8A65", "A4": "#BA68C8"}
ACCENT  = "#7C83FD"
DANGER  = "#EF5350"
WARNING = "#FFB74D"
SUCCESS = "#66BB6A"


# ============================================================
# WEATHER PROCESS  (Section 22.4)
# ============================================================

WEATHER_STATES = ["clear", "rain", "heat"]
T_WEATHER = np.array([
    [0.95, 0.04, 0.01],
    [0.08, 0.91, 0.01],
    [0.05, 0.02, 0.93],
])
# Density thresholds (all values are zone-level spatial averages: total people / usable m²).
# Zone averages map to local front-of-stage densities roughly 2.5× higher.
#
# D_MAX_BY_WEATHER — Purple Guide permit standard (unchanged). Used for A_max /
#                    ticket capacity calculations only. Do NOT use for violation checks.
# THETA_WARN       — zone avg 1.5 ≈ local front ~3.5 p/m²: normal dense festival.
#                    Fires a soft warning; does NOT increment V(t).
# THETA_VIOLATION  — zone avg 1.8 ≈ local front ~4.5 p/m²: intervention required.
#                    Increments V(t). Relaxed to D_MAX_PERFORMANCE during scheduled acts.
D_MAX_BY_WEATHER         = {"clear": 2.0, "rain": 1.7, "heat": 1.5}
THETA_WARN               = {"clear": 1.5, "rain": 1.3, "heat": 1.1}
THETA_VIOLATION          = {"clear": 1.8, "rain": 1.5, "heat": 1.3}
D_MAX_PERFORMANCE        = {"clear": 4.0, "rain": 3.2, "heat": 2.8}
PERFORMANCE_WINDOW_STEPS = 4   # ±4 steps (±1 hour) around each scheduled act
PHI_BY_WEATHER       = {"clear": 82.0, "rain": 70.0, "heat": 75.0}
COST_MULT_BY_WEATHER = {"clear": 1.00, "rain": 1.15, "heat": 1.40}
# ============================================================
# FESTIVAL TYPE & DYNAMIC NO-SHOW MODEL
# ============================================================
#
# No-show rate depends on four festival attributes (research basis):
#
#   1. is_camping: Camping festivals have large sunk costs — accommodation,
#      travel, gear all pre-arranged months ahead. Campers on-site by Friday
#      cannot no-show for Saturday/Sunday. (Ticket Fairy; Billboard; empirical
#      Lowlands/Glastonbury/Roskilde data)
#
#   2. sold_out_fraction (0–1): Sell-out events have near-zero no-shows
#      because anyone who can't attend resells at face value or above.
#      Unsold-capacity events have casual buyers with lower commitment.
#      (Billboard pre-pandemic baseline 5%; sell-out festivals empirically ~1-3%)
#
#   3. n_days: Multi-day festivals require more planning commitment than
#      single-day. A 3-day camping ticket is ~3× harder to abandon than a
#      Saturday afternoon pass. Each additional day reduces no-show rate.
#      (Ticket Fairy multi-day vs single-day analysis; industry planning data)
#
#   4. has_official_resale: Official resale (Ticketmaster, DICE, TicketSwap
#      with price cap) absorbs would-be no-shows into new buyers, compressing
#      effective no-show rate. Lowlands closes resale Thu 05:00 before festival
#      — by gate-open virtually all tickets are in the hands of attendees.
#      (lowlands.nl/tickets/tickets-doorverkopen; Festileaks 2024/2025)
#
# Weather sensitivity: rain/heat reduces day-tripper attendance but
# campers already on site cannot no-show. So weather multiplier scales
# with (1 - camping_fraction). Campers on a rain day don't leave —
# they put on a poncho.
#
# Formula:
#   base_noshow = BASE_RATES[n_days] * (1 - sold_out_fraction * SELLOUT_FACTOR)
#                 * (1 - is_camping * CAMPING_DISCOUNT)
#                 * (1 - has_official_resale * RESALE_DISCOUNT)
#   weather_sensitivity = (1 - camping_fraction)
#   noshow(w) = base_noshow * (1 + WEATHER_UPLIFT[w] * weather_sensitivity)
#   clamped to [0.005, 0.35]
#
# Sources:
#   Billboard (2022): pre-pandemic baseline 5%, post-pandemic up to 15–20%
#   Ticket Fairy (2025): festival planning baseline 5–10%; camping reduces
#   Gitnux (2026): 60% of US festival-goers camp, spending $300 on accommodation
#   Lowlands empirical: ~1% (all 65,000 attend, maxiaxi.com 2025)
#   Glastonbury: sells out in <1hr, full capacity (gitnux 2026)
#   Roskilde: 100,000 tickets fully attended (billetto 2025)

@dataclass(frozen=True)
class FestivalType:
    """
    Describes the structural characteristics of a festival that drive
    no-show rates. Pass into FestivalScenario to get dynamic no-show rates.

    Parameters
    ----------
    n_days : int
        Number of festival days (1 = single-day, 3 = Lowlands-style weekend)
    is_camping : bool
        Whether the festival includes on-site camping (sunk-cost commitment)
    camping_fraction : float
        Fraction of attendees who camp on site (0–1). Drives weather sensitivity.
    sold_out_fraction : float
        How sold-out the event is (0 = tickets still available, 1 = fully sold out)
    has_official_resale : bool
        Whether an official face-value resale platform absorbs no-show tickets
    """
    n_days:                  int   = 1
    is_camping:              bool  = False
    camping_fraction:        float = 0.0
    sold_out_fraction:       float = 0.5
    has_official_resale:     bool  = False
    multiday_ticket_fraction: float = 1.0   # fraction of ticket holders with a pass for ALL days
                                             # 1.0 = everyone has a full-weekend ticket
                                             # 0.5 = half bought single-day, half bought weekend

    def noshow_rates(self) -> Dict[str, float]:
        """
        Compute no-show rate per weather state dynamically from festival attributes.
        Returns dict {"clear": float, "rain": float, "heat": float}.
        """
        # Base rates by number of days (single-day most volatile, multi-day committed)
        BASE_BY_DAYS   = {1: 0.12, 2: 0.08, 3: 0.05}
        base = BASE_BY_DAYS.get(self.n_days, 0.05 / self.n_days)

        # Adjustments (multiplicative discounts)
        SELLOUT_FACTOR  = 0.60   # full sell-out reduces no-show by up to 60%
        CAMPING_DISCOUNT = 0.55  # camping commitment reduces no-show by up to 55%
        RESALE_DISCOUNT  = 0.35  # official resale absorbs up to 35% of residual no-shows

        base *= (1.0 - self.sold_out_fraction   * SELLOUT_FACTOR)
        base *= (1.0 - float(self.is_camping)   * CAMPING_DISCOUNT)
        base *= (1.0 - float(self.has_official_resale) * RESALE_DISCOUNT)

        # Weather uplift — only affects day-trippers (non-campers)
        # Campers already on site cannot no-show due to weather
        WEATHER_UPLIFT = {"clear": 0.0, "rain": 0.80, "heat": 0.25}
        day_tripper_fraction = 1.0 - self.camping_fraction

        rates = {}
        for w, uplift in WEATHER_UPLIFT.items():
            r = base * (1.0 + uplift * day_tripper_fraction)
            rates[w] = float(np.clip(r, 0.005, 0.35))  # floor 0.5%, ceiling 35%
        return rates


# ── Pre-built festival type profiles ─────────────────────────────────────────
# These cover the main archetypes; override any field for custom scenarios.

FESTIVAL_TYPE = {
    # Large multi-day camping sell-out (Lowlands, Glastonbury, Roskilde,
    # Wacken, Bonnaroo). Near-zero no-shows — official resale absorbs dropouts,
    # campers on site from Thursday, full sunk cost.
    "camping_sellout": FestivalType(
        n_days=3, is_camping=True, camping_fraction=0.95,
        sold_out_fraction=1.0, has_official_resale=True),

    # Multi-day camping but not a sell-out (e.g. mid-tier festivals with
    # remaining capacity). Some casual buyers, weather sensitivity present
    # for day-trippers.
    "camping_general": FestivalType(
        n_days=3, is_camping=True, camping_fraction=0.70,
        sold_out_fraction=0.75, has_official_resale=False),

    # Single-day festival, sold out (e.g. a single-day pop/dance event at
    # capacity). High weather sensitivity for all attendees since nobody camps.
    "singleday_sellout": FestivalType(
        n_days=1, is_camping=False, camping_fraction=0.0,
        sold_out_fraction=1.0, has_official_resale=False),

    # Single-day festival, not sold out — most volatile scenario.
    # High base no-show + full weather sensitivity.
    "singleday_general": FestivalType(
        n_days=1, is_camping=False, camping_fraction=0.0,
        sold_out_fraction=0.5, has_official_resale=False),

    # Two-day festival (e.g. Defqon.1, Download Festival).
    # Mix of campers and day-trippers.
    "twoday_camping": FestivalType(
        n_days=2, is_camping=True, camping_fraction=0.60,
        sold_out_fraction=0.90, has_official_resale=False),
}

# Convenience: show what rates each profile produces
def print_noshow_table() -> None:
    print("\nDynamic no-show rates by festival type and weather:")
    print(f"{'Type':<22} {'Clear':>7} {'Rain':>7} {'Heat':>7}")
    print("-" * 48)
    for name, ft in FESTIVAL_TYPE.items():
        r = ft.noshow_rates()
        print(f"{name:<22} {r['clear']:>6.1%} {r['rain']:>6.1%} {r['heat']:>6.1%}")
    print()
INCIDENT_MULT = {
    "clear": (1.0, 1.0, 1.0),
    "rain":  (1.3, 1.5, 1.2),
    "heat":  (2.0, 2.5, 3.0),
}

def draw_initial_weather(rng: np.random.Generator) -> str:
    return rng.choice(WEATHER_STATES, p=[0.45, 0.40, 0.15])

def step_weather(w: str, rng: np.random.Generator) -> str:
    return rng.choice(WEATHER_STATES, p=T_WEATHER[WEATHER_STATES.index(w)])


# ============================================================
# ARRIVAL AND DEPARTURE PROFILES  (Section 22.5)
# Research basis (Ticket Fairy / industry sources):
#   - Arrivals have TWO peaks: gate-open rush + pre-headliner surge
#   - Gate opens 10:00; first peak ~11:00-12:00 (t=4-8); second peak
#     before headliner ~18:00-20:00 (t=32-40) which is roughly 40% through
#     the 64-step day.  Approx split: 60% first wave, 40% second wave.
#   - Departures are sharp and compressed: almost no one leaves until the
#     last 2-3 hours, then a rapid exodus around headliner end (~23:00,
#     t=52) with a long tail until gate close at 02:00 (t=64).
#   - For a camping festival (Lowlands) on days 2-3, the second-wave
#     pre-headliner spike is stronger because day campers are already inside.
# ============================================================

def build_bimodal_arrival_fraction(n_steps: int = 64,
                                   gate_open_hour: int = 10,
                                   headliner_start_hour: int = 19,
                                   n_stages: int = 1,
                                   day: int = 1,
                                   camping_fraction: float = 0.0,
                                   ) -> np.ndarray:
    """
    Bimodal arrival profile for one day: gate-open rush + pre-headliner surge.

    Parameters derived from research (Ticket Fairy 2025-2026):
    - Single-stage evening: narrow peaks, strong pre-headliner surge (40/60 split)
    - Multi-stage all-day: wider softer peaks, gate-open weighted (60/40)
    - Camping days 2+: campers already inside, single gate-open peak only
    """
    steps = np.arange(n_steps, dtype=float)

    # Peak 1: ~1hr after gate open (gate-open rush)
    peak1_step = 4  # 1 hour after gate open = 4 steps

    # Peak 2: ~1hr before headliner starts (pre-headliner surge)
    peak2_step = int((headliner_start_hour - gate_open_hour - 1) * 4)
    peak2_step = max(peak1_step + 4, min(peak2_step, n_steps - 4))

    # Shape parameters depend on festival type (research table)
    if n_stages == 1:
        # Single-stage evening: sharp clustered arrivals, strong pre-headliner
        sigma1, sigma2 = 2.0, 2.5
        weight1, weight2 = 0.40, 0.60
    else:
        # Multi-stage all-day: staggered arrivals, softer peaks, gate-open weighted
        sigma1 = 2.5 if camping_fraction < 0.3 else 3.5
        sigma2 = 5.0
        weight1, weight2 = 0.60, 0.40  # 60% first wave, 40% pre-headliner

    g1 = np.exp(-0.5 * ((steps - peak1_step) / sigma1) ** 2)

    if day > 1 and camping_fraction >= 0.5:
        # Camping days 2+: campers already inside, single gate-open peak only
        raw = g1
    else:
        g2 = np.exp(-0.5 * ((steps - peak2_step) / sigma2) ** 2)
        raw = weight1 * g1 + weight2 * g2

    raw = np.maximum(raw, 0.0)
    return raw / raw.sum()


def build_departure_fraction(n_steps: int = 64,
                             dt_hours: float = 0.25,
                             day: int = 2,
                             total_days: int = 3,
                             camping_fraction: float = 0.95,
                             n_stages: int = 1,
                             staggered_end_times: bool = False,
                             headliner_end_step: int = 52,  # ~23:00
                             ) -> np.ndarray:
    """
    Festival-type-aware departure profile (Ticket Fairy / industry research).

    Non-last days:
      - Camping festival: only ~15% depart (day-trippers). Low camping_fraction
        means more day-trippers → higher trickle rate (up to ~35%).
      - Day-tripper festival (camping_fraction~0): 30-40% leave each non-last day.

    Last day (day == total_days):
      - Single-stage hard end: sharp spike, ~95% leave in a 1hr window. Dangerous.
      - Multi-stage simultaneous end: broader spike, ~70% in 2hr window.
      - Multi-stage staggered ends: broadest, safest, ~60% in 2hr window.
      - Camping last day adds a broad camper tail (packing up tents) on top of spike.
    """
    steps = np.arange(n_steps, dtype=float)

    if day < total_days:
        # Non-last day departures:
        #   - Pure campers (camping_fraction=1): ~15% depart (a few day-trippers/early leavers)
        #   - Pure day-trippers (camping_fraction=0): ~100% depart — everyone goes home
        #   - Mixed: interpolate linearly between the two
        # For day-tripper festivals the shape is still a late-evening concentration
        # (people don't leave mid-show), but the total fraction is much higher.
        trickle_rate = 0.15 * camping_fraction + 1.00 * (1.0 - camping_fraction)
        trickle = np.zeros(n_steps)
        trickle[44:] = 1.0   # gradual from ~21:00 onward
        raw = trickle / trickle.sum()
        return raw * trickle_rate
    else:
        # Last day: shape depends on stage structure
        if n_stages == 1:
            # Single headliner hard end: sharp spike, minimal tail (Ultra Miami scenario)
            sigma = 2.5
            spike_w, tail_w = 0.95, 0.05
        elif staggered_end_times:
            # Multi-stage staggered: broadest and safest profile
            sigma = 4.5
            spike_w, tail_w = 0.60, 0.40
        else:
            # Multi-stage simultaneous end: intermediate
            sigma = 3.5
            spike_w, tail_w = 0.70, 0.30

        spike = np.exp(-0.5 * ((steps - headliner_end_step) / sigma) ** 2)

        if camping_fraction >= 0.3:
            # Camping festivals: add a broad tail for tent pack-up (1-4hrs after headliner)
            tail_centre = headliner_end_step + int(2.0 / dt_hours)   # 2 hours later
            camper_tail = np.exp(-0.5 * ((steps - tail_centre) / 6.0) ** 2)
            raw = spike_w * spike / spike.sum() + tail_w * camper_tail / camper_tail.sum()
        else:
            # Day-tripper festival: sharp exit, no tent-packing tail
            raw = spike

        raw = np.maximum(raw, 0.0)
        return raw / raw.sum()  # 100% depart on last day


# ============================================================
# ZONE FRAMEWORK  (Section 22.1)
# ============================================================

@dataclass(frozen=True)
class ZoneSpec:
    """Static definition of a zone — instantiate one per zone."""
    name: str
    area_m2: float
    n_gates: int                            # kept for compatibility; unused since shared entrance
    gate_base_throughput_per_hr: float = 400.0
    exit_width_m: float = 0.0              # W_l for egress A_max
    arrival_share: float = 0.0             # fraction of festival-wide dispersal
    v_z: int = 1                           # vendor stalls
    adjacent: Tuple[str, ...] = field(default_factory=tuple)
    is_exogenous: bool = False             # True for zones managed on separate schedule (e.g. camping)


# Two entrances — ENTRANCE_SHARE derived from lane ratio (Section 22)
# ── Realistic Lowlands zone topology (Section 22.1) ──────────────────
# Total site: ~60 ha (Walibi Holland event terrain, Festival Fans NL)
# Festival terrain (3 areas): ~10.5 ha usable crowd space
# Camping: ~27 ha across 7 sections (Festileaks 2023/2024)
#
# Three festival areas (Festileaks 2023 plattegrond):
#   Alpha: main stage area (tent ≈ 1 football pitch = 7,140 m²,
#          plus standing crowd space) → ~28,000 m²
#   Cape Lowlands (south): Bravo, India, X-Ray, Hacienda, ArmadiLLow → ~42,000 m²
#   Planet Paradise (north): Heineken, Lima, Echo, Juliet + food village → ~35,000 m²
#
# Walk time Alpha↔Bravo without crowds: ~15 min (Festileaks); with crowds: ~30 min.
# This implies ~600–800m between the two largest stages.
#
# Exit widths: at 65,000 people and 70 p/m (SGSA standard), ~929m total needed.
# Distributed proportionally across zones.
#
# Camping is modelled as a separate zone (separate terrain, own entrance/exit,
# 7 sections). Camping density is much lower than festival terrain — people
# have tent pitches of ~4–9 m²/person, giving 0.1–0.25 p/m².

ENTRANCE_LANES   = {"main": 35, "secondary": 22}
_total_lanes     = sum(ENTRANCE_LANES.values())
ENTRANCE_SHARE   = {n: v / _total_lanes for n, v in ENTRANCE_LANES.items()}
ENTRANCE_AREA_M2 = {"main": 2000.0, "secondary": 1200.0}

DEFAULT_ZONES: Dict[str, ZoneSpec] = {
    # Alpha: main stage zone
    # Gross area ~28,000 m² — infrastructure deducted:
    #   Stage deck + production (40% of 7,140m² tent): 2,850 m²
    #   Backstage compound (fenced artist/crew area):   1,500 m²
    #   FOH mixing tower + exclusion zone:                300 m²
    #   Crush barrier zone (5m deep front of stage):      500 m²
    #   Toilets, first aid post, access paths:           1,200 m²
    # Usable crowd area: 21,650 m² (77% of gross)
    # Exit width: 43% of 929m total ≈ 400m
    # v_z=113: ceil(21,650 × 0.431 × 65000/78850 / 250)
    "alpha_main_stage": ZoneSpec(
        "alpha_main_stage", area_m2=21650, n_gates=0,
        exit_width_m=400, arrival_share=0.431, v_z=113,
        adjacent=("cape_lowlands", "planet_paradise")),

    # Cape Lowlands: Bravo + India + X-Ray + Hacienda + ArmadiLLow (5 stages)
    # Gross area ~42,000 m² — infrastructure deducted:
    #   Bravo stage + production (40% of ~5,000m² tent):  2,000 m²
    #   India stage + production (40% of ~2,500m² tent):  1,000 m²
    #   3 smaller stages (X-Ray, Hacienda, ArmadiLLow):   1,800 m²
    #   Backstage compounds (5 stages × ~600m²):          3,000 m²
    #   Walkways, toilets, service paths between stages:   3,500 m²
    # Usable crowd area: 30,700 m² (73% of gross)
    # Exit width: 25% of 929m ≈ 232m
    # v_z=64: ceil(30,700 × 0.246 × 65000/78850 / 250)
    "cape_lowlands": ZoneSpec(
        "cape_lowlands", area_m2=30700, n_gates=0,
        exit_width_m=232, arrival_share=0.246, v_z=64,
        adjacent=("alpha_main_stage", "camping")),

    # Planet Paradise: Heineken + Lima + Echo + Juliet + food village
    # Gross area ~35,000 m² — infrastructure deducted:
    #   Heineken stage + production (40% of ~3,000m²):    1,200 m²
    #   3 smaller stages (Lima, Echo, Juliet):             1,800 m²
    #   Food vendor back-of-house (~200 stalls × 15m²):   3,000 m²
    #   Paths, toilets, art installations (non-walkable):  2,500 m²
    # Usable crowd area: 26,500 m² (76% of gross)
    # Exit width: 19% of 929m ≈ 177m
    # v_z=122: food hub premium — serves adjacent zones too (2.5× base)
    "planet_paradise": ZoneSpec(
        "planet_paradise", area_m2=26500, n_gates=0,
        exit_width_m=177, arrival_share=0.185, v_z=122,
        adjacent=("alpha_main_stage", "cape_lowlands", "camping")),

    # Camping: 7 sections (lowlands.nl/camping; Ticketmaster UK official)
    # "Virtually all 65,000 visitors camp overnight" (maxiaxi.com 2025)
    # Area: 65,000 × ~4.2 m²/person pitch ≈ 273,000 m² (27.3 ha) across 7 sections
    # Note: 4.2 m²/person already represents usable tent pitch space —
    #       no infrastructure deduction needed (paths/showers excluded from pitch density)
    # Population modelled as EXOGENOUS SCHEDULE (wristband checkpoint separates
    # camping from festival terrain — people do not flow via density gradient)
    # Exit width: 120m to car parks / shuttle buses (separate from festival exits)
    # v_z=25: fewer stalls — campers self-cater; camping bars serve drinks only
    "camping": ZoneSpec(
        "camping", area_m2=273000, n_gates=0,
        exit_width_m=120, arrival_share=0.0,  # camping receives NO gate arrivals
        v_z=25,
        adjacent=("cape_lowlands", "planet_paradise"),
        is_exogenous=True),
}

# Boundary widths between zones — based on ~600-800m Alpha↔Cape walk (Festileaks)
# Festival terrain is open fields so connecting widths are wide walkways
BOUNDARY_WIDTH_M = {
    frozenset({"alpha_main_stage", "cape_lowlands"}):   80.0,   # main walkway between Alpha and Cape
    frozenset({"alpha_main_stage", "planet_paradise"}): 100.0,  # Planet Paradise is central, wide approach
    frozenset({"cape_lowlands",    "camping"}):          150.0, # camping abuts southern festival area
    frozenset({"planet_paradise",  "camping"}):          200.0, # widest connection — main camping→festival route
    frozenset({"cape_lowlands",    "planet_paradise"}):  60.0,  # walkway between southern and central areas
}


# ============================================================
# SEVERITY-WEIGHTED INCIDENTS  (Section 17 / 22.9)
# ============================================================

RATE_PER_1000 = {"minor": 10.0, "moderate": 0.57, "critical": 0.003}
OMEGA         = {"minor": 1.0,  "moderate": 10.0,  "critical": 1000.0}

def sample_incidents(a_z: float, weather: str, dt_hours: float,
                     rng: np.random.Generator) -> Dict[str, int]:
    mult = dict(zip(["minor", "moderate", "critical"], INCIDENT_MULT[weather]))
    return {tier: rng.poisson(max((RATE_PER_1000[tier]/1000)*mult[tier]*a_z*(dt_hours/16), 0))
            for tier in RATE_PER_1000}


# ============================================================
# QUEUE THRESHOLDS  (Section 21)
# sigma_v = 30/hr (2-min transaction); rho_v = 0.047/step (~3 visits/person/day)
# Both calibrated so utilisation rho < 1 at peak, preventing unbounded growth.
# ============================================================

RHO_V          = 0.047           # vendor demand fraction per 15-min step
SIGMA_V        = 30.0 / 60.0    # vendor service rate: orders/min/stall (30/hr)
G_BASE_PER_MIN = 400.0 / 60.0   # gate scan rate: scans/min/lane

Q_MAX_VENDOR = SIGMA_V * 10.0    # 10-min equivalent queue per stall → intervention
Q_MAX_GATE   = G_BASE_PER_MIN * 50.0  # 50-min equivalent queue → surge trigger


# ============================================================
# ATTENDEE UTILITY CONSTANTS  (Section 6.3 — FIX 4)
# u_A = 3J + 2S + H + P - 3W - 2D - T_evac - I
# J, S, H, P are latent; we proxy them from simulated outputs.
# ============================================================

U_A_BASE       = 33.0   # base enjoyment when festival runs smoothly
GAMMA_P        = 0.03   # sunk-cost sensitivity (ticket price)
TICKET_PRICE   = 365.0  # Lowlands weekend ticket (EUR)
U_A_MIN        = U_A_BASE - GAMMA_P * TICKET_PRICE  # = 22.05

# Scaling factors to map simulated quantities onto u_A terms
ALPHA_A = 3.0   # enjoyment weight
LAMBDA_A = 3.0  # waiting-time penalty weight
MU_A = 2.0      # density penalty weight
NU_A = 1.0      # evacuation-time penalty weight
RHO_A = 1.0     # infrastructure-strain penalty weight


# ============================================================
# RUNTIME STATE
# ============================================================

@dataclass
class EntranceState:
    name: str
    q: float = 0.0   # people queuing at this entrance

@dataclass
class ZoneState:
    spec: ZoneSpec
    a: float = 0.0
    q_vendor: float = 0.0
    extra_stalls: int = 0
    incidents_cum: Dict[str, int] = field(
        default_factory=lambda: {"minor": 0, "moderate": 0, "critical": 0})

    @property
    def density(self) -> float:
        return self.a / self.spec.area_m2

    @property
    def v_z_effective(self) -> int:
        return self.spec.v_z + self.extra_stalls


# ============================================================
# COST MODEL  (Fix 3 — dynamic cost function)
# ============================================================

@dataclass(frozen=True)
class CostParams:
    # Sources: VVNL CAO Veiligheidsdomein 2025-2027 (NL), Aljohani & Kennedy 2016,
    # Dutch municipal penalty schedule, NL festival operator interviews.
    omega_s:          float = 400.0    # €/person/day — VVNL CAO 2026 ~€40/hr × 10hr shift
    omega_surge:      float = 520.0    # €/activation — 2 staff × €42/hr × 6hr + admin
    omega_v:          float = 350.0    # €/stall/day — organiser infrastructure cost
    omega_t:          float = 110.0    # €/cubicle/day — NL festival rate incl. mid-day service
    omega_f:          float = 2000.0   # €/bay/day — Aljohani & Kennedy 2016
    omega_l:          float = 3100.0   # €/route — 6 stewards × €42/hr × 9hr + equipment
    omega_viol:       float = 10000.0  # €/violation — Dutch municipal penalty mid-range
    omega_g:          float = 0.20     # €/scan — RFID equipment amortised (calibrated)
    omega_z:          float = 8000.0   # €/zone/day — conservative planning estimate
    # Gate infrastructure
    omega_lane:       float = 200.0    # €/lane/day — portable RFID scanner rental incl. setup
    omega_gate_staff: float = 400.0    # €/gate staff/day — VVNL CAO 2026 (1 per lane)

DEFAULT_COST_PARAMS = CostParams()


def compute_cost(
    timeline: "pd.DataFrame",
    zone_timeline: "pd.DataFrame",
    scenario: "FestivalScenario",
    zones: Dict[str, ZoneSpec],
    cost_params: CostParams = DEFAULT_COST_PARAMS,
    total_lanes: int = 0,
) -> float:
    """Compute total operational cost from actual zone usage and events.

    total_lanes: sum of all turnstile lanes across all entrance gates.
                 Each lane requires one dedicated gate staff member.
    """
    festival_zone_names = [n for n, s in zones.items() if not s.is_exogenous]
    zone_cost = 0.0
    for zname in festival_zone_names:
        ztl_z = zone_timeline[zone_timeline["zone"] == zname]
        a_z = ztl_z["a_z"].max()
        v_z = zones[zname].v_z + ztl_z["extra_stalls"].max()
        s_z = np.ceil(a_z / 100)
        t_z = np.ceil(a_z / 75)
        f_z = np.ceil(a_z / 5000)
        zone_cost += (s_z * cost_params.omega_s
                    + v_z * cost_params.omega_v
                    + t_z * cost_params.omega_t
                    + f_z * cost_params.omega_f
                    + cost_params.omega_z)
    surge_activations = timeline["entrance_surge"].sum()
    total_scans = timeline["total_ever_admitted"].iloc[-1]
    violations = timeline["V"].iloc[-1]
    weather_mult = timeline["weather"].map(COST_MULT_BY_WEATHER).mean()
    # Gate infrastructure: lane hire + one staff member per lane
    gate_cost = total_lanes * (cost_params.omega_lane + cost_params.omega_gate_staff)
    return (zone_cost * weather_mult
            + gate_cost
            + surge_activations * cost_params.omega_surge
            + total_scans * cost_params.omega_g
            + violations * cost_params.omega_viol) / 1000.0


# ============================================================
# FESTIVAL SIMULATION
# ============================================================

@dataclass(frozen=True)
class FestivalScenario:
    name: str
    a_total: int          # intended total sold attendance
    t_evac_min: int       # T_evac: 8 (strict) or 10 (lenient)
    ticket_price: float = 365.0
    dt_hours: float = 0.25
    kappa_m: float = 0.05        # inter-zone equilibration rate
    v_max_violation: int = 5
    # day is no longer a field — it is derived from t // 64 inside run_festival_once
    festival_type: FestivalType = field(
        default_factory=lambda: FESTIVAL_TYPE["camping_sellout"])
    seed: Optional[int] = None
    # Stage and schedule parameters (drive arrival/departure shapes)
    n_stages: int = 1                    # 1=single headliner, 2+=multi-stage
    staggered_end_times: bool = False    # multi-stage: are end times offset?
    gate_open_hour: int = 10             # wall-clock hour gate opens (10=10:00)
    headliner_start_hour: int = 19       # wall-clock hour headliner starts (~1hr before end)
    # Act changeover schedule for schedule-driven inter-zone pulses
    # List of (step_within_day, from_zone, to_zone) — CHANGEOVER_FRACTION moves at each
    act_schedule: Tuple[Tuple, ...] = field(default_factory=tuple)
    # Weather forecast: list of (weather_state, duration_hours) tuples in chronological order.
    # Duration is converted to steps internally (duration_hours / dt_hours).
    # e.g. [("rain", 24.0), ("clear", 48.0)] = rain day 1, clear days 2-3.
    # Steps beyond the forecast revert to pure Markov. Empty = pure Markov throughout.
    forecast: Tuple[Tuple[str, float], ...] = field(default_factory=tuple)
    # 0.0 = ignore forecast (pure Markov), 1.0 = treat forecast as certain.
    # At each forecasted step: use forecast state with p=forecast_confidence,
    # otherwise draw from Markov chain as normal.
    forecast_confidence: float = 0.0

    @property
    def horizon_steps(self) -> int:
        return 64 * self.festival_type.n_days

@dataclass(frozen=True)
class AlternativeA(FestivalScenario):
    """Convenience subclass for A1–A4."""
    pass

def make_alternative(label: str,
                     festival_type: FestivalType = None,
                     n_stages: int = 3,
                     staggered_end_times: bool = True,
                     gate_open_hour: int = 10,
                     headliner_start_hour: int = 23) -> FestivalScenario:
    """
    Build one of the four Lowlands-style alternatives (A1–A4).
    Runs the full multi-day festival as a single continuous simulation.
    festival_type defaults to camping_sellout (3-day Lowlands profile).
    """
    table = {
        "A1": dict(a_total=45000, t_evac_min=8),
        "A2": dict(a_total=45000, t_evac_min=10),
        "A3": dict(a_total=55000, t_evac_min=10),
        "A4": dict(a_total=65000, t_evac_min=10),
    }
    if label not in table:
        raise ValueError(f"Unknown alternative {label!r}")
    ft = festival_type or FESTIVAL_TYPE["camping_sellout"]
    return FestivalScenario(name=label, festival_type=ft,
                            n_stages=n_stages, staggered_end_times=staggered_end_times,
                            gate_open_hour=gate_open_hour,
                            headliner_start_hour=headliner_start_hour,
                            **table[label])


def make_scenario(name: str, a_total: int, t_evac_min: int,
                  festival_type_key: str = "camping_sellout",
                  festival_type: FestivalType = None,
                  n_stages: int = 1,
                  staggered_end_times: bool = False,
                  gate_open_hour: int = 10,
                  headliner_start_hour: int = 19,
                  **kwargs) -> FestivalScenario:
    """
    Generic factory for any festival scenario.

    Parameters
    ----------
    name : str
        Scenario label (e.g. "Glastonbury_A1", "MyFestival_Base")
    a_total : int
        Total tickets sold
    t_evac_min : int
        Maximum permitted evacuation time in minutes
    festival_type_key : str
        Key into FESTIVAL_TYPE dict. One of:
        "camping_sellout", "camping_general", "singleday_sellout",
        "singleday_general", "twoday_camping"
    festival_type : FestivalType, optional
        Custom FestivalType instance (overrides festival_type_key)
    day : int
        Day of multi-day festival (1=first, 2=middle, 3=last)

    Examples
    --------
    # Model a generic single-day festival
    scn = make_scenario("MyFest_Low", 20000, 8, "singleday_general")

    # Model a custom two-day camping festival with 80% camping fraction
    ft = FestivalType(n_days=2, is_camping=True, camping_fraction=0.80,
                      sold_out_fraction=0.95, has_official_resale=True)
    scn = make_scenario("DownloadFest", 80000, 10, festival_type=ft)
    """
    ft = festival_type or FESTIVAL_TYPE[festival_type_key]
    return FestivalScenario(name=name, a_total=a_total, t_evac_min=t_evac_min,
                            festival_type=ft,
                            n_stages=n_stages, staggered_end_times=staggered_end_times,
                            gate_open_hour=gate_open_hour,
                            headliner_start_hour=headliner_start_hour,
                            **kwargs)

def egress_capacity(zones: Dict[str, ZoneSpec], weather: str, t_evac_min: int) -> float:
    return sum(z.exit_width_m for z in zones.values()) * PHI_BY_WEATHER[weather] * t_evac_min

def holding_capacity(zones: Dict[str, ZoneSpec], d_min: float = 0.5) -> float:
    return sum(z.area_m2 for z in zones.values()) / d_min

def a_max(zones: Dict[str, ZoneSpec], weather: str, t_evac_min: int) -> float:
    return min(holding_capacity(zones), egress_capacity(zones, weather, t_evac_min))


def validate_scenario(scenario: FestivalScenario) -> None:
    assert scenario.a_total > 0, "a_total must be positive"
    assert scenario.t_evac_min > 0, "t_evac_min must be positive"
    assert 1 <= scenario.festival_type.n_days <= 8, "n_days must be 1–8"

def validate_festival_type(ft: FestivalType) -> None:
    assert ft.n_days >= 1, "n_days must be at least 1"
    assert 0.0 <= ft.camping_fraction <= 1.0, "camping_fraction must be in [0, 1]"
    assert 0.0 <= ft.sold_out_fraction <= 1.0, "sold_out_fraction must be in [0, 1]"


def run_festival_once(
    scenario: FestivalScenario,
    zones: Dict[str, ZoneSpec] = DEFAULT_ZONES,
    seed: Optional[int] = None,
    entrance_lanes: Optional[Dict[str, int]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run one full multi-day replication as a single continuous simulation.
    horizon_steps = 64 * n_days. Day number and within-day step are derived
    from t at runtime: day = t // 64 + 1, s = t % 64.
    Returns (timeline, zone_timeline) with a 'day' column added.
    """
    validate_scenario(scenario)
    validate_festival_type(scenario.festival_type)
    rng = np.random.default_rng(seed if seed is not None else scenario.seed)

    n_days       = scenario.festival_type.n_days
    total_steps  = scenario.horizon_steps   # = 64 * n_days
    camping_fraction = scenario.festival_type.camping_fraction

    _lane_map  = entrance_lanes if entrance_lanes is not None else ENTRANCE_LANES
    _lane_total = sum(_lane_map.values())
    _lane_share = {n: v / _lane_total for n, v in _lane_map.items()}

    states    = {n: ZoneState(s) for n, s in zones.items()}
    entrances = {n: EntranceState(n) for n in _lane_map}

    # Build per-step forecast lookup: forecast_state[t] = weather str or None.
    # Organizer supplies (state, duration_hours); we convert to steps via dt_hours.
    forecast_state: List[Optional[str]] = [None] * total_steps
    if scenario.forecast:
        cursor = 0
        for (fcast_w, dur_hours) in scenario.forecast:
            n_fcast_steps = max(1, round(dur_hours / scenario.dt_hours))
            for i in range(cursor, min(cursor + n_fcast_steps, total_steps)):
                forecast_state[i] = fcast_w
            cursor += n_fcast_steps
            if cursor >= total_steps:
                break

    weather   = draw_initial_weather(rng)
    # Override initial weather if forecast covers step 0
    if forecast_state[0] is not None and rng.random() < scenario.forecast_confidence:
        weather = forecast_state[0]

    # No-show rate based on opening-day weather (committed before festival starts)
    noshow_rates = scenario.festival_type.noshow_rates()
    a_eff_total  = scenario.a_total * (1 - noshow_rates[weather])

    # Gate arrivals per day:
    #   Day 1: everyone with a ticket for that day shows up through the gate.
    #   Day 2+: only people who return through the gate:
    #     - Campers are already on site — they don't gate-scan again.
    #     - Single-day ticket holders (1 - multiday_ticket_fraction) don't return.
    #     - Multi-day non-campers (hotel guests, locals) re-enter through the gate.
    day_tripper_fraction      = 1.0 - camping_fraction
    multiday_frac             = scenario.festival_type.multiday_ticket_fraction
    # returning_day_fraction: of total attendance, who re-enters the gate on days 2+
    returning_day_fraction    = day_tripper_fraction * multiday_frac
    gate_arrivals_by_day = {
        d: (a_eff_total if d == 1 else a_eff_total * returning_day_fraction)
        for d in range(1, n_days + 1)
    }

    # Festival terrain zones only (exogenous zones excluded)
    festival_zone_names = [n for n, s in zones.items() if not s.is_exogenous]
    total_festival_share = sum(zones[n].arrival_share for n in festival_zone_names)
    festival_arrival_share = {
        n: zones[n].arrival_share / total_festival_share
        for n in festival_zone_names
    }

    # Precompute per-day arrival fractions (64 steps each)
    arrival_frac_by_day = {
        d: build_bimodal_arrival_fraction(
            64,
            gate_open_hour=scenario.gate_open_hour,
            headliner_start_hour=scenario.headliner_start_hour,
            n_stages=scenario.n_stages,
            day=d,
            camping_fraction=camping_fraction,
        )
        for d in range(1, n_days + 1)
    }

    # Precompute per-day departure fractions (64 steps each)
    headliner_end_step = int((scenario.headliner_start_hour - scenario.gate_open_hour + 1) * 4)
    departure_frac_by_day = {
        d: build_departure_fraction(
            64,
            day=d,
            total_days=n_days,
            camping_fraction=camping_fraction,
            n_stages=scenario.n_stages,
            staggered_end_times=scenario.staggered_end_times,
            headliner_end_step=headliner_end_step,
        )
        for d in range(1, n_days + 1)
    }

    # Precompute camping population per day per within-day step.
    # Same sigmoid shape each morning — campers leave camping for festival terrain,
    # return at night (except last day: full exodus).
    n_campers = a_eff_total * camping_fraction
    _s64 = np.arange(64, dtype=float)
    _morning_out = 1.0 / (1.0 + np.exp(-0.8 * (_s64 - 6)))
    _evening_in  = 1.0 / (1.0 + np.exp( 0.6 * (_s64 - 50)))
    _camper_on_festival_normal   = np.clip(_morning_out - (1.0 - _evening_in) * 0.85, 0.05, 0.95)
    _camper_on_festival_last_day = np.clip(_morning_out, 0.05, 0.95)
    camping_pop_by_day = {}
    for d in range(1, n_days + 1):
        cof = _camper_on_festival_last_day if d == n_days else _camper_on_festival_normal
        camping_pop_by_day[d] = n_campers * (1.0 - cof)

    # Vendor meal multiplier — same pattern each day (anchored to within-day step)
    _lunch  = np.exp(-0.5 * ((_s64 - 8)  / 2.5) ** 2)
    _dinner = np.exp(-0.5 * ((_s64 - 32) / 3.0) ** 2)
    _snack  = np.exp(-0.5 * ((_s64 - 48) / 4.0) ** 2)
    _meal_raw = 0.3 + 1.0 * _lunch + 1.5 * _dinner + 0.8 * _snack
    vendor_mult_day = _meal_raw / _meal_raw.mean()

    V, nu, evacuated = 0, 0, False
    W_density = 0          # cumulative density warnings (soft tier, does not drive evacuation)
    total_ever_admitted = 0.0
    peak_occupancy      = 0.0
    fest_rows, zone_rows = [], []

    # Precompute the set of within-day steps that fall inside a performance window.
    # act_schedule entries are (step_within_day, from_zone, to_zone) — the step is
    # treated as the act changeover moment; the window covers ±PERFORMANCE_WINDOW_STEPS.
    performance_steps: set = set()
    if scenario.act_schedule:
        for (pulse_step, _fz, _tz) in scenario.act_schedule:
            for offset in range(-PERFORMANCE_WINDOW_STEPS, PERFORMANCE_WINDOW_STEPS + 1):
                performance_steps.add(pulse_step + offset)

    for t in range(total_steps):
        day = t // 64 + 1          # 1-indexed festival day
        s   = t % 64               # within-day step (0-63)

        d_max_w  = D_MAX_BY_WEATHER[weather]
        amax_now = a_max(zones, weather, scenario.t_evac_min)
        entrance_surge = False

        if evacuated:
            for st in states.values():
                st.a = 0.0
        else:
            # ── exogenous zones: set from daily schedule ───────────
            for n, spec in zones.items():
                if spec.is_exogenous:
                    states[n].a = camping_pop_by_day[day][s]

            # ── shared entrance queue ──────────────────────────────
            step_in = gate_arrivals_by_day[day] * arrival_frac_by_day[day][s]
            total_admitted = 0.0
            surge_map = {}
            for en, ent in entrances.items():
                ent.q += step_in * _lane_share[en]
                nl    = _lane_map[en]
                surge = ent.q > Q_MAX_GATE
                surge_map[en] = surge
                g_eff    = G_BASE_PER_MIN * nl * (1.6 if surge else 1.0)
                admitted = min(ent.q, g_eff * scenario.dt_hours * 60)
                ent.q    = max(0.0, ent.q - admitted)
                total_admitted += admitted
            entrance_surge = any(surge_map.values())
            total_ever_admitted += total_admitted

            # Disperse gate admissions to festival terrain zones
            for n in festival_zone_names:
                states[n].a += total_admitted * festival_arrival_share[n]

            # ── departures from festival terrain ───────────────────
            festival_a_now = sum(states[n].a for n in festival_zone_names)
            departures_this_step = festival_a_now * departure_frac_by_day[day][s]
            if festival_a_now > 0:
                for n in festival_zone_names:
                    zone_share = states[n].a / festival_a_now
                    states[n].a = max(0.0, states[n].a - departures_this_step * zone_share)

            # ── vendor queues ──────────────────────────────────────
            for st in states.values():
                demand = rng.poisson(RHO_V * vendor_mult_day[s] * st.a)
                served = st.v_z_effective * SIGMA_V * scenario.dt_hours * 60
                st.q_vendor = max(0.0, st.q_vendor + demand - served)
                if st.q_vendor > Q_MAX_VENDOR * st.v_z_effective:
                    st.extra_stalls += 1

            # ── inter-zone movement ────────────────────────────────
            moves = {n: 0.0 for n in festival_zone_names}
            for n in festival_zone_names:
                st = states[n]
                for nbr in st.spec.adjacent:
                    if nbr not in festival_zone_names:
                        continue
                    grad = st.density - states[nbr].density
                    if grad > 0:
                        width = BOUNDARY_WIDTH_M.get(frozenset({n, nbr}), 0.0)
                        flow  = scenario.kappa_m * grad * width * 50.0
                        moves[n]   -= flow
                        moves[nbr] += flow

            # Schedule-driven act-changeover pulses (step_within_day matches)
            CHANGEOVER_FRACTION = 0.25
            for (pulse_step, from_zone, to_zone) in scenario.act_schedule:
                if s == pulse_step and from_zone in festival_zone_names \
                        and to_zone in festival_zone_names:
                    pulse = states[from_zone].a * CHANGEOVER_FRACTION
                    moves[from_zone] -= pulse
                    moves[to_zone]   += pulse

            for n in festival_zone_names:
                states[n].a = max(0.0, states[n].a + moves[n])

            # ── incidents ─────────────────────────────────────────
            for st in states.values():
                inc = sample_incidents(st.a, weather, scenario.dt_hours, rng)
                for k, v in inc.items():
                    st.incidents_cum[k] += v

            # ── density tiers + violations: festival terrain only ──
            fest_a_tot = sum(states[n].a for n in festival_zone_names)
            total_a    = sum(st.a for st in states.values())

            # Violation check uses THETA_WARN / THETA_VIOLATION (zone-average thresholds),
            # not D_MAX_BY_WEATHER — D_max is only for permit capacity, not operations.
            # During a scheduled performance window the hard limit relaxes to D_MAX_PERFORMANCE.
            in_performance = bool(performance_steps) and (s in performance_steps)
            d_hard = (D_MAX_PERFORMANCE[weather] if in_performance else THETA_VIOLATION[weather])

            for n in festival_zone_names:
                dens = states[n].density
                if dens > d_hard:
                    V += 1          # hard violation — drives evacuation
                    break           # one V per step regardless of how many zones breach
                elif dens > THETA_WARN[weather]:
                    W_density += 1  # soft warning — logged only

            capacity_ok = fest_a_tot <= amax_now
            if not capacity_ok:
                V += 1

            # Noise complaints — sampled hourly (every 4 within-day steps)
            if s % 4 == 0:
                chi_nu = 0.8 if weather == "rain" else 1.0
                nu += rng.poisson(0.3 * (total_a / max(1, scenario.a_total)) * chi_nu)

            if V > scenario.v_max_violation:
                evacuated = True

        # ── log ───────────────────────────────────────────────────
        total_a_now  = sum(st.a for st in states.values())
        festival_a   = sum(states[n].a for n in festival_zone_names)
        peak_occupancy = max(peak_occupancy, festival_a)
        for n, st in states.items():
            zone_rows.append({
                "t": t, "day": day, "zone": n,
                "a_z": st.a, "density": st.density,
                "q_vendor": st.q_vendor, "extra_stalls": st.extra_stalls,
                "minor":    st.incidents_cum["minor"],
                "moderate": st.incidents_cum["moderate"],
                "critical": st.incidents_cum["critical"],
            })
        fest_rows.append({
            "t": t, "day": day, "weather": weather, "V": V, "nu": nu,
            "W_density": W_density,
            "A_max": amax_now,
            "total_a":             total_a_now,
            "festival_a":          festival_a,
            "entrance_q":          sum(e.q for e in entrances.values()),
            "entrance_surge":      entrance_surge,
            "evacuated":           evacuated,
            "total_ever_admitted": total_ever_admitted,
            "peak_occupancy":      peak_occupancy,
        })
        next_t = t + 1
        markov_next = step_weather(weather, rng)
        if (next_t < total_steps
                and forecast_state[next_t] is not None
                and rng.random() < scenario.forecast_confidence):
            weather = forecast_state[next_t]
        else:
            weather = markov_next

    return pd.DataFrame(fest_rows), pd.DataFrame(zone_rows)


# ============================================================
# AGGREGATION INTO UTILITIES  (Section 24)
# ============================================================

def aggregate_run(
    timeline: pd.DataFrame,
    zone_timeline: pd.DataFrame,
    scenario: FestivalScenario,
    zones: Dict[str, ZoneSpec] = DEFAULT_ZONES,
    cost_params: CostParams = DEFAULT_COST_PARAMS,
    total_lanes: int = 0,
) -> Dict:
    p  = scenario.ticket_price
    T_evac = scenario.t_evac_min

    # ── Revenue R: based on total people who attended (not end-of-night occupancy) ──
    R = p * timeline["total_ever_admitted"].iloc[-1] / 1000.0

    # ── Cost C: dynamic from actual zone usage, events, and weather ─────────────
    C = compute_cost(timeline, zone_timeline, scenario, zones, cost_params, total_lanes)

    # ── Peak density D ──────────────────────────────────────────
    D = zone_timeline["density"].max()

    # ── Queue pressure Q: peak vendor queue per stall (festival zones only) ─
    # Exogenous zones (camping) excluded: campers self-cater
    festival_zones = [n for n, s in zones.items() if not s.is_exogenous]
    zone_tl_festival = zone_timeline[zone_timeline["zone"].isin(festival_zones)]
    Q = (zone_tl_festival["q_vendor"] /
         zone_tl_festival.apply(
             lambda r: zones[r["zone"]].v_z + r["extra_stalls"], axis=1)
         ).max()

    # ── Severity penalty Phi  (FIX 5: per-zone squaring) ────────
    #    Document: Phi = sum_z [omega1*r_minor_z + omega2*r_mod_z + omega3*(r_crit_z)^2]
    minor_z    = zone_timeline.groupby("zone")["minor"].max()
    moderate_z = zone_timeline.groupby("zone")["moderate"].max()
    critical_z = zone_timeline.groupby("zone")["critical"].max()
    phi = (OMEGA["minor"]    * minor_z.sum()
         + OMEGA["moderate"] * moderate_z.sum()
         + OMEGA["critical"] * (critical_z ** 2).sum())  # per-zone squaring

    V_f       = timeline["V"].iloc[-1]
    nu_f      = timeline["nu"].iloc[-1]
    W_dens_f  = timeline["W_density"].iloc[-1]
    total_ever   = timeline["total_ever_admitted"].iloc[-1]
    peak_occ     = timeline["peak_occupancy"].iloc[-1]

    # ── Infrastructure strain I (proxy: peak occupancy / max capacity) ─
    I = peak_occ / max(1, scenario.a_total)

    # ── u_O  (Section 13.4 / 6.1) ──────────────────────────────
    # u_O = 3R - 2C - gamma_O*D - delta_O*Q - Phi
    u_O = 3*R - 2*C - 1*D - 1*Q - phi

    # ── u_G  (Section 6.2) ──────────────────────────────────────
    # u_G = 2*E_econ - N - 3*V - 2*T_evac - I
    u_G = 2*R - 1*nu_f - 3*V_f - 2*T_evac - 1*I

    # ── u_A  (Section 6.3) ─────────────────────────────────────
    # J: enjoyment stays near-baseline until density exceeds warning threshold
    # (0.85 * D_max). Above that, enjoyment degrades quickly.
    # Research: crowd comfort degrades sharply above ~1.7 p/m² standing,
    # not linearly from zero.
    d_max_ref = 2.0   # clear-weather ceiling
    theta_warn = 0.85 * d_max_ref  # = 1.70
    comfort = max(0.0, 1.0 - max(0.0, D - theta_warn) / (d_max_ref - theta_warn))
    J = U_A_BASE * comfort * max(0.0, 1.0 - 0.15 * V_f)

    # W: entrance wait normalised (W=0 if queue is negligible)
    W = timeline["entrance_q"].mean() / max(1.0, peak_occ / 10.0)

    # T_evac: attendee penalty normalised to 0–1 (8 min → 0.8, 10 min → 1.0)
    T_evac_norm = T_evac / 10.0

    # u_A = alpha*J - lambda*W - mu*D - nu*T_evac_norm - rho*I
    u_A = ALPHA_A * J - LAMBDA_A * W - MU_A * D - NU_A * T_evac_norm - RHO_A * I
    attends = u_A >= U_A_MIN

    feasible = V_f == 0 and not timeline["evacuated"].any()
    surge_activations = int(timeline["entrance_surge"].sum())

    return {
        "scenario":           scenario.name,
        "u_O":                u_O,
        "u_G":                u_G,
        "u_A":                u_A,
        "attends":            attends,
        "D":                  D,
        "Q":                  Q,
        "V":                  V_f,
        "W_density":          W_dens_f,
        "nu":                 nu_f,
        "Phi":                phi,
        "I":                  I,
        "minor":              minor_z.sum(),
        "moderate":           moderate_z.sum(),
        "critical":           critical_z.sum(),
        "peak_occupancy":     peak_occ,
        "total_ever_admitted":total_ever,
        "feasible":           feasible,
        "surge_activations":  surge_activations,
    }


def monte_carlo(scenario: FestivalScenario, n_runs: int = 100,
                seed: int = None,
                zones: Dict[str, ZoneSpec] = DEFAULT_ZONES,
                cost_params: CostParams = DEFAULT_COST_PARAMS,
                total_lanes: int = 0,
                entrance_lanes: Optional[Dict[str, int]] = None) -> pd.DataFrame:
    base_seed = seed if seed is not None else (scenario.seed if scenario.seed is not None else 2026)
    rows = []
    for run in range(n_runs):
        tl, ztl = run_festival_once(scenario, zones=zones, seed=base_seed + 17*run,
                                    entrance_lanes=entrance_lanes)
        rows.append(aggregate_run(tl, ztl, scenario, zones=zones,
                                  cost_params=cost_params, total_lanes=total_lanes))
    return pd.DataFrame(rows)


def summarize_alternative(df: pd.DataFrame) -> Dict:
    return {
        "E_uO":              df["u_O"].mean(),
        "E_uG":              df["u_G"].mean(),
        "E_uA":              df["u_A"].mean(),
        "frac_attends":      df["attends"].mean(),
        "minimax_uO":        df["u_O"].min(),
        "frac_feasible":     df["feasible"].mean(),
        "mean_D":            df["D"].mean(),
        "mean_Q":            df["Q"].mean(),
        "mean_critical":     df["critical"].mean(),
        "mean_peak_occ":     df["peak_occupancy"].mean(),
        "mean_ever_admitted":df["total_ever_admitted"].mean(),
        "mean_W_density":    df["W_density"].mean(),
    }


# ============================================================
# VISUALISATION
# ============================================================

_HOURS = np.arange(512) * 0.25  # supports up to 8-day festivals (8 × 64 steps)

def _time_label(h: float) -> str:
    total_min = int(h * 60)
    return f"{10 + total_min // 60:02d}:{total_min % 60:02d}"

def _hours_ticks(ax: plt.Axes, every: int = 2) -> None:
    ticks = list(range(0, 17, every))
    ax.set_xticks(ticks)
    ax.set_xticklabels([_time_label(h) for h in ticks])

def _shade_weather(ax: plt.Axes, timeline: pd.DataFrame, hours: np.ndarray) -> None:
    prev_w, prev_h = timeline["weather"].iloc[0], hours[0]
    for i, (h, w) in enumerate(zip(hours, timeline["weather"])):
        if w != prev_w or i == len(hours) - 1:
            ax.axvspan(prev_h, h, color=WEATHER_COLOURS[prev_w], alpha=0.07, lw=0)
            prev_w, prev_h = w, h


def plot_day_overview(timeline: pd.DataFrame, zone_tl: pd.DataFrame,
                      scenario: FestivalScenario) -> plt.Figure:
    hours = _HOURS[:len(timeline)]
    fig = plt.figure(figsize=(13, 14))
    fig.suptitle(
        f"Festival Day Overview  ·  {scenario.name} ({scenario.a_total:,} attendees)", y=0.98)
    gs = gridspec.GridSpec(4, 1, figure=fig, hspace=0.45, top=0.94, bottom=0.06)

    ax = fig.add_subplot(gs[0])
    _shade_weather(ax, timeline, hours)
    ax.fill_between(hours, timeline["total_a"], alpha=0.18, color=ACCENT)
    ax.plot(hours, timeline["total_a"], color=ACCENT, lw=2.2, label="Total occupancy")
    ax.plot(hours, timeline["A_max"],   color=DANGER, lw=1.5, ls="--",
            label="$A_{max}$(weather)")
    over = timeline["total_a"] > timeline["A_max"]
    ax.fill_between(hours, timeline["total_a"], timeline["A_max"],
                    where=over, color=DANGER, alpha=0.35, label="Over capacity")
    ax.set_title("Festival Occupancy vs. Permitted Maximum")
    ax.set_ylabel("People")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.legend(loc="upper left")
    _hours_ticks(ax)

    ax2 = fig.add_subplot(gs[1])
    _shade_weather(ax2, timeline, hours)
    ax2.fill_between(hours, timeline["entrance_q"], alpha=0.18, color=WARNING)
    ax2.plot(hours, timeline["entrance_q"], color=WARNING, lw=2.2,
             label="Entrance queue (both)")
    surge = timeline["entrance_surge"].astype(bool)
    ax2.fill_between(hours, 0, timeline["entrance_q"],
                     where=surge, color=WARNING, alpha=0.4, label="Surge active")
    ax2.set_title("Entrance Queue — Main (35) + Secondary (22) lanes")
    ax2.set_ylabel("People queuing")
    ax2.legend(loc="upper left")
    _hours_ticks(ax2)

    ax3 = fig.add_subplot(gs[2])
    _shade_weather(ax3, timeline, hours)
    ax3.step(hours, timeline["V"], color=DANGER, lw=2.2, where="post", label="$V(t)$")
    ax3.axhline(scenario.v_max_violation, color=DANGER, ls="--", lw=1.2, alpha=0.6,
                label=f"$V_{{max}}$ = {scenario.v_max_violation}")
    evac_t = timeline[timeline["evacuated"]]["t"]
    if not evac_t.empty:
        ax3.axvline(hours[evac_t.iloc[0]], color=DANGER, lw=1.5, alpha=0.5, ls=":",
                    label="Evacuation triggered")
    ax3.set_title("Cumulative Safety Violations $V(t)$")
    ax3.set_ylabel("Violations")
    ax3.set_ylim(bottom=0)
    ax3.legend(loc="upper left")
    _hours_ticks(ax3)

    ax4 = fig.add_subplot(gs[3])
    weather_num = [WEATHER_STATES.index(w) for w in timeline["weather"]]
    for i, (w, c) in enumerate(WEATHER_COLOURS.items()):
        ax4.fill_between(hours, i - 0.45, i + 0.45,
                         where=[wn == i for wn in weather_num],
                         color=c, alpha=0.55, step="post")
    ax4.step(hours, weather_num, color="#E8EAF0", lw=1.5, where="post")
    ax4.set_yticks([0, 1, 2])
    ax4.set_yticklabels(["Clear", "Rain", "Heat"])
    ax4.set_title("Weather State $w(t)$ — Markov Chain")
    ax4.set_ylabel("Weather")
    ax4.set_xlabel("Time of day")
    _hours_ticks(ax4)
    return fig


def plot_zone_density(zone_tl: pd.DataFrame, timeline: pd.DataFrame = None) -> plt.Figure:
    zone_names = sorted(zone_tl["zone"].unique())
    n = len(zone_names)
    ncols = min(n, 2)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 4 * nrows), sharex=True, sharey=True)
    axes_flat = np.array(axes).flat if n > 1 else [axes]
    fig.suptitle("Zone Crowd Density over Festival Day", y=0.98)
    hours = _HOURS[:zone_tl["t"].max() + 1]

    for idx, (ax, zname) in enumerate(zip(axes_flat, zone_names)):
        colour = _zone_colour(zname, idx)
        zdata   = zone_tl[zone_tl["zone"] == zname]
        density = zdata["density"].values
        ax.fill_between(hours, density, alpha=0.2, color=colour)
        ax.plot(hours, density, color=colour, lw=2.2)
        for w, dmax in D_MAX_BY_WEATHER.items():
            ax.axhline(dmax, color=WEATHER_COLOURS[w], ls="--", lw=1.1, alpha=0.9,
                       label=f"$D_{{max}}$ {w} = {dmax}")
        ax.fill_between(hours, D_MAX_BY_WEATHER["heat"], 2.5,
                        alpha=0.07, color=DANGER)
        ax.set_title(zname.replace("_", " ").title())
        ax.set_ylabel("Density (p/m²)")
        ax.set_ylim(0, 2.5)
        ax.legend(fontsize=7, loc="upper left")
        _hours_ticks(ax)

    for ax in axes[1]:
        ax.set_xlabel("Time of day")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def plot_vendor_queues(zone_tl: pd.DataFrame,
                       zones: Dict[str, ZoneSpec] = DEFAULT_ZONES) -> plt.Figure:
    zone_names = sorted(zone_tl["zone"].unique())
    n = len(zone_names)
    ncols = min(n, 2)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 4 * nrows), sharex=True)
    axes_flat = np.array(axes).flat if n > 1 else [axes]
    fig.suptitle("Vendor Queue per Stall by Zone", y=0.98)
    hours = _HOURS[:zone_tl["t"].max() + 1]

    for idx, (ax, zname) in enumerate(zip(axes_flat, zone_names)):
        colour = _zone_colour(zname, idx)
        zdata = zone_tl[zone_tl["zone"] == zname]
        v_z   = zones[zname].v_z if zname in zones else 1
        q_per_stall = zdata["q_vendor"].values / (
            (v_z + zdata["extra_stalls"].values).clip(min=1))
        ax.fill_between(hours, q_per_stall, alpha=0.18, color=colour)
        ax.plot(hours, q_per_stall, colour, lw=2.2)
        deployed = zdata["extra_stalls"].values > 0
        ax.fill_between(hours, 0, q_per_stall, where=deployed,
                        color=SUCCESS, alpha=0.15, label="Extra stalls deployed")
        ax.axhline(Q_MAX_VENDOR, color=DANGER, ls="--", lw=1.3,
                   label=f"$Q_{{max}}^{{vendor}}$ = {Q_MAX_VENDOR:.1f}")
        ax.axhline(Q_MAX_VENDOR * 0.5, color=WARNING, ls=":", lw=1.0,
                   label="50% threshold")
        ax.set_title(zname.replace("_", " ").title())
        ax.set_ylabel("Queue / stall (people)")
        ax.legend(fontsize=7, loc="upper left")
        _hours_ticks(ax)

    for ax in axes[1]:
        ax.set_xlabel("Time of day")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def plot_incidents(zone_tl: pd.DataFrame) -> plt.Figure:
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Incident Analysis over Festival Day", y=0.98)
    hours = _HOURS[:zone_tl["t"].max() + 1]

    minor = zone_tl.groupby("t")["minor"].sum().values
    mod   = zone_tl.groupby("t")["moderate"].sum().values
    crit  = zone_tl.groupby("t")["critical"].sum().values

    ax.stackplot(hours, minor, mod, crit,
                 labels=["Minor", "Moderate", "Critical"],
                 colors=["#66BB6A", "#FFA726", "#EF5350"], alpha=0.85)
    ax.set_title("Cumulative Incidents (stacked)")
    ax.set_xlabel("Time of day")
    ax.set_ylabel("Cumulative count")
    ax.legend(loc="upper left")
    _hours_ticks(ax)

    zone_names_inc = sorted(zone_tl["zone"].unique())
    final      = zone_tl[zone_tl["t"] == zone_tl["t"].max()]
    x          = np.arange(len(zone_names_inc))
    w          = 0.25
    bars_minor = [final[final["zone"] == z]["minor"].values[0]    for z in zone_names_inc]
    bars_mod   = [final[final["zone"] == z]["moderate"].values[0] for z in zone_names_inc]
    bars_crit  = [final[final["zone"] == z]["critical"].values[0] for z in zone_names_inc]
    ax2.bar(x - w, bars_minor, w, label="Minor",    color="#66BB6A", alpha=0.85)
    ax2.bar(x,     bars_mod,   w, label="Moderate", color="#FFA726", alpha=0.85)
    ax2.bar(x + w, bars_crit,  w, label="Critical", color="#EF5350", alpha=0.85)
    ax2.set_xticks(x)
    ax2.set_xticklabels([z.replace("_", " ").title() for z in zone_names_inc], fontsize=8)
    ax2.set_title("End-of-Day Incidents by Zone")
    ax2.set_ylabel("Count")
    ax2.legend()
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def plot_policy_comparison(summaries: Dict, results: Dict) -> plt.Figure:
    labels  = list(summaries.keys())
    x       = np.arange(len(labels))
    width   = 0.55
    colours = [ALT_COLOURS.get(l, "#888") for l in labels]

    fig, axes = plt.subplots(1, 4, figsize=(18, 6))
    fig.suptitle("A1–A4 Policy Comparison  ·  50 replications each", y=0.98)

    # 1. E[u_O] + minimax
    ax = axes[0]
    e_uo = [summaries[l]["E_uO"]       for l in labels]
    mini = [summaries[l]["minimax_uO"] for l in labels]
    bars = ax.bar(x, e_uo, width, color=colours, alpha=0.85, zorder=3,
                  edgecolor="#0F1117", lw=0.5)
    for bar, val in zip(bars, e_uo):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 300,
                f"{int(val):,}", ha="center", va="bottom", fontsize=8, color="#C8CDD8")
    yerr_lo = [max(0, e - m) for e, m in zip(e_uo, mini)]
    ax.errorbar(x, e_uo, yerr=[yerr_lo, [0]*len(labels)],
                fmt="none", color="#E8EAF0", capsize=7, lw=1.8, zorder=4)
    for i, (xi, m) in enumerate(zip(x, mini)):
        if m < 0:
            ax.text(xi, m - 2000, f"{int(m):,}", ha="center", va="top",
                    fontsize=7, color=DANGER)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_title("$\\mathbb{E}[u_O]$ + minimax")
    ax.set_ylabel("Organiser utility")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.axhline(0, color="#444", lw=0.8)

    # 2. E[u_A] — new panel
    ax2 = axes[1]
    e_ua = [summaries[l]["E_uA"] for l in labels]
    bars2 = ax2.bar(x, e_ua, width, color=colours, alpha=0.85, zorder=3,
                    edgecolor="#0F1117", lw=0.5)
    for bar, val in zip(bars2, e_ua):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                 f"{val:.1f}", ha="center", va="bottom", fontsize=8, color="#C8CDD8")
    ax2.axhline(U_A_MIN, color=DANGER, ls="--", lw=1.2,
                label=f"$u_A^{{min}}$ = {U_A_MIN:.1f}")
    ax2.set_xticks(x); ax2.set_xticklabels(labels)
    ax2.set_title("$\\mathbb{E}[u_A]$ (Attendee)")
    ax2.set_ylabel("Attendee utility")
    ax2.legend(fontsize=8)

    # 3. Fraction feasible
    ax3 = axes[2]
    feas = [summaries[l]["frac_feasible"] * 100 for l in labels]
    bars3 = ax3.bar(x, feas, width, color=colours, alpha=0.85, zorder=3,
                    edgecolor="#0F1117", lw=0.5)
    for bar, val in zip(bars3, feas):
        col = SUCCESS if val >= 90 else (WARNING if val >= 60 else DANGER)
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f"{val:.0f}%", ha="center", va="bottom", fontsize=9,
                 color=col, fontweight="bold")
    ax3.axhline(100, color="#444", ls="--", lw=0.8)
    ax3.axhspan(0,  60, color=DANGER,  alpha=0.07)
    ax3.axhspan(60, 90, color=WARNING, alpha=0.07)
    ax3.axhspan(90, 110, color=SUCCESS, alpha=0.07)
    ax3.set_xticks(x); ax3.set_xticklabels(labels)
    ax3.set_title("Fraction Feasible")
    ax3.set_ylabel("% runs: V=0, no evacuation")
    ax3.set_ylim(0, 112)

    # 4. Mean peak density
    ax4 = axes[3]
    dens = [summaries[l]["mean_D"] for l in labels]
    bars4 = ax4.bar(x, dens, width, color=colours, alpha=0.85, zorder=3,
                    edgecolor="#0F1117", lw=0.5)
    for bar, val in zip(bars4, dens):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=8, color="#C8CDD8")
    for w, c in WEATHER_COLOURS.items():
        ax4.axhline(D_MAX_BY_WEATHER[w], color=c, ls="--", lw=1.3,
                    label=f"$D_{{max}}$ {w}")
    ax4.set_xticks(x); ax4.set_xticklabels(labels)
    ax4.set_title("Mean Peak Density (p/m²)")
    ax4.set_ylabel("Density (p/m²)")
    ax4.legend(fontsize=8, loc="upper left")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def print_summary_table(summaries: Dict, results: Dict) -> None:
    rows = []
    for label, s in summaries.items():
        df = results[label]
        rows.append({
            "Alt.":            label,
            "E[u_O]":          f"{s['E_uO']:,.0f}",
            "Minimax u_O":     f"{s['minimax_uO']:,.0f}",
            "E[u_G]":          f"{s['E_uG']:,.0f}",
            "E[u_A]":          f"{s['E_uA']:.1f}",
            "Attends (%)":     f"{s['frac_attends']*100:.0f}%",
            "Feasible (%)":    f"{s['frac_feasible']*100:.0f}%",
            "Mean D (p/m²)":   f"{s['mean_D']:.3f}",
            "Density warns":   f"{s['mean_W_density']:.0f}",
            "Mean Q (q/stall)":f"{s['mean_Q']:.2f}",
            "Mean critical":   f"{s['mean_critical']:.2f}",
            "Peak occ.":       f"{s['mean_peak_occ']:,.0f}",
            "Ever admitted":   f"{s['mean_ever_admitted']:,.0f}",
        })
    tbl = pd.DataFrame(rows)
    print("\n" + "=" * 100)
    print("POLICY COMPARISON SUMMARY TABLE")
    print("=" * 100)
    print(tbl.to_string(index=False))
    print("=" * 100 + "\n")


# ============================================================
# PLANNING LAYER
# ============================================================

@dataclass
class FestivalPlan:
    name: str
    zone_areas: Dict[str, float]       # {zone_name: usable_area_m2}
    entrance_lanes: Dict[str, int]     # {"main": 20, "secondary": 10}
    ticket_sales: int
    ticket_price: float
    n_days: int
    t_evac_min: int
    exit_widths: Dict[str, float] = field(default_factory=dict)   # auto-estimated if empty
    is_camping: bool = False
    camping_fraction: float = 0.0
    sold_out_fraction: float = 1.0
    has_official_resale: bool = False
    multiday_ticket_fraction: float = 1.0
    total_budget: Optional[float] = None
    n_stages: int = 1
    staggered_end_times: bool = False
    gate_open_hour: int = 10
    headliner_start_hour: int = 19
    n_runs: int = 100
    seed: int = 2026


def _estimate_exit_widths(
    zone_areas: Dict[str, float],
    ticket_sales: int,
    t_evac_min: int,
) -> Dict[str, float]:
    """SGSA standard: total width = ticket_sales / (70 p/m × T_evac), split by zone area."""
    total_width = ticket_sales / (70 * t_evac_min)
    total_area  = sum(zone_areas.values())
    return {name: round(total_width * area / total_area, 1) for name, area in zone_areas.items()}


@dataclass
class PlanReport:
    plan: FestivalPlan
    resources: Dict[str, Dict]
    baseline_cost: float
    a_max_by_weather: Dict[str, float]
    mc_results: pd.DataFrame
    summary: Dict
    noshow_rates: Dict[str, float]
    total_lanes: int = 0
    gate_cost: float = 0.0
    cost_params: CostParams = None

    def print_summary(self) -> None:
        p, s = self.plan, self.summary
        print(f"\n{'='*60}")
        print(f"  FESTIVAL PLANNING REPORT: {p.name}")
        print(f"{'='*60}")
        print(f"  Tickets sold:  {p.ticket_sales:,}")
        print(f"  Ticket price:  €{p.ticket_price:.2f}")
        print(f"  Days:          {p.n_days}    T_evac: {p.t_evac_min} min")
        print(f"  Camping:       {'Yes' if p.is_camping else 'No'} ({p.camping_fraction:.0%} fraction)")
        print(f"  Stages:        {p.n_stages}  Staggered ends: {p.staggered_end_times}")

        print(f"\n--- FEASIBILITY BY WEATHER ---")
        for w, amax in self.a_max_by_weather.items():
            noshow  = self.noshow_rates[w]
            eff     = int(p.ticket_sales * (1 - noshow))
            status  = "✓ FEASIBLE" if eff <= amax else "✗ INFEASIBLE"
            print(f"  {w:<8}  A_max={amax:,.0f}  effective={eff:,}  {status}")

        print(f"\n--- MINIMUM RESOURCE REQUIREMENTS ---")
        print(f"  {'Zone':<20} {'Attend':>8} {'Staff':>6} {'Stalls':>7} {'Toilets':>8} {'First Aid':>10}")
        print(f"  {'-'*60}")
        totals: Dict[str, int] = {k: 0 for k in ("attendance","staff","vendor_stalls","toilets","first_aid")}
        for zname, r in self.resources.items():
            print(f"  {zname:<20} {r['attendance']:>8,} {r['staff']:>6} "
                  f"{r['vendor_stalls']:>7} {r['toilets']:>8} {r['first_aid']:>10}")
            for k in totals:
                totals[k] += r[k]
        print(f"  {'TOTAL':<20} {totals['attendance']:>8,} {totals['staff']:>6} "
              f"{totals['vendor_stalls']:>7} {totals['toilets']:>8} {totals['first_aid']:>10}")

        print(f"\n--- COST ESTIMATE ---")
        print(f"  Baseline (min resources, clear weather):  €{self.baseline_cost:,.0f}/day")
        print(f"  Rain scenario  (+15%):                    €{self.baseline_cost * 1.15:,.0f}/day")
        print(f"  Heat scenario  (+40%):                    €{self.baseline_cost * 1.40:,.0f}/day")
        if p.total_budget is not None:
            status = "✓ within budget" if self.baseline_cost <= p.total_budget else "✗ EXCEEDS BUDGET"
            print(f"  Budget provided:                          €{p.total_budget:,.0f}/day  →  {status}")

        print(f"\n--- MONTE CARLO RESULTS ({p.n_runs} runs) ---")
        print(f"  E[u_O]:         {s['E_uO']:,.0f}")
        print(f"  E[u_A]:         {s['E_uA']:.2f}")
        print(f"  Feasibility:    {s['frac_feasible']:.0%} of runs (V=0, no evacuation)")
        print(f"  Mean peak occ.: {s['mean_peak_occ']:,.0f}")
        print(f"  Mean density:   {s['mean_D']:.3f} p/m²")
        print(f"  Density warns:  {s['mean_W_density']:.0f} avg per run")

        print(f"\n--- BAYESIAN CAPACITY RULES ---")
        rain_cap = int(min(self.a_max_by_weather["rain"], p.ticket_sales))
        print(f"  Clear forecast (P(rain) < 0.20):  operate at full {p.ticket_sales:,}")
        print(f"  Rain forecast  (P(rain) > 0.85):  reduce to {rain_cap:,} (rain egress cap)")
        print(f"  Double rain signal (P > 0.95):    mandatory reduction or extend T_evac")
        print(f"{'='*60}\n")


def generate_plan(
    plan: FestivalPlan,
    cost_params: CostParams = None,
) -> PlanReport:
    if cost_params is None:
        cost_params = CostParams()

    # ── 1. Allocate attendance and compute minimum resources per zone ──
    S_MAX, V_MAX, R_MAX, F_MAX = 100, 250, 75, 5000
    total_area = sum(plan.zone_areas.values())
    resources: Dict[str, Dict] = {}
    for zname, area in plan.zone_areas.items():
        a_z = int(plan.ticket_sales * area / total_area)
        resources[zname] = {
            "attendance":    a_z,
            "staff":         math.ceil(a_z / S_MAX),
            "vendor_stalls": math.ceil(a_z / V_MAX),
            "toilets":       math.ceil(a_z / R_MAX),
            "first_aid":     math.ceil(a_z / F_MAX),
        }

    # ── 2. Estimate exit widths if not provided ────────────────────────
    exit_widths = plan.exit_widths if plan.exit_widths else _estimate_exit_widths(
        plan.zone_areas, plan.ticket_sales, plan.t_evac_min
    )

    # ── 3. Build ZoneSpec objects ──────────────────────────────────────
    zones: Dict[str, ZoneSpec] = {
        zname: ZoneSpec(
            name=zname,
            area_m2=area,
            n_gates=0,
            exit_width_m=exit_widths.get(zname, 0.0),
            arrival_share=area / total_area,
            v_z=resources[zname]["vendor_stalls"],
            is_exogenous=(zname == "camping" or resources[zname]["attendance"] == 0),
        )
        for zname, area in plan.zone_areas.items()
    }

    # ── 4. Build FestivalType and FestivalScenario ─────────────────────
    ft = FestivalType(
        n_days=plan.n_days,
        is_camping=plan.is_camping,
        camping_fraction=plan.camping_fraction,
        sold_out_fraction=plan.sold_out_fraction,
        has_official_resale=plan.has_official_resale,
        multiday_ticket_fraction=plan.multiday_ticket_fraction,
    )
    scenario = make_scenario(
        name=plan.name,
        a_total=plan.ticket_sales,
        t_evac_min=plan.t_evac_min,
        ticket_price=plan.ticket_price,
        festival_type=ft,
        n_stages=plan.n_stages,
        staggered_end_times=plan.staggered_end_times,
        gate_open_hour=plan.gate_open_hour,
        headliner_start_hour=plan.headliner_start_hour,
        seed=plan.seed,
    )

    # ── 5. Run Monte Carlo ─────────────────────────────────────────────
    total_lanes = sum(plan.entrance_lanes.values())
    mc_results = monte_carlo(scenario, n_runs=plan.n_runs, seed=plan.seed, zones=zones,
                             cost_params=cost_params, total_lanes=total_lanes)

    # ── 6. Baseline cost from minimum resources ────────────────────────
    gate_cost = total_lanes * (cost_params.omega_lane + cost_params.omega_gate_staff)
    baseline_cost = (
        sum(
            r["staff"]         * cost_params.omega_s
            + r["vendor_stalls"] * cost_params.omega_v
            + r["toilets"]       * cost_params.omega_t
            + r["first_aid"]     * cost_params.omega_f
            + cost_params.omega_z
            for r in resources.values()
        )
        + gate_cost
    )

    # ── 7. A_max per weather state ─────────────────────────────────────
    a_max_by_weather = {w: a_max(zones, w, plan.t_evac_min) for w in WEATHER_STATES}

    return PlanReport(
        plan=plan,
        resources=resources,
        baseline_cost=baseline_cost,
        a_max_by_weather=a_max_by_weather,
        mc_results=mc_results,
        summary=summarize_alternative(mc_results),
        noshow_rates=ft.noshow_rates(),
        total_lanes=total_lanes,
        gate_cost=gate_cost / 1000.0,
        cost_params=cost_params,
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    # ── Dynamic no-show rates across festival types ───────────
    print_noshow_table()

    # ── Validate profiles across festival types ───────────────
    print("Arrival profiles by festival type:")
    for label, n_st, cf in [("Multi-stage camping (Lowlands)", 3, 0.95),
                              ("Single-stage evening concert",  1, 0.0),
                              ("Multi-stage day-tripper",       3, 0.0)]:
        af = build_bimodal_arrival_fraction(n_stages=n_st, camping_fraction=cf,
                                            gate_open_hour=10, headliner_start_hour=23
                                            if n_st > 1 else 19)
        print(f"  {label}: first-wave {af[:20].sum()*100:.0f}%  "
              f"second-wave {af[20:].sum()*100:.0f}%")

    print("Departure profiles by festival type (last day):")
    for label, n_st, stag, cf in [
            ("Single headliner",          1, False, 0.0),
            ("Multi-stage simultaneous",  3, False, 0.0),
            ("Multi-stage staggered",     3, True,  0.0),
            ("Camping multi-day last day",3, True,  0.95)]:
        dep = build_departure_fraction(day=3, total_days=3, n_stages=n_st,
                                       staggered_end_times=stag, camping_fraction=cf)
        print(f"  {label}: last-3hrs {dep[48:].sum()*100:.0f}% of total")
    for d, lbl in [(1,"Fri"),(2,"Sat")]:
        dep = build_departure_fraction(day=d, total_days=3, camping_fraction=0.95)
        print(f"  Camping non-last [{lbl}]: {dep.sum()*100:.0f}% depart")

    # ── Single A4 full 3-day run ───────────────────────────────
    print("\nRunning single A4 full 3-day simulation...")
    scn_a4 = make_alternative("A4")
    timeline, zone_tl = run_festival_once(scn_a4, seed=2026)
    for d in [1, 2, 3]:
        tl_d = timeline[timeline["day"] == d]
        print(f"  Day {d}: peak_festival_a={tl_d['festival_a'].max():,.0f}"
              f"  ever_admitted_by_eod={tl_d['total_ever_admitted'].iloc[-1]:,.0f}")

    plot_day_overview(timeline, zone_tl, scn_a4).savefig(
        "plot1_day_overview.png", dpi=150, bbox_inches="tight")
    plot_zone_density(zone_tl, timeline).savefig(
        "plot2_zone_density.png", dpi=150, bbox_inches="tight")
    plot_vendor_queues(zone_tl).savefig(
        "plot3_vendor_queues.png", dpi=150, bbox_inches="tight")
    plot_incidents(zone_tl).savefig(
        "plot4_incidents.png", dpi=150, bbox_inches="tight")
    plt.close("all")
    print("  → plot1–4 saved")

    # ── 50-rep comparison A1–A4 (each rep = full 3-day run) ───
    print("\nRunning 50-replication comparison for A1–A4...")
    results, summaries = {}, {}
    for label in ["A1", "A2", "A3", "A4"]:
        scn = make_alternative(label)
        df  = monte_carlo(scn, n_runs=50, seed=2026)
        results[label]   = df
        summaries[label] = summarize_alternative(df)
        s = summaries[label]
        print(f"  {label}: E[uO]={s['E_uO']:,.0f}  E[uA]={s['E_uA']:.1f}"
              f"  feasible={s['frac_feasible']*100:.0f}%"
              f"  peak_occ={s['mean_peak_occ']:,.0f}"
              f"  ever_admitted={s['mean_ever_admitted']:,.0f}")

    plot_policy_comparison(summaries, results).savefig(
        "plot5_policy_comparison.png", dpi=150, bbox_inches="tight")
    plt.close("all")
    print("  → plot5 saved")
    print_summary_table(summaries, results)

    # ── Cross-festival-type comparison (same capacity, different type) ─
    print("Cross-festival type comparison (A3, 55k, 25-rep each)...")
    cross_results, cross_summaries = {}, {}
    for ft_key in ["camping_sellout", "camping_general", "singleday_sellout", "singleday_general"]:
        scn = make_scenario(ft_key, a_total=55000, t_evac_min=10,
                            festival_type_key=ft_key)
        df  = monte_carlo(scn, n_runs=25, seed=2026)
        cross_results[ft_key]   = df
        cross_summaries[ft_key] = summarize_alternative(df)
        nr = FESTIVAL_TYPE[ft_key].noshow_rates()
        s  = cross_summaries[ft_key]
        print(f"  {ft_key:<22}: noshow clear={nr['clear']:.1%} rain={nr['rain']:.1%}"
              f"  ever_admitted={s['mean_ever_admitted']:,.0f}"
              f"  E[uO]={s['E_uO']:,.0f}  feasible={s['frac_feasible']*100:.0f}%")

    # ── Planning layer example ─────────────────────────────────────────
    print("\nRunning planning layer example (MyFestival_2026)...")
    plan = FestivalPlan(
        name="MyFestival_2026",
        zone_areas={"main_stage": 20000, "food_village": 12000, "chill_zone": 8000},
        entrance_lanes={"main": 20, "secondary": 10},
        ticket_sales=35000,
        ticket_price=95.0,
        n_days=1,
        t_evac_min=10,
        is_camping=False,
        total_budget=300000,
        n_runs=50,
    )
    report = generate_plan(plan)
    report.print_summary()