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
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker

from or_des_toolkit_week6 import QueueScenario  # kept for compatibility


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
D_MAX_BY_WEATHER     = {"clear": 2.0, "rain": 1.7, "heat": 1.5}
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
    n_days:              int   = 1
    is_camping:          bool  = False
    camping_fraction:    float = 0.0
    sold_out_fraction:   float = 0.5
    has_official_resale: bool  = False

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
                                   dt_hours: float = 0.25,
                                   peak1_step: int = 5,    # ~11:15 (gate-open rush)
                                   peak2_step: int = 36,   # ~19:00 (pre-headliner)
                                   sigma1: float = 3.0,
                                   sigma2: float = 4.0,
                                   weight1: float = 0.60,  # 60% arrive in first wave
                                   weight2: float = 0.40,  # 40% in pre-headliner wave
                                   ) -> np.ndarray:
    """
    Bimodal arrival profile: gate-open rush + pre-headliner surge.
    Both components are Gaussian; the result is normalised so sum = 1.
    """
    steps = np.arange(n_steps, dtype=float)
    g1 = np.exp(-0.5 * ((steps - peak1_step) / sigma1) ** 2)
    g2 = np.exp(-0.5 * ((steps - peak2_step) / sigma2) ** 2)
    raw = weight1 * g1 + weight2 * g2
    raw = np.maximum(raw, 0.0)
    return raw / raw.sum()


def build_departure_fraction(n_steps: int = 64,
                             dt_hours: float = 0.25,
                             day: int = 2,
                             headliner_end_step: int = 52,  # ~23:00
                             sigma_dep: float = 3.5,
                             ) -> np.ndarray:
    """
    Day-dependent departure profile (Ticket Fairy / industry research):
      Day 1 (Friday) / Day 2 (Saturday): camping festival — the vast majority
        stay overnight. Only ~15% leave (mostly day-trippers), spread across
        the last few hours. No sharp exodus.
      Day 3 (Sunday, last day): full exodus — all campers AND day-visitors leave.
        Sharp post-headliner spike (75% of departures in last 3 hours) plus
        a broader tail as campers pack up. This is the critical egress scenario.
    """
    steps = np.arange(n_steps, dtype=float)

    if day in (1, 2):
        # Camping night: only ~15% of attendees leave (day-trippers).
        # Distributed across last 4 hours, no strong spike.
        trickle = np.zeros(n_steps)
        trickle[44:] = 1.0   # gradual from ~21:00 onward
        raw = trickle / trickle.sum()
        # Scale so only 15% of total admitted departs
        return raw * 0.15
    else:
        # Sunday last day: full exodus. Sharp headliner-end spike +
        # broader camper-departure tail (people leaving staggered as
        # they pack tents, 22:00-02:00).
        exodus = np.exp(-0.5 * ((steps - headliner_end_step) / sigma_dep) ** 2)
        # Camper tail: broader Gaussian centred 1h after headliner end
        camper_tail = np.exp(-0.5 * ((steps - (headliner_end_step + 4)) / 6.0) ** 2)
        raw = 0.55 * exodus / exodus.sum() + 0.45 * camper_tail / camper_tail.sum()
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
        adjacent=("cape_lowlands", "planet_paradise")),
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
# FESTIVAL SIMULATION
# ============================================================

@dataclass(frozen=True)
class FestivalScenario:
    name: str
    a_total: int          # intended total sold attendance
    t_evac_min: int       # T_evac: 8 (strict) or 10 (lenient)
    ticket_price: float = 365.0
    horizon_steps: int = 64      # 15-min steps over 16 hours
    dt_hours: float = 0.25
    kappa_m: float = 0.05        # inter-zone equilibration rate
    v_max_violation: int = 5
    day: int = 2   # 1=Friday, 2=Saturday, 3=Sunday (last day). Drives departure curve.
    festival_type: FestivalType = field(
        default_factory=lambda: FESTIVAL_TYPE["camping_sellout"])
    seed: Optional[int] = None

@dataclass(frozen=True)
class AlternativeA(FestivalScenario):
    """Convenience subclass for A1–A4."""
    pass

def make_alternative(label: str, day: int = 2,
                     festival_type: FestivalType = None) -> FestivalScenario:
    """
    Build one of the four Lowlands-style alternatives (A1–A4).

    day=1 Friday: most people stay overnight, small day-tripper exodus.
    day=2 Saturday: same as Friday, minimal overnight departures.
    day=3 Sunday: full exodus — campers + day visitors all leave.

    festival_type defaults to camping_sellout (Lowlands profile).
    Pass a different FestivalType to model other festival archetypes.
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
    return FestivalScenario(name=label, day=day, festival_type=ft, **table[label])


def make_scenario(name: str, a_total: int, t_evac_min: int,
                  festival_type_key: str = "camping_sellout",
                  festival_type: FestivalType = None,
                  day: int = 2, **kwargs) -> FestivalScenario:
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
                            day=day, festival_type=ft, **kwargs)

def egress_capacity(zones: Dict[str, ZoneSpec], weather: str, t_evac_min: int) -> float:
    return sum(z.exit_width_m for z in zones.values()) * PHI_BY_WEATHER[weather] * t_evac_min

def holding_capacity(zones: Dict[str, ZoneSpec], d_min: float = 0.5) -> float:
    return sum(z.area_m2 for z in zones.values()) / d_min

def a_max(zones: Dict[str, ZoneSpec], weather: str, t_evac_min: int) -> float:
    return min(holding_capacity(zones), egress_capacity(zones, weather, t_evac_min))


def run_festival_once(
    scenario: FestivalScenario,
    zones: Dict[str, ZoneSpec] = DEFAULT_ZONES,
    seed: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run one stochastic replication.
    Returns (timeline, zone_timeline).
    """
    rng = np.random.default_rng(seed if seed is not None else scenario.seed)

    states    = {n: ZoneState(s) for n, s in zones.items()}
    entrances = {n: EntranceState(n) for n in ENTRANCE_LANES}
    weather   = draw_initial_weather(rng)
    # No-show rate computed dynamically from festival type attributes
    noshow_rates = scenario.festival_type.noshow_rates()
    a_eff_total = scenario.a_total * (1 - noshow_rates[weather])

    V, nu, evacuated = 0, 0, False
    total_ever_admitted = 0.0  # cumulative admissions (not net of departures)
    peak_occupancy = 0.0       # max simultaneous occupancy across the day
    fest_rows, zone_rows = [], []

    # Bimodal arrival profile (gate-open rush + pre-headliner surge)
    arrival_fraction   = build_bimodal_arrival_fraction(scenario.horizon_steps, scenario.dt_hours)
    # Day-dependent departure profile
    departure_fraction = build_departure_fraction(scenario.horizon_steps, scenario.dt_hours,
                                                  day=scenario.day)

    # ── Camping population model ────────────────────────────────────────────
    # Camping is physically separated from festival terrain by a wristband
    # checkpoint. People do NOT receive arrivals into camping through the main
    # gate, and do NOT flow between camping and festival zones via density
    # gradient. Instead the camping population follows a daily schedule:
    #
    #   Morning (t=0–8, 10:00–12:00): campers wake and walk to festival terrain.
    #     ~95% of campers cross to festival by midday (sigmoid morning outflow).
    #   Afternoon/Evening (t=8–48): small residual in camping (resting, etc).
    #   Night (t=48–64, 22:00–02:00): people trickle back to camping after shows.
    #     On Sunday (last day): no evening return — full exodus instead.
    #
    # camping_fraction from FestivalType: for Lowlands camping_sellout = 0.95
    camping_fraction = scenario.festival_type.camping_fraction
    n_campers        = a_eff_total * camping_fraction
    n_daytrippers    = a_eff_total * (1.0 - camping_fraction)

    _ts = np.arange(scenario.horizon_steps, dtype=float)
    _morning_out = 1.0 / (1.0 + np.exp(-0.8 * (_ts - 6)))   # sigmoid: half by t=6 (11:30)
    _evening_in  = 1.0 / (1.0 + np.exp( 0.6 * (_ts - 50)))  # return from t=50 (22:30)
    camper_on_festival = np.clip(_morning_out - (1.0 - _evening_in) * 0.85, 0.05, 0.95)
    if scenario.day == 3:  # Sunday: no evening return, full exodus
        camper_on_festival = np.clip(_morning_out, 0.05, 0.95)

    # Festival terrain zones only (camping excluded from all arrival/flow/departure)
    festival_zone_names = [n for n in zones if n != "camping"]
    total_festival_share = sum(zones[n].arrival_share for n in festival_zone_names)
    festival_arrival_share = {
        n: zones[n].arrival_share / total_festival_share
        for n in festival_zone_names
    }

    # Pre-build camping population time series (exogenous, schedule-driven)
    camping_pop = np.zeros(scenario.horizon_steps)
    for t in range(scenario.horizon_steps):
        camping_pop[t] = n_campers * (1.0 - camper_on_festival[t])

    # Time-varying vendor demand multiplier: meal peaks at lunch (t≈12) and dinner (t≈28)
    # Base RHO_V is the average; peaks are ~2.5x the off-peak rate.
    # Steps: t=0 → 10:00, t=8 → 12:00 (lunch), t=28 → 17:00, t=32 → 18:00 (dinner)
    _steps = np.arange(scenario.horizon_steps, dtype=float)
    _lunch  = np.exp(-0.5 * ((_steps - 8)  / 2.5) ** 2)   # peak 12:00
    _dinner = np.exp(-0.5 * ((_steps - 32) / 3.0) ** 2)   # peak 18:00
    _snack  = np.exp(-0.5 * ((_steps - 48) / 4.0) ** 2)   # late snack ~22:00
    _meal_mult_raw = 0.3 + 1.0 * _lunch + 1.5 * _dinner + 0.8 * _snack
    # Normalise so mean = 1.0 (RHO_V stays calibrated on average)
    vendor_mult = _meal_mult_raw / _meal_mult_raw.mean()

    for t in range(scenario.horizon_steps):
        d_max_w  = D_MAX_BY_WEATHER[weather]
        amax_now = a_max(zones, weather, scenario.t_evac_min)
        entrance_surge = False

        if evacuated:
            for st in states.values():
                st.a = 0.0
        else:
            # ── camping: set exogenously from schedule ──────────────
            # Wristband checkpoint — not connected to gate arrivals or gradient flow
            states["camping"].a = camping_pop[t]

            # ── shared entrance queue (main + secondary) ───────────
            step_in = a_eff_total * arrival_fraction[t]
            total_admitted = 0.0
            surge_map = {}
            for en, ent in entrances.items():
                ent.q += step_in * ENTRANCE_SHARE[en]
                nl    = ENTRANCE_LANES[en]
                surge = ent.q > Q_MAX_GATE
                surge_map[en] = surge
                g_eff    = G_BASE_PER_MIN * nl * (1.6 if surge else 1.0)
                admitted = min(ent.q, g_eff * scenario.dt_hours * 60)
                ent.q    = max(0.0, ent.q - admitted)
                total_admitted += admitted
            entrance_surge = any(surge_map.values())
            total_ever_admitted += total_admitted

            # Dispersal to FESTIVAL TERRAIN ZONES ONLY (not camping)
            for n in festival_zone_names:
                states[n].a += total_admitted * festival_arrival_share[n]

            # ── departures from festival terrain only ───────────────
            festival_a_now = sum(states[n].a for n in festival_zone_names)
            departures_this_step = festival_a_now * departure_fraction[t]
            if festival_a_now > 0:
                for n in festival_zone_names:
                    zone_share = states[n].a / festival_a_now
                    states[n].a = max(0.0, states[n].a - departures_this_step * zone_share)

            # ── vendor queues (Section 21) ─────────────────────────
            # RHO_V * vendor_mult[t] gives time-varying demand:
            # peaks at lunch (~12:00), dinner (~18:00), late snack (~22:00)
            for st in states.values():
                demand = rng.poisson(RHO_V * vendor_mult[t] * st.a)
                served = st.v_z_effective * SIGMA_V * scenario.dt_hours * 60
                st.q_vendor = max(0.0, st.q_vendor + demand - served)
                if st.q_vendor > Q_MAX_VENDOR * st.v_z_effective:
                    st.extra_stalls += 1

            # ── inter-zone movement: festival terrain only ──────────
            # Camping excluded — wristband checkpoint prevents gradient flow
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
            for n in festival_zone_names:
                states[n].a = max(0.0, states[n].a + moves[n])

            # ── incidents (Section 17 / 22.9) ──────────────────────
            for st in states.values():
                inc = sample_incidents(st.a, weather, scenario.dt_hours, rng)
                for k, v in inc.items():
                    st.incidents_cum[k] += v

            # ── violations: festival terrain only ───────────────────
            # Camping excluded — separately managed, low density by design
            fest_a_tot = sum(states[n].a for n in festival_zone_names)
            total_a    = sum(st.a for st in states.values())
            density_ok = all(states[n].density <= d_max_w for n in festival_zone_names)
            capacity_ok = fest_a_tot <= amax_now
            if not density_ok or not capacity_ok:
                V += 1

            # noise complaints — sampled hourly (every 4 steps)
            if t % 4 == 0:
                chi_nu = 0.8 if weather == "rain" else 1.0
                nu += rng.poisson(0.3 * (total_a / 65000.0) * chi_nu)

            if V > scenario.v_max_violation:
                evacuated = True

        # ── log ────────────────────────────────────────────────────
        # total_a_now  = everyone on site (festival + camping)
        # festival_a   = festival terrain only (safety-relevant for density/capacity)
        # peak_occupancy tracks festival terrain peak only
        total_a_now  = sum(st.a for st in states.values())
        festival_a   = sum(states[n].a for n in festival_zone_names)
        peak_occupancy = max(peak_occupancy, festival_a)
        for n, st in states.items():
            zone_rows.append({
                "t": t, "zone": n,
                "a_z": st.a, "density": st.density,
                "q_vendor": st.q_vendor, "extra_stalls": st.extra_stalls,
                "minor":    st.incidents_cum["minor"],
                "moderate": st.incidents_cum["moderate"],
                "critical": st.incidents_cum["critical"],
            })
        fest_rows.append({
            "t": t, "weather": weather, "V": V, "nu": nu,
            "A_max": amax_now,
            "total_a":      total_a_now,    # everyone on site incl. camping
            "festival_a":   festival_a,     # festival terrain only (safety-relevant)
            "entrance_q":          sum(e.q for e in entrances.values()),
            "entrance_surge":      entrance_surge,
            "evacuated":           evacuated,
            "total_ever_admitted": total_ever_admitted,
            "peak_occupancy":      peak_occupancy,
        })
        weather = step_weather(weather, rng)

    return pd.DataFrame(fest_rows), pd.DataFrame(zone_rows)


# ============================================================
# AGGREGATION INTO UTILITIES  (Section 24)
# ============================================================

def aggregate_run(
    timeline: pd.DataFrame,
    zone_timeline: pd.DataFrame,
    scenario: FestivalScenario,
    omega_z_total: float = 391090.0,   # sum_z b_z baseline (Section 25, A4)
) -> Dict:
    p  = scenario.ticket_price
    T_evac = scenario.t_evac_min

    # ── Revenue R: based on total people who attended (not end-of-night occupancy) ──
    R = p * timeline["total_ever_admitted"].iloc[-1] / 1000.0

    # ── Cost C  (time-averaged, weather-multiplied) ─────────────
    C = (timeline["weather"].map(COST_MULT_BY_WEATHER) * (omega_z_total / 1000.0)).mean()

    # ── Peak density D ──────────────────────────────────────────
    D = zone_timeline["density"].max()

    # ── Queue pressure Q: peak vendor queue per stall (festival zones only) ─
    # Camping excluded: campers self-cater, camping stalls serve snacks only
    festival_zones = [z for z in DEFAULT_ZONES if z != "camping"]
    zone_tl_festival = zone_timeline[zone_timeline["zone"].isin(festival_zones)]
    Q = (zone_tl_festival["q_vendor"] /
         zone_tl_festival.apply(
             lambda r: DEFAULT_ZONES[r["zone"]].v_z + r["extra_stalls"], axis=1)
         ).max()

    # ── Severity penalty Phi  (FIX 5: per-zone squaring) ────────
    #    Document: Phi = sum_z [omega1*r_minor_z + omega2*r_mod_z + omega3*(r_crit_z)^2]
    minor_z    = zone_timeline.groupby("zone")["minor"].max()
    moderate_z = zone_timeline.groupby("zone")["moderate"].max()
    critical_z = zone_timeline.groupby("zone")["critical"].max()
    phi = (OMEGA["minor"]    * minor_z.sum()
         + OMEGA["moderate"] * moderate_z.sum()
         + OMEGA["critical"] * (critical_z ** 2).sum())  # per-zone squaring

    V_f  = timeline["V"].iloc[-1]
    nu_f = timeline["nu"].iloc[-1]
    total_ever   = timeline["total_ever_admitted"].iloc[-1]
    peak_occ     = timeline["peak_occupancy"].iloc[-1]

    # ── Infrastructure strain I (proxy: peak occupancy / max capacity) ─
    I = peak_occ / 65000.0

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

    return {
        "scenario":           scenario.name,
        "u_O":                u_O,
        "u_G":                u_G,
        "u_A":                u_A,
        "attends":            attends,
        "D":                  D,
        "Q":                  Q,
        "V":                  V_f,
        "nu":                 nu_f,
        "Phi":                phi,
        "I":                  I,
        "minor":              minor_z.sum(),
        "moderate":           moderate_z.sum(),
        "critical":           critical_z.sum(),
        "peak_occupancy":     peak_occ,
        "total_ever_admitted":total_ever,
        "feasible":           feasible,
    }


def monte_carlo(scenario: FestivalScenario, n_runs: int = 100,
                seed: int = 2026,
                zones: Dict[str, ZoneSpec] = DEFAULT_ZONES) -> pd.DataFrame:
    rows = []
    for run in range(n_runs):
        tl, ztl = run_festival_once(scenario, zones=zones, seed=seed + 17*run)
        rows.append(aggregate_run(tl, ztl, scenario))
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
    }


# ============================================================
# VISUALISATION
# ============================================================

_HOURS = np.arange(64) * 0.25

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
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True, sharey=True)
    fig.suptitle("Zone Crowd Density over Festival Day", y=0.98)
    hours = _HOURS[:zone_tl["t"].max() + 1]

    for ax, (zname, colour) in zip(axes.flat, ZONE_COLOURS.items()):
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


def plot_vendor_queues(zone_tl: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    fig.suptitle("Vendor Queue per Stall by Zone", y=0.98)
    hours = _HOURS[:zone_tl["t"].max() + 1]

    for ax, (zname, colour) in zip(axes.flat, ZONE_COLOURS.items()):
        zdata = zone_tl[zone_tl["zone"] == zname]
        v_z   = DEFAULT_ZONES[zname].v_z
        q_per_stall = zdata["q_vendor"].values / (
            (v_z + zdata["extra_stalls"].values).clip(min=1))
        ax.fill_between(hours, q_per_stall, alpha=0.18, color=colour)
        ax.plot(hours, q_per_stall, color=colour, lw=2.2)
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

    zones      = list(ZONE_COLOURS.keys())
    final      = zone_tl[zone_tl["t"] == zone_tl["t"].max()]
    x          = np.arange(len(zones))
    w          = 0.25
    bars_minor = [final[final["zone"] == z]["minor"].values[0]    for z in zones]
    bars_mod   = [final[final["zone"] == z]["moderate"].values[0] for z in zones]
    bars_crit  = [final[final["zone"] == z]["critical"].values[0] for z in zones]
    ax2.bar(x - w, bars_minor, w, label="Minor",    color="#66BB6A", alpha=0.85)
    ax2.bar(x,     bars_mod,   w, label="Moderate", color="#FFA726", alpha=0.85)
    ax2.bar(x + w, bars_crit,  w, label="Critical", color="#EF5350", alpha=0.85)
    ax2.set_xticks(x)
    ax2.set_xticklabels([z.replace("_", " ").title() for z in zones], fontsize=8)
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
# MAIN
# ============================================================

if __name__ == "__main__":
    # ── Dynamic no-show rates across festival types ───────────
    print_noshow_table()

    # ── Validate profiles ─────────────────────────────────────
    af = build_bimodal_arrival_fraction()
    print(f"Arrival profile: peak1 at t=5 (11.2h), peak2 at t=36 (19.0h)")
    print(f"  First wave (t<20, first 5hrs): {af[:20].sum()*100:.1f}% of arrivals")
    for d, label in [(1,"Fri"),(2,"Sat"),(3,"Sun")]:
        df_dep = build_departure_fraction(day=d)
        print(f"Departure [{label}]: departs {df_dep.sum()*100:.0f}% of admitted, "
              f"last-3hrs share={df_dep[48:].sum()/max(df_dep.sum(),1e-9)*100:.0f}%")

    # ── Single A4 Sunday run (worst-case day) ─────────────────
    print("\nRunning single A4 Sunday simulation (worst-case egress)...")
    scn_a4 = make_alternative("A4", day=3)
    timeline, zone_tl = run_festival_once(scn_a4, seed=2026)
    print(f"  Peak occupancy: {timeline['peak_occupancy'].max():,.0f}"
          f"  Ever admitted: {timeline['total_ever_admitted'].iloc[-1]:,.0f}")

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

    # ── 50-rep comparison A1–A4 across all three days ─────────
    print("\nRunning 50-replication comparison for A1–A4 across 3 days...")
    results, summaries = {}, {}
    for label in ["A1", "A2", "A3", "A4"]:
        # Weight days: Fri/Sat/Sun each contribute one day of the weekend
        day_dfs = []
        for day in [1, 2, 3]:
            scn = make_alternative(label, day=day)
            df  = monte_carlo(scn, n_runs=50, seed=2026 + day*1000)
            day_dfs.append(df)
        # Average across the three days (equal weight)
        combined = pd.concat(day_dfs, ignore_index=True)
        results[label]   = combined
        summaries[label] = summarize_alternative(combined)
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
                            festival_type_key=ft_key, day=2)
        df  = monte_carlo(scn, n_runs=25, seed=2026)
        cross_results[ft_key]   = df
        cross_summaries[ft_key] = summarize_alternative(df)
        nr = FESTIVAL_TYPE[ft_key].noshow_rates()
        s  = cross_summaries[ft_key]
        print(f"  {ft_key:<22}: noshow clear={nr['clear']:.1%} rain={nr['rain']:.1%}"
              f"  ever_admitted={s['mean_ever_admitted']:,.0f}"
              f"  E[uO]={s['E_uO']:,.0f}  feasible={s['frac_feasible']*100:.0f}%")