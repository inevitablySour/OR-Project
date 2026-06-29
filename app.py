"""
Festival Safety & Capacity Planning Tool
Panel + Material UI  |  Run: panel serve app.py --show --port 5006
"""
from __future__ import annotations

import itertools
import threading
from typing import Dict, List

import numpy as np
import panel as pn
import plotly.graph_objects as go

from main import (
    FestivalPlan,
    FestivalType,
    ZoneSpec,
    generate_plan,
    make_scenario,
    run_festival_once,
    THETA_WARN,
    THETA_VIOLATION,
    _estimate_exit_widths,
)

pn.extension("plotly", design="material", sizing_mode="stretch_width")

# ── Design tokens ─────────────────────────────────────────────────────────────
C_PRIMARY  = "#1A56DB"
C_ACCENT   = "#3B82F6"
C_BG       = "#F1F5F9"
C_SURFACE  = "#FFFFFF"
C_BORDER   = "#E2E8F0"
C_TEXT     = "#0F172A"
C_TEXT2    = "#64748B"
C_TEXT3    = "#94A3B8"
C_GREEN    = "#15803D"
C_YELLOW   = "#B45309"
C_RED      = "#B91C1C"
C_GREEN_L  = "#DCFCE7"
C_YELLOW_L = "#FEF9C3"
C_RED_L    = "#FEE2E2"
C_BLUE_L   = "#DBEAFE"

ZONE_PALETTE = ["#3B82F6", "#10B981", "#F59E0B", "#8B5CF6", "#EF4444", "#06B6D4"]
W_COL = {"clear": "#D97706", "rain": "#2563EB", "heat": "#DC2626"}


# ── Plotly base layout ────────────────────────────────────────────────────────
def base_layout(height=300):
    return dict(
        paper_bgcolor=C_SURFACE,
        plot_bgcolor="#F8FAFC",
        font=dict(color=C_TEXT2, size=11, family="Inter, sans-serif"),
        height=height,
        margin=dict(l=56, r=20, t=32, b=48),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10, color=C_TEXT2)),
        xaxis=dict(gridcolor=C_BORDER, zerolinecolor=C_BORDER, linecolor=C_BORDER,
                   tickfont=dict(color=C_TEXT2), title_font=dict(color=C_TEXT2, size=11)),
        yaxis=dict(gridcolor=C_BORDER, zerolinecolor=C_BORDER, linecolor=C_BORDER,
                   tickfont=dict(color=C_TEXT2), title_font=dict(color=C_TEXT2, size=11)),
    )


def empty_fig(msg="Run to see results", height=300):
    fig = go.Figure()
    fig.add_annotation(text=msg, x=0.5, y=0.5, xref="paper", yref="paper",
                       showarrow=False, font=dict(color=C_TEXT3, size=13))
    fig.update_layout(**base_layout(height))
    return fig


# ── Reusable HTML components ──────────────────────────────────────────────────
CARD_CSS = """
:host {
    background: white;
    border-radius: 10px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08), 0 0 0 1px rgba(0,0,0,0.05);
    padding: 20px 22px;
}
"""

SIDEBAR_CSS = """
:host {
    background: white;
    border-right: 1px solid #E2E8F0;
    padding: 0;
    height: 100vh;
    overflow-y: auto;
}
"""


def card(content, title: str = "", subtitle: str = ""):
    items = []
    if title:
        hdr = f'<div style="font-size:13px;font-weight:600;color:{C_TEXT};margin-bottom:{"2px" if subtitle else "14px"}">{title}</div>'
        if subtitle:
            hdr += f'<div style="font-size:11px;color:{C_TEXT3};margin-bottom:12px">{subtitle}</div>'
        items.append(pn.pane.HTML(hdr))
    items.append(content)
    return pn.Column(*items, stylesheets=[CARD_CSS], sizing_mode="stretch_width")


def metric_card(title: str, value: str, colour: str = C_TEXT, sub: str = ""):
    sub_html = f'<div style="font-size:11px;color:{C_TEXT3};margin-top:5px;line-height:1.3">{sub}</div>' if sub else ""
    return pn.pane.HTML(
        f'<div style="font-family:Inter,sans-serif;padding:2px 0">'
        f'<div style="font-size:10px;font-weight:600;color:{C_TEXT3};text-transform:uppercase;'
        f'letter-spacing:0.09em;margin-bottom:8px">{title}</div>'
        f'<div style="font-size:26px;font-weight:700;color:{colour};line-height:1;'
        f'font-variant-numeric:tabular-nums">{value}</div>'
        f'{sub_html}</div>',
        stylesheets=[CARD_CSS], min_height=95,
    )


def section_label(text: str):
    return pn.pane.HTML(
        f'<div style="font-size:10px;font-weight:700;color:{C_TEXT3};'
        f'text-transform:uppercase;letter-spacing:0.12em;'
        f'padding:18px 20px 8px;border-top:1px solid {C_BORDER};margin-top:4px">'
        f'{text}</div>'
    )


def info_badge(text: str, colour: str, bg: str):
    return pn.pane.HTML(
        f'<span style="display:inline-block;padding:3px 10px;border-radius:20px;'
        f'font-size:11px;font-weight:500;color:{colour};background:{bg};'
        f'margin:2px 3px 2px 0">{text}</span>'
    )


def _table_html(rows: list, headers: list) -> str:
    th_s = (f"font-size:10px;font-weight:600;color:{C_TEXT3};text-transform:uppercase;"
            f"letter-spacing:0.06em;padding:0 14px 10px 0;white-space:nowrap")
    td_s = f"font-size:12px;color:{C_TEXT};padding:8px 14px 8px 0;border-bottom:1px solid {C_BORDER}"
    head = "".join(f'<th style="{th_s}">{h}</th>' for h in headers)
    body = "".join(
        "<tr>" + "".join(f'<td style="{td_s}">{c}</td>' for c in row) + "</tr>"
        for row in rows
    )
    return (f'<table style="width:100%;border-collapse:collapse">'
            f'<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>')


# ── Sidebar widgets ───────────────────────────────────────────────────────────
_inp = dict(sizing_mode="stretch_width")

w_name      = pn.widgets.TextInput(name="Festival Name", value="MyFestival_2026", **_inp)
w_budget    = pn.widgets.FloatInput(name="Daily Budget (€)", value=300000.0, start=0, **_inp)
w_days      = pn.widgets.Select(name="Event Duration",
                                 options={f"{i} Day{'s' if i>1 else ''}": i for i in range(1, 9)},
                                 value=1, **_inp)
w_tevac     = pn.widgets.Select(name="Evacuation Time",
                                 options={"6 min": 6, "8 min": 8, "10 min": 10,
                                          "12 min": 12, "15 min": 15},
                                 value=10, **_inp)

# ── Ticket tier builder ────────────────────────────────────────────────────────
# Each tier: name, price, quantity, days_valid (how many festival days it covers)
_DEFAULT_TIERS = [
    {"name": "Weekend Pass",  "price": 195.0, "qty": 25000, "days": 2},
    {"name": "Day Ticket",    "price":  95.0, "qty": 10000, "days": 1},
]
_ticket_tiers: List[Dict] = list(_DEFAULT_TIERS)

_tier_name_inputs  = []
_tier_price_inputs = []
_tier_qty_inputs   = []
_tier_days_inputs  = []

ticket_tier_col    = pn.Column(margin=(0, 0, 0, 0))
ticket_summary_pane = pn.pane.HTML("", sizing_mode="stretch_width")

_tier_inp_name  = pn.widgets.TextInput(name="Ticket name", placeholder="e.g. Day 1 only", **_inp)
_tier_inp_price = pn.widgets.FloatInput(name="Price (€)", value=95.0, start=0, step=5, **_inp)
_tier_inp_qty   = pn.widgets.IntInput(name="Quantity", value=5000, start=1, step=100, **_inp)
_tier_inp_days  = pn.widgets.Select(name="Days covered",
                                     options={f"{i} day{'s' if i>1 else ''}": i for i in range(1, 9)},
                                     value=1)
btn_add_tier    = pn.widgets.Button(name="Add ticket type", button_type="primary",
                                     sizing_mode="stretch_width", height=34,
                                     stylesheets=["button{font-size:12px;border-radius:6px}"])


def _ticket_summary_html():
    if not _ticket_tiers:
        return f'<div style="font-size:11px;color:{C_TEXT3};font-style:italic">No ticket types defined.</div>'
    total_qty   = sum(t["qty"] for t in _ticket_tiers)
    total_rev   = sum(t["price"] * t["qty"] for t in _ticket_tiers)
    avg_price   = total_rev / total_qty if total_qty else 0
    n_days_fest = w_days.value or 1
    # weighted multiday fraction: tiers whose days_valid >= n_days count as "full festival"
    multiday_qty = sum(t["qty"] for t in _ticket_tiers if t["days"] >= n_days_fest)
    multiday_frac = multiday_qty / total_qty if total_qty else 1.0
    html = (
        f'<div style="background:{C_BG};border-radius:8px;padding:10px 12px;'
        f'font-size:12px;color:{C_TEXT};margin-top:6px">'
        f'<div style="display:flex;justify-content:space-between;margin-bottom:4px">'
        f'<span style="color:{C_TEXT3}">Total tickets sold</span>'
        f'<span style="font-weight:600">{total_qty:,}</span></div>'
        f'<div style="display:flex;justify-content:space-between;margin-bottom:4px">'
        f'<span style="color:{C_TEXT3}">Avg ticket price</span>'
        f'<span style="font-weight:600">€{avg_price:.2f}</span></div>'
        f'<div style="display:flex;justify-content:space-between;margin-bottom:4px">'
        f'<span style="color:{C_TEXT3}">Total revenue</span>'
        f'<span style="font-weight:600">€{total_rev:,.0f}</span></div>'
        f'<div style="display:flex;justify-content:space-between">'
        f'<span style="color:{C_TEXT3}">Multi-day ticket holders</span>'
        f'<span style="font-weight:600">{multiday_frac:.0%}</span></div>'
        f'</div>'
    )
    return html


def _rebuild_ticket_tiers():
    ticket_tier_col.clear()
    if not _ticket_tiers:
        ticket_tier_col.append(pn.pane.HTML(
            f'<div style="font-size:11px;color:{C_TEXT3};font-style:italic;padding:4px 0">'
            f'No ticket types — add at least one.</div>'
        ))
        ticket_summary_pane.object = ""
        return

    # Column headers
    ticket_tier_col.append(pn.pane.HTML(
        f'<div style="display:grid;grid-template-columns:1fr 60px 70px 55px 28px;gap:4px;'
        f'font-size:10px;font-weight:600;color:{C_TEXT3};text-transform:uppercase;'
        f'letter-spacing:0.06em;padding:2px 0 4px">'
        f'<span>Name</span><span>Price</span><span>Qty</span>'
        f'<span>Days</span><span></span></div>'
    ))

    for i, tier in enumerate(_ticket_tiers):
        del_btn = pn.widgets.Button(name="✕", button_type="light", width=26, height=26,
                                     stylesheets=[f"button{{color:{C_RED};font-size:11px;"
                                                  f"padding:0;border-radius:4px;"
                                                  f"border:1px solid {C_BORDER}}}"])
        def _make_del(idx):
            def _del(_):
                if 0 <= idx < len(_ticket_tiers):
                    _ticket_tiers.pop(idx)
                _rebuild_ticket_tiers()
            return _del
        del_btn.on_click(_make_del(i))

        days_label = f"{tier['days']}d"
        ticket_tier_col.append(pn.Row(
            pn.pane.HTML(
                f'<div style="font-size:12px;color:{C_TEXT};padding:5px 0;line-height:1.4">'
                f'<div style="font-weight:500">{tier["name"]}</div>'
                f'<div style="font-size:11px;color:{C_TEXT3}">'
                f'€{tier["price"]:.0f} · {tier["qty"]:,} tickets · {days_label}</div></div>',
                sizing_mode="stretch_width",
            ),
            del_btn,
            align="center",
            sizing_mode="stretch_width",
            stylesheets=[f".bk-Row{{border-bottom:1px solid {C_BORDER};padding:1px 0}}"],
        ))

    ticket_summary_pane.object = _ticket_summary_html()


def _add_tier(_):
    name = _tier_inp_name.value.strip()
    if not name:
        return
    _ticket_tiers.append({
        "name":  name,
        "price": _tier_inp_price.value or 0.0,
        "qty":   _tier_inp_qty.value   or 0,
        "days":  _tier_inp_days.value  or 1,
    })
    _tier_inp_name.value = ""
    _rebuild_ticket_tiers()


def _get_ticket_totals():
    """Derive total tickets, weighted avg price, and multiday fraction from tiers."""
    if not _ticket_tiers:
        return 35000, 95.0, 1.0
    n_days_fest  = w_days.value or 1
    total_qty    = sum(t["qty"] for t in _ticket_tiers)
    total_rev    = sum(t["price"] * t["qty"] for t in _ticket_tiers)
    multiday_qty = sum(t["qty"] for t in _ticket_tiers if t["days"] >= n_days_fest)
    avg_price    = total_rev / total_qty if total_qty else 95.0
    multiday_frac = multiday_qty / total_qty if total_qty else 1.0
    return max(1, total_qty), avg_price, multiday_frac


btn_add_tier.on_click(_add_tier)
w_days.param.watch(lambda _: _rebuild_ticket_tiers(), "value")
_rebuild_ticket_tiers()

# Keep w_tickets / w_price as hidden derived values so existing code still works
w_tickets = pn.widgets.IntInput(name="", value=35000, visible=False)
w_price   = pn.widgets.FloatInput(name="", value=95.0, visible=False)
w_camping_enabled = pn.widgets.Checkbox(name="Festival includes overnight camping", value=False)
w_camping   = pn.widgets.FloatSlider(name="What fraction of ticket holders are campers?",
                                      start=0.05, end=1, step=0.05, value=0.3,
                                      format="0%", **_inp)
w_camping_zone = pn.widgets.Select(name="Which zone is the campsite?",
                                    options=["(none yet — add a zone first)"], **_inp)
w_gate      = pn.widgets.IntInput(name="Gates open at (hour of day, 0–23)", value=10, start=0, end=23, **_inp)
w_headliner = pn.widgets.IntInput(name="Fallback headliner hour", value=19, start=0, end=23, **_inp)
w_runs      = pn.widgets.IntInput(name="MC Runs per Config", value=30, start=5, end=200, **_inp)

# Zone editor
_DEFAULT_ZONES = [
    {"name": "Main Stage",   "area": 20000},
    {"name": "Food Village", "area": 12000},
    {"name": "Chill Zone",   "area": 8000},
]
zone_name_inputs = [pn.widgets.TextInput(value=z["name"], width=120) for z in _DEFAULT_ZONES]
zone_area_inputs = [pn.widgets.IntInput(value=z["area"], start=100, width=90) for z in _DEFAULT_ZONES]


def get_zones() -> List[Dict]:
    return [{"name": zone_name_inputs[i].value or f"zone_{i+1}",
             "area": zone_area_inputs[i].value or 5000}
            for i in range(len(zone_name_inputs))]


def _sync_zone_dropdowns(*_):
    """Keep camping zone and schedule zone selects in sync with the zone list."""
    names = [z["name"] for z in get_zones() if z["name"]]
    if not names:
        names = ["(no zones defined)"]
    # Camping zone
    prev_camp = w_camping_zone.value
    w_camping_zone.options = names
    if prev_camp in names:
        w_camping_zone.value = prev_camp
    else:
        w_camping_zone.value = names[-1]  # default camping to last zone
    # Schedule zone
    prev_sched = w_sched_zone.value if hasattr(w_sched_zone, 'value') else names[0]
    w_sched_zone.options = names
    if prev_sched in names:
        w_sched_zone.value = prev_sched
    else:
        w_sched_zone.value = names[0]


def _zone_row(i: int):
    col = ZONE_PALETTE[i % len(ZONE_PALETTE)]
    dot = pn.pane.HTML(
        f'<div style="width:9px;height:9px;border-radius:50%;background:{col};'
        f'display:inline-block;margin-top:10px;flex-shrink:0"></div>', width=16)
    return pn.Row(dot, zone_name_inputs[i], zone_area_inputs[i],
                  margin=(0, 0, 4, 0), sizing_mode="stretch_width")


_zone_col_header = pn.pane.HTML(
    f'<div style="display:flex;gap:6px;font-size:10px;font-weight:600;color:{C_TEXT3};'
    f'text-transform:uppercase;letter-spacing:0.07em;padding:0 20px;margin-bottom:4px">'
    f'<span style="width:16px"></span>'
    f'<span style="width:120px">Zone Name</span>'
    f'<span style="width:90px">Area (m²)</span></div>'
)
zone_list_col = pn.Column(
    _zone_col_header,
    *[_zone_row(i) for i in range(len(_DEFAULT_ZONES))],
    stylesheets=["padding: 0 20px"],
)

btn_add_zone = pn.widgets.Button(
    name="+ Add Zone", button_type="light", sizing_mode="stretch_width",
    stylesheets=[f"button{{border:1px dashed {C_BORDER};color:{C_TEXT3};border-radius:6px;font-size:12px}}"],
    margin=(4, 20, 0, 20),
)


def _add_zone(_):
    i = len(zone_name_inputs)
    new_name = pn.widgets.TextInput(value=f"Zone {i+1}", width=120)
    zone_name_inputs.append(new_name)
    zone_area_inputs.append(pn.widgets.IntInput(value=8000, start=100, width=90))
    zone_list_col.append(_zone_row(i))
    new_name.param.watch(_sync_zone_dropdowns, "value")
    _sync_zone_dropdowns()


for _zi in range(len(zone_name_inputs)):
    zone_name_inputs[_zi].param.watch(_sync_zone_dropdowns, "value")


btn_add_zone.on_click(_add_zone)

# ── Forecast builder ──────────────────────────────────────────────────────────
_WX_ICONS  = {"clear": "☀", "rain": "🌧", "heat": "🔥"}
_WX_LABELS = {"clear": "Clear", "rain": "Rain", "heat": "Heat wave"}
_WX_BARCOL = {"clear": "#93C5FD", "rain": "#60A5FA", "heat": "#FCA5A5"}

w_fcast_weather = pn.widgets.RadioButtonGroup(
    name="", value="clear",
    options={"☀ Clear": "clear", "🌧 Rain": "rain", "🔥 Heat": "heat"},
    button_type="default", sizing_mode="stretch_width",
    stylesheets=["button{font-size:12px;padding:5px 8px;border-radius:6px}"],
)
w_fcast_hours = pn.widgets.Select(
    name="Duration",
    options={**{f"{h}h": float(h) for h in [2, 4, 6, 8, 12]},
             **{"Full day (24h)": 24.0, "All weekend (48h)": 48.0}},
    value=24.0, **_inp,
)
w_fcast_conf = pn.widgets.Select(
    name="How confident are you in this forecast?",
    options={"Almost certain (95%)": 0.95,
             "Very likely (85%)": 0.85,
             "Likely (70%)": 0.70,
             "Uncertain (50%)": 0.50},
    value=0.85, **_inp,
)
btn_add_fcast   = pn.widgets.Button(name="Add period", button_type="primary",
                                     sizing_mode="stretch_width", height=34,
                                     stylesheets=["button{font-size:12px;border-radius:6px}"])
btn_clear_fcast = pn.widgets.Button(name="Clear all", button_type="light",
                                     width=80, height=34,
                                     stylesheets=[f"button{{font-size:12px;border-radius:6px;"
                                                  f"color:{C_RED};border:1px solid {C_BORDER}}}"])
forecast_timeline = pn.Column(margin=(0, 0, 0, 0))
_forecast_list: List[Dict] = []


def _rebuild_forecast_timeline():
    forecast_timeline.clear()
    if not _forecast_list:
        forecast_timeline.append(pn.pane.HTML(
            f'<div style="font-size:11px;color:{C_TEXT3};padding:6px 4px;font-style:italic">'
            f'No forecast set — simulation uses random weather variation.</div>'
        ))
        return
    total_h = sum(f["hours"] for f in _forecast_list)
    bar_html = '<div style="display:flex;border-radius:6px;overflow:hidden;height:20px;margin-bottom:6px">'
    for f in _forecast_list:
        pct = f["hours"] / total_h * 100
        bar_html += (
            f'<div style="width:{pct:.0f}%;background:{_WX_BARCOL[f["weather"]]};'
            f'display:flex;align-items:center;justify-content:center;'
            f'font-size:10px;font-weight:600;color:white;overflow:hidden;white-space:nowrap;'
            f'padding:0 4px">{_WX_ICONS[f["weather"]]} {f["hours"]:.0f}h</div>'
        )
    bar_html += '</div>'
    rows_html = ""
    for i, f in enumerate(_forecast_list):
        rows_html += (
            f'<div style="display:flex;justify-content:space-between;align-items:center;'
            f'font-size:11px;padding:3px 0;border-bottom:1px solid {C_BORDER}">'
            f'<span>{_WX_ICONS[f["weather"]]} Period {i+1} — {_WX_LABELS[f["weather"]]}</span>'
            f'<span style="color:{C_TEXT3}">{f["hours"]:.0f}h</span></div>'
        )
    forecast_timeline.append(pn.pane.HTML(bar_html + rows_html))


def _add_fcast(_):
    hrs = w_fcast_hours.value
    wth = w_fcast_weather.value
    if hrs and hrs > 0:
        _forecast_list.append({"weather": wth, "hours": float(hrs)})
        _rebuild_forecast_timeline()


def _clear_fcast(_):
    _forecast_list.clear()
    _rebuild_forecast_timeline()


btn_add_fcast.on_click(_add_fcast)
btn_clear_fcast.on_click(_clear_fcast)
_rebuild_forecast_timeline()

# ── Gate / Infrastructure widgets ─────────────────────────────────────────────
w_n_gates     = pn.widgets.IntInput(name="Number of Entrance Gates", value=2, start=1, end=20, **_inp)
w_lanes_gate  = pn.widgets.IntInput(name="Turnstile Lanes per Gate", value=18, start=1, end=100, **_inp)
w_lane_cost   = pn.widgets.FloatInput(name="Lane Hire Cost (€/lane/day)", value=200.0, start=0, **_inp)
w_gate_staff  = pn.widgets.FloatInput(name="Gate Staff Wage (€/person/day)", value=400.0, start=0, **_inp)

# ── Cost parameter widgets ─────────────────────────────────────────────────────
w_cost_staff    = pn.widgets.FloatInput(name="Security/Steward Wage (€/person/day)", value=400.0, start=0, **_inp)
w_cost_vendor   = pn.widgets.FloatInput(name="Vendor Stall Cost (€/stall/day)", value=350.0, start=0, **_inp)
w_cost_toilet   = pn.widgets.FloatInput(name="Toilet Hire (€/cubicle/day)", value=110.0, start=0, **_inp)
w_cost_firstaid = pn.widgets.FloatInput(name="First Aid Bay (€/bay/day)", value=2000.0, start=0, **_inp)
w_cost_zone     = pn.widgets.FloatInput(name="Zone Overhead (€/zone/day)", value=8000.0, start=0, **_inp)
w_cost_viol     = pn.widgets.FloatInput(name="Safety Violation Penalty (€)", value=10000.0, start=0, **_inp)

# ── Schedule builder ──────────────────────────────────────────────────────────
_schedule: List[Dict] = []

w_sched_day    = pn.widgets.Select(name="Day",
                                    options={f"Day {i}": i for i in range(1, 9)},
                                    value=1, width=72)
w_sched_hour   = pn.widgets.Select(name="Time",
                                    options={f"{h:02d}:{m:02d}": h + m/60
                                             for h in range(8, 24) for m in (0, 30)},
                                    value=20.0, width=80)
w_sched_name   = pn.widgets.TextInput(name="Act / Artist Name", value="",
                                       placeholder="e.g. The Prodigy", **_inp)
w_sched_zone   = pn.widgets.Select(name="Stage / Zone",
                                    options=["(add zones first)"], **_inp)
w_sched_headliner = pn.widgets.Toggle(name="⭐ Headliner", value=False,
                                       button_type="warning", width=110,
                                       stylesheets=["button{font-size:12px;border-radius:6px}"])
btn_add_sched  = pn.widgets.Button(name="Add to schedule", button_type="primary",
                                    sizing_mode="stretch_width", height=36,
                                    stylesheets=["button{font-size:12px;border-radius:6px}"])
schedule_list  = pn.Column(margin=(0, 0, 0, 0))


def _zone_options_for_schedule():
    zones = get_zones()
    opts = [z["name"] for z in zones if z["name"]]
    return opts if opts else ["(add zones first)"]


def _rebuild_schedule_list():
    schedule_list.clear()
    # Always sync zone dropdown to current zones
    w_sched_zone.options = _zone_options_for_schedule()
    if not w_sched_zone.value in w_sched_zone.options:
        w_sched_zone.value = w_sched_zone.options[0]

    if not _schedule:
        schedule_list.append(pn.pane.HTML(
            f'<div style="font-size:11px;color:{C_TEXT3};padding:8px 4px;font-style:italic">'
            f'No acts added yet — the schedule is optional but improves crowd flow accuracy.</div>'
        ))
        return

    for i, act in enumerate(_schedule):
        star_badge = (
            f'<span style="background:#FEF3C7;color:#92400E;border-radius:10px;'
            f'padding:1px 7px;font-size:10px;font-weight:600;margin-left:4px">⭐ HEADLINER</span>'
            if act["headliner"] else ""
        )
        # Delete button inline via a separate Panel Button stored per row
        del_btn = pn.widgets.Button(name="✕", button_type="light", width=28, height=28,
                                     stylesheets=[f"button{{color:{C_RED};font-size:11px;"
                                                  f"padding:0;border-radius:4px;border:1px solid {C_BORDER}}}"])
        _act_idx = i  # capture
        def _make_del(idx):
            def _del(_):
                if 0 <= idx < len(_schedule):
                    _schedule.pop(idx)
                _rebuild_schedule_list()
            return _del
        del_btn.on_click(_make_del(i))

        schedule_list.append(pn.Row(
            pn.pane.HTML(
                f'<div style="font-size:12px;color:{C_TEXT};padding:6px 0;line-height:1.5">'
                f'<div style="font-weight:{"700" if act["headliner"] else "500"}">'
                f'🎵 {act["name"]}{star_badge}</div>'
                f'<div style="font-size:11px;color:{C_TEXT3}">'
                f'Day {act["day"]} · {int(act["hour"]):02d}:{int((act["hour"]%1)*60):02d}'
                f' · {act["zone"]}</div></div>',
                sizing_mode="stretch_width",
            ),
            del_btn,
            align="center",
            sizing_mode="stretch_width",
            stylesheets=[f".bk-Row{{border-bottom:1px solid {C_BORDER};padding:2px 0}}"],
        ))


def _add_sched(_):
    name = w_sched_name.value.strip()
    if not name:
        return
    zones = get_zones()
    zone = w_sched_zone.value
    if zone == "(add zones first)" or zone not in [z["name"] for z in zones]:
        zone = zones[0]["name"] if zones else ""
    _schedule.append({
        "day":       w_sched_day.value,
        "hour":      w_sched_hour.value,
        "name":      name,
        "zone":      zone,
        "headliner": w_sched_headliner.value,
    })
    _schedule.sort(key=lambda a: (a["day"], a["hour"]))
    w_sched_name.value = ""
    w_sched_headliner.value = False
    _rebuild_schedule_list()


btn_add_sched.on_click(_add_sched)
_rebuild_schedule_list()


def _schedule_to_act_tuples(gate_open_hour: int) -> tuple:
    """Convert schedule entries to (step_within_day, from_zone, to_zone) tuples."""
    zones = get_zones()
    zone_names = [z["name"] for z in zones]
    acts = []
    for act in _schedule:
        step = int((act["hour"] - gate_open_hour) * 4)
        step = max(0, min(step, 63))
        # from_zone: previous zone or first zone; to_zone: act's zone
        to_zone   = act["zone"] if act["zone"] in zone_names else (zone_names[0] if zone_names else "")
        from_zone = zone_names[1] if len(zone_names) > 1 else to_zone
        acts.append((step, from_zone, to_zone))
    return tuple(acts)


btn_run = pn.widgets.Button(
    name="▶  Analyse Festival", button_type="primary",
    sizing_mode="stretch_width", height=46,
    stylesheets=[
        f"button{{background:{C_PRIMARY};color:white;font-weight:600;"
        f"font-size:14px;border-radius:8px;letter-spacing:0.02em}}"
    ],
    margin=(8, 20, 4, 20),
)

btn_optimise = pn.widgets.Button(
    name="⚡  Find Best Configuration", button_type="success",
    sizing_mode="stretch_width", height=46,
    stylesheets=[
        "button{background:#059669;color:white;font-weight:600;"
        "font-size:14px;border-radius:8px;letter-spacing:0.02em}"
    ],
    margin=(4, 20, 16, 20),
)

run_spinner = pn.indicators.LoadingSpinner(value=False, size=22, color="primary")
run_status  = pn.pane.HTML("", styles={"font-size": "12px", "color": C_TEXT2,
                                        "padding": "4px 20px 8px"})

# ── Sidebar assembly ──────────────────────────────────────────────────────────
sidebar = pn.Column(
    # Header
    pn.pane.HTML(
        f'<div style="padding:20px 20px 14px;border-bottom:1px solid {C_BORDER}">'
        f'<div style="font-size:17px;font-weight:700;color:{C_TEXT};letter-spacing:-0.02em">'
        f'🎪 Festi-Flow</div>'
        f'<div style="font-size:11px;color:{C_TEXT3};margin-top:2px">'
        f'Safety & Capacity Planning</div></div>'
    ),

    section_label("Festival Details"),
    pn.Column(
        w_name,
        pn.Row(w_budget, w_days, sizing_mode="stretch_width"),
        w_tevac,
        styles={"padding": "0 20px"},
    ),

    section_label("Ticket Types"),
    pn.Column(
        pn.pane.HTML(
            f'<div style="font-size:11px;color:{C_TEXT3};margin-bottom:8px;line-height:1.5">'
            f'Define each ticket tier. The simulation derives total attendance, '
            f'average revenue per head, and gate pressure on day 2+ automatically.</div>'
        ),
        ticket_tier_col,
        ticket_summary_pane,
        pn.Spacer(height=8),
        pn.pane.HTML(
            f'<div style="font-size:11px;font-weight:600;color:{C_TEXT};margin-bottom:6px">'
            f'Add a ticket type:</div>'
        ),
        _tier_inp_name,
        pn.Row(_tier_inp_price, _tier_inp_qty, sizing_mode="stretch_width"),
        _tier_inp_days,
        btn_add_tier,
        styles={"padding": "0 20px"},
    ),

    section_label("Timing"),
    pn.Column(
        w_gate,
        pn.pane.HTML(
            f'<div style="font-size:11px;color:{C_TEXT3};margin-top:2px;margin-bottom:8px">'
            f'The hour the entrance gates open, e.g. 10 = 10:00. '
            f'Arrivals begin here and build toward the headliner peak.</div>'
        ),
        w_headliner,
        pn.pane.HTML(
            f'<div style="font-size:11px;color:{C_TEXT3};margin-top:2px">'
            f'Used only when no ⭐ Headliner is marked in the schedule. '
            f'If a headliner is added, their act time overrides this automatically.</div>'
        ),
        styles={"padding": "0 20px"},
    ),

    section_label("Venue Zones"),
    zone_list_col,
    btn_add_zone,

    section_label("Camping"),
    pn.Column(
        w_camping_enabled,
        pn.pane.HTML(
            f'<div style="font-size:11px;color:{C_TEXT3};margin:6px 0 4px">'
            f'Campers arrive on Day 1 and stay overnight — they don\'t pass through '
            f'the gates again on Day 2+, so they reduce gate pressure on multi-day events.</div>'
        ),
        pn.pane.HTML(
            f'<div id="camping-fields" style="display:none"><!-- shown via JS when enabled --></div>'
        ),
        w_camping,
        w_camping_zone,
        styles={"padding": "0 20px"},
    ),

    section_label("Entrance Infrastructure"),
    pn.Column(
        pn.Row(w_n_gates, w_lanes_gate, sizing_mode="stretch_width"),
        pn.Row(w_lane_cost, w_gate_staff, sizing_mode="stretch_width"),
        pn.pane.HTML(
            f'<div style="font-size:11px;color:{C_TEXT3};margin-top:2px">'
            f'Leave blank — the optimiser will calculate the minimum lanes needed. '
            f'1 staff member deployed per lane. Edit costs in ⚙ Settings.</div>'
        ),
        styles={"padding": "0 20px"},
    ),

    section_label("Performance Schedule  (optional)"),
    pn.Column(
        pn.pane.HTML(
            f'<div style="font-size:11px;color:{C_TEXT3};margin-bottom:10px;line-height:1.5">'
            f'Adding acts lets the simulation model crowd movement between zones '
            f'(e.g. everyone moving to the main stage at headliner time). '
            f'Mark the biggest act as ⭐ Headliner — that sets the pre-show arrival surge peak.</div>'
        ),
        w_sched_name,
        pn.Row(w_sched_day, w_sched_hour, sizing_mode="stretch_width"),
        w_sched_zone,
        w_sched_headliner,
        btn_add_sched,
        pn.Spacer(height=4),
        schedule_list,
        styles={"padding": "0 20px"},
    ),

    section_label("Weather Forecast  (optional)"),
    pn.Column(
        pn.pane.HTML(
            f'<div style="font-size:11px;color:{C_TEXT3};margin-bottom:10px;line-height:1.5">'
            f'Build a weather timeline for your event. The simulation will follow this '
            f'forecast with the confidence level you set — the rest of the time it uses '
            f'random variation. Leave blank for fully random weather.</div>'
        ),
        pn.pane.HTML(f'<div style="font-size:11px;font-weight:600;color:{C_TEXT};margin-bottom:4px">'
                     f'Select weather type:</div>'),
        w_fcast_weather,
        w_fcast_hours,
        pn.Row(btn_add_fcast, btn_clear_fcast, sizing_mode="stretch_width"),
        pn.Spacer(height=4),
        forecast_timeline,
        pn.Spacer(height=8),
        w_fcast_conf,
        styles={"padding": "0 20px"},
    ),

    section_label("Simulation"),
    pn.Column(w_runs, styles={"padding": "0 20px"}),

    pn.Spacer(height=8),
    btn_run,
    btn_optimise,
    pn.Row(run_spinner, run_status, align="center", margin=(0, 20)),
    pn.Spacer(height=20),

    width=360,
    stylesheets=[SIDEBAR_CSS],
    scroll=True,
)


# ── Result panes ──────────────────────────────────────────────────────────────
card_peak  = metric_card("Peak On-Site",       "—")
card_ever  = metric_card("Total Admitted",      "—")
card_safe  = metric_card("Safe Runs",           "—")
card_dens  = metric_card("Avg Crowd Density",   "—")
card_score = metric_card("Operator Score",      "—")
card_warns = metric_card("Density Warnings",    "—")

pane_verdict = pn.pane.HTML(
    f'<div style="background:{C_BG};border:1px solid {C_BORDER};border-radius:10px;'
    f'padding:16px 20px;font-size:12px;color:{C_TEXT3}">'
    f'Run the simulation to see the scenario verdict.</div>',
    sizing_mode="stretch_width",
)

pane_occ   = pn.pane.Plotly(empty_fig(height=320), config={"displayModeBar": False},
                              sizing_mode="stretch_width")
pane_feas  = pn.pane.Plotly(empty_fig(height=320), config={"displayModeBar": False},
                              sizing_mode="stretch_width")
pane_dens  = pn.pane.Plotly(empty_fig(height=300), config={"displayModeBar": False},
                              sizing_mode="stretch_width")
pane_q     = pn.pane.Plotly(empty_fig(height=300), config={"displayModeBar": False},
                              sizing_mode="stretch_width")
pane_mc    = pn.pane.Plotly(empty_fig(height=300), config={"displayModeBar": False},
                              sizing_mode="stretch_width")
pane_res   = pn.pane.HTML(
    f'<p style="color:{C_TEXT3};font-size:12px">Run the simulation to see staffing requirements.</p>')
pane_pfeas = pn.pane.HTML(
    f'<p style="color:{C_TEXT3};font-size:12px">Run the simulation to see feasibility by weather.</p>')
pane_cost  = pn.pane.HTML(
    f'<p style="color:{C_TEXT3};font-size:12px">Run the simulation to see cost estimates.</p>')
pane_bayes = pn.pane.HTML(
    f'<p style="color:{C_TEXT3};font-size:12px">Run the simulation to see capacity guidance.</p>')

# Optimiser panes
pane_opt_summary = pn.pane.HTML(
    f'<p style="color:{C_TEXT3};font-size:12px">'
    f'Click <b>Find Best Configuration</b> to sweep configurations within your budget.</p>')
pane_opt_chart   = pn.pane.Plotly(empty_fig("Click 'Find Best Configuration' to start", height=340),
                                   config={"displayModeBar": False}, sizing_mode="stretch_width")
pane_opt_detail  = pn.pane.HTML(
    f'<p style="color:{C_TEXT3};font-size:12px">Top configurations will appear here.</p>')
pane_opt_gates   = pn.pane.HTML(
    f'<p style="color:{C_TEXT3};font-size:12px">Gate sizing will appear here after optimisation.</p>')
pane_opt_gate_chart = pn.pane.Plotly(empty_fig("Run optimiser to see gate sweep", height=260),
                                      config={"displayModeBar": False}, sizing_mode="stretch_width")
pane_opt_weather = pn.pane.Plotly(empty_fig("Run optimiser to see weather cost comparison", height=300),
                                   config={"displayModeBar": False}, sizing_mode="stretch_width")

page_title = pn.pane.HTML(
    f'<div style="font-size:20px;font-weight:700;color:{C_TEXT};letter-spacing:-0.02em">'
    f'MyFestival_2026</div>'
    f'<div style="font-size:12px;color:{C_TEXT3};margin-top:2px">'
    f'Festival safety simulation & capacity optimisation</div>'
)


def _sync_title(_):
    name = w_name.value or "Festival"
    page_title.object = (
        f'<div style="font-size:20px;font-weight:700;color:{C_TEXT};letter-spacing:-0.02em">'
        f'{name}</div>'
        f'<div style="font-size:12px;color:{C_TEXT3};margin-top:2px">'
        f'Festival safety simulation & capacity optimisation</div>'
    )


w_name.param.watch(_sync_title, "value")


# ── Tab layouts ───────────────────────────────────────────────────────────────
tab_overview = pn.Column(
    pane_verdict,
    pn.Spacer(height=8),
    pn.GridBox(card_peak, card_ever, card_safe, card_dens, card_score, card_warns,
               ncols=6, sizing_mode="stretch_width"),
    pn.Spacer(height=10),
    pn.Row(
        card(pane_occ,  title="People On-Site Over Time",
             subtitle="Shading shows weather state · × marks a safety violation"),
        card(pane_feas, title="Venue Capacity vs. Expected Attendance",
             subtitle="Bars show maximum safe capacity (A_max) vs. effective attendance per weather"),
        sizing_mode="stretch_width",
    ),
    sizing_mode="stretch_width",
)

tab_crowd = pn.Row(
    card(pane_dens, title="Zone Crowd Density",
         subtitle="Average spatial density per zone — warning at 1.5 p/m², safety violation at 1.8 p/m²"),
    card(pane_q,    title="Food & Drink Queue Depth",
         subtitle="Queue length per stall — extra stalls are deployed automatically above the threshold"),
    sizing_mode="stretch_width",
)

tab_mc = pn.Row(
    card(pane_mc,  title="Operator Score Distribution",
         subtitle="Score = 3×Revenue − 2×Cost − Crowd Density − Queue Penalty − Safety Penalty, across all MC runs"),
    card(pane_res, title="Minimum Staffing & Resource Requirements",
         subtitle="Based on SGSA safety ratios for the configured attendance"),
    sizing_mode="stretch_width",
)

tab_planning = pn.Column(
    pn.Row(
        card(pane_pfeas, title="Can the Venue Handle Attendance?",
             subtitle="Feasibility check per weather condition — effective attendance must stay below A_max"),
        card(pane_cost,  title="Estimated Operational Cost",
             subtitle="Baseline (clear weather) plus weather-adjusted scenarios"),
        sizing_mode="stretch_width",
    ),
    card(pane_bayes, title="Weather-Based Capacity Decisions",
         subtitle="Bayesian rules for adjusting admitted attendance based on forecast confidence"),
    sizing_mode="stretch_width",
)

tab_optimise = pn.Column(
    pn.pane.HTML(
        f'<div style="background:{C_BLUE_L};border:1px solid #BFDBFE;border-radius:8px;'
        f'padding:14px 18px;margin-bottom:4px">'
        f'<div style="font-size:13px;font-weight:600;color:#1E40AF;margin-bottom:4px">'
        f'How the optimiser works</div>'
        f'<div style="font-size:12px;color:#1E3A8A;line-height:1.6">'
        f'Given your budget, total area, and attendance range, the optimiser sweeps '
        f'across evacuation times (6–15 min), stage counts (1–3), and zone splits '
        f'(concentrated vs. distributed). Each configuration is evaluated with Monte Carlo '
        f'simulation. Results are ranked by <b>Operator Score × % Safe Runs</b> — '
        f'so a high-scoring but unsafe configuration will not win. '
        f'Gate count and lane layout are also auto-sized by simulation: the optimiser finds '
        f'the minimum lanes that reduce entrance surge to under 10% of gate-open time.</div></div>'
    ),
    pn.Row(
        card(pane_opt_chart,   title="Configuration Comparison",
             subtitle="Each point is one tested configuration — higher and further right is better"),
        card(pane_opt_summary, title="Recommended Configuration",
             subtitle="Best configuration found within your budget"),
        sizing_mode="stretch_width",
    ),
    pn.Row(
        card(pane_opt_gate_chart, title="Gate Infrastructure Sizing",
             subtitle="Surge rate vs total lanes — optimiser finds the minimum lane count that clears congestion"),
        card(pane_opt_gates, title="Recommended Gate Layout",
             subtitle="Derived from simulation — no manual input required"),
        sizing_mode="stretch_width",
    ),
    card(pane_opt_weather, title="Top 5 — Cost vs Weather Scenario",
         subtitle="Each configuration's daily cost under clear / rain / heat — budget line shown if set"),
    card(pane_opt_detail, title="Top 5 Configurations",
         subtitle="Ranked by Operator Score × Feasibility — all passed the worst-case (heat) budget filter"),
    sizing_mode="stretch_width",
)

def _cost_field(label, value_widget, description):
    return pn.Column(
        pn.pane.HTML(
            f'<div style="font-size:12px;font-weight:500;color:{C_TEXT};margin-bottom:2px">{label}</div>'
            f'<div style="font-size:11px;color:{C_TEXT3};margin-bottom:4px">{description}</div>'
        ),
        value_widget,
        sizing_mode="stretch_width",
    )

tab_settings = pn.Column(
    pn.pane.HTML(
        f'<div style="background:{C_YELLOW_L};border:1px solid #FDE68A;border-radius:8px;'
        f'padding:12px 16px;margin-bottom:16px;font-size:12px;color:{C_YELLOW}">'
        f'These defaults are based on the Dutch <b>CAO Veiligheidsdomein 2025–2027</b> and '
        f'NL festival industry rates (2026). Override any value before running the simulation.</div>'
    ),
    pn.Row(
        card(
            pn.Column(
                _cost_field("Security / Steward Wage", w_cost_staff,
                            "Per person per day — VVNL CAO 2026 ~€40/hr × 10hr shift"),
                _cost_field("Vendor Stall Infrastructure", w_cost_vendor,
                            "Organiser-side cost per stall per day (power, water, waste)"),
                _cost_field("Portable Toilet Hire", w_cost_toilet,
                            "Per cubicle per day including one mid-day service"),
                _cost_field("First Aid Bay", w_cost_firstaid,
                            "Per bay per day — Aljohani & Kennedy 2016"),
                sizing_mode="stretch_width",
            ),
            title="Zone Operations",
            subtitle="Staff, vendors, facilities — scaled automatically by zone attendance",
        ),
        card(
            pn.Column(
                _cost_field("Zone Overhead", w_cost_zone,
                            "Fixed cost per zone per day (power, comms, fencing, lighting)"),
                _cost_field("Safety Violation Penalty", w_cost_viol,
                            "Per violation — Dutch municipal penalty schedule mid-range"),
                _cost_field("Turnstile Lane Hire", w_lane_cost,
                            "Per lane per day — portable RFID scanner rental incl. setup"),
                _cost_field("Gate Staff Wage", w_gate_staff,
                            "Per gate staff member per day — 1 deployed per lane"),
                sizing_mode="stretch_width",
            ),
            title="Infrastructure & Penalties",
            subtitle="Fixed overheads and risk costs",
        ),
        sizing_mode="stretch_width",
    ),
    sizing_mode="stretch_width",
)

tabs = pn.Tabs(
    ("Overview",    tab_overview),
    ("Crowd Flow",  tab_crowd),
    ("Monte Carlo", tab_mc),
    ("Planning",    tab_planning),
    ("⚡ Optimise", tab_optimise),
    ("⚙ Settings",  tab_settings),
    dynamic=True,
    stylesheets=[
        f".bk-tab{{font-size:13px;font-weight:500;color:{C_TEXT2};padding:10px 18px}}"
        f".bk-tab.bk-active{{color:{C_PRIMARY};border-bottom:2px solid {C_PRIMARY};font-weight:600}}"
    ],
)

content = pn.Column(
    pn.Row(page_title, pn.Spacer(), sizing_mode="stretch_width",
           styles={"border-bottom": f"1px solid {C_BORDER}",
                   "padding-bottom": "14px", "margin-bottom": "18px"}),
    tabs,
    styles={"padding": "22px 26px", "background": C_BG},
    sizing_mode="stretch_both",
)

layout = pn.Row(sidebar, content, sizing_mode="stretch_both")


# ── Simulation helpers ────────────────────────────────────────────────────────
def _build_zones(zones_data: List[Dict], ticket_sales: int, t_evac_min: int,
                 camping_zone: str = None) -> Dict[str, ZoneSpec]:
    total_area = sum(z["area"] for z in zones_data) or 1
    auto_exits = _estimate_exit_widths(
        {z["name"]: z["area"] for z in zones_data}, ticket_sales, t_evac_min
    )
    return {
        z["name"]: ZoneSpec(
            name=z["name"], area_m2=float(z["area"]), n_gates=0,
            exit_width_m=auto_exits.get(z["name"], 10.0),
            arrival_share=z["area"] / total_area,
            v_z=max(1, int(np.ceil(ticket_sales * z["area"] / total_area / 250))),
            is_exogenous=(z["name"].lower() == "camping" or
                          z["name"] == camping_zone),
        )
        for z in zones_data
    }


def _colour_for(value, lo, hi, low_good=True):
    if low_good:
        return C_GREEN if value <= lo else (C_YELLOW if value <= hi else C_RED)
    return C_GREEN if value >= hi else (C_YELLOW if value >= lo else C_RED)


_running = False


def _resolve_inputs() -> dict:
    """Return all simulation inputs with sensible defaults for any empty fields."""
    n_days       = w_days.value        or 1
    tickets, price, multiday_ticket_frac = _get_ticket_totals()
    budget       = w_budget.value      or None
    t_evac       = w_tevac.value       or 10
    camping      = (w_camping.value or 0.3) if w_camping_enabled.value else 0.0
    camping_zone = w_camping_zone.value if w_camping_enabled.value else None
    gate_hour    = w_gate.value        or 10
    headliner_h  = w_headliner.value   or 19
    n_runs       = w_runs.value        or 30
    name         = (w_name.value or "").strip() or "Festival"
    # Gates: if not filled, calculate minimum needed (SGSA: 1 lane per ~300 arrivals/hr peak)
    n_gates      = w_n_gates.value     or max(1, round(tickets / 5000))
    lanes_gate   = w_lanes_gate.value  or max(5, round(tickets / (n_gates * 300)))
    lane_cost    = w_lane_cost.value   or 200.0
    gate_staff   = w_gate_staff.value  or 400.0
    # Cost params — fall back to CAO defaults if blank
    from main import CostParams as _CP
    _d = _CP()
    cost_params_kw = dict(
        omega_s        = w_cost_staff.value    or _d.omega_s,
        omega_v        = w_cost_vendor.value   or _d.omega_v,
        omega_t        = w_cost_toilet.value   or _d.omega_t,
        omega_f        = w_cost_firstaid.value or _d.omega_f,
        omega_z        = w_cost_zone.value     or _d.omega_z,
        omega_viol     = w_cost_viol.value     or _d.omega_viol,
        omega_lane     = lane_cost,
        omega_gate_staff = gate_staff,
    )
    zones_data   = get_zones()
    return dict(
        name=name, tickets=tickets, price=price, budget=budget,
        n_days=n_days, t_evac=t_evac, camping=camping, camping_zone=camping_zone,
        multiday_ticket_frac=multiday_ticket_frac,
        gate_hour=gate_hour, headliner_h=headliner_h, n_runs=n_runs,
        n_gates=n_gates, lanes_gate=lanes_gate,
        cost_params_kw=cost_params_kw, zones_data=zones_data,
    )


def _run_simulation():
    global _running
    _running = True
    run_spinner.value = True
    run_status.object = "Simulating…"
    try:
        r = _resolve_inputs()
        zones_data   = r["zones_data"]
        ticket_sales = r["tickets"]
        t_evac_min   = r["t_evac"]
        n_days       = r["n_days"]
        camping_frac = r["camping"]
        gate_hour    = r["gate_hour"]
        headliner_h  = r["headliner_h"]
        n_runs       = r["n_runs"]
        budget       = r["budget"]
        name         = r["name"]
        n_gates      = r["n_gates"]
        lanes_gate   = r["lanes_gate"]

        from main import CostParams
        cost_params = CostParams(**r["cost_params_kw"])
        entrance_lanes = {f"gate_{i+1}": lanes_gate for i in range(n_gates)}

        # Derive headliner hour from schedule if a headliner act is marked
        headliners = [a for a in _schedule if a["headliner"] and a["day"] == 1]
        effective_headliner_h = headliners[0]["hour"] if headliners else headliner_h

        # N stages = number of unique zones with acts scheduled (min 1)
        sched_zones = {a["zone"] for a in _schedule if a["zone"]}
        n_stages    = max(1, len(sched_zones)) if _schedule else 1
        staggered   = len({a["hour"] for a in _schedule if a["headliner"]}) > 1

        plan = FestivalPlan(
            name=name,
            zone_areas={z["name"]: float(z["area"]) for z in zones_data},
            entrance_lanes=entrance_lanes,
            ticket_sales=ticket_sales, ticket_price=r["price"],
            n_days=n_days, t_evac_min=t_evac_min,
            is_camping=camping_frac > 0, camping_fraction=camping_frac,
            sold_out_fraction=1.0,
            multiday_ticket_fraction=r["multiday_ticket_frac"],
            total_budget=float(budget) if budget else None,
            n_stages=n_stages, staggered_end_times=staggered,
            gate_open_hour=gate_hour, headliner_start_hour=int(effective_headliner_h),
            n_runs=n_runs, seed=2026,
        )
        report = generate_plan(plan, cost_params=cost_params)
        s      = report.summary

        # Detailed single run for timeline charts
        act_schedule = _schedule_to_act_tuples(gate_hour)
        ft = FestivalType(n_days=n_days, is_camping=camping_frac > 0,
                          camping_fraction=camping_frac, sold_out_fraction=1.0)
        zones    = _build_zones(zones_data, ticket_sales, t_evac_min, r["camping_zone"])
        scenario = make_scenario(
            name=name, a_total=ticket_sales, t_evac_min=t_evac_min,
            ticket_price=r["price"], festival_type=ft,
            n_stages=n_stages, staggered_end_times=staggered,
            gate_open_hour=gate_hour, headliner_start_hour=int(effective_headliner_h),
            seed=2026,
            act_schedule=act_schedule,
            forecast=tuple((f["weather"], f["hours"]) for f in _forecast_list),
            forecast_confidence=float(w_fcast_conf.value),
        )
        timeline, zone_tl = run_festival_once(scenario, zones=zones, seed=2026)
        hours      = timeline["t"] * 0.25
        zone_names = list(plan.zone_areas.keys())

        # ── Metric cards ──────────────────────────────────────────
        safe_col  = _colour_for(s["frac_feasible"], 0.8, 0.5, low_good=False)
        dens_col  = _colour_for(s["mean_D"], 1.5, 1.8)
        warn_col  = _colour_for(s["mean_W_density"], 0, 20)

        # Total admitted = unique ticket holders (≤ ticket_sales).
        # mean_ever_admitted counts gate scans across all days — on a 2-day non-camping
        # event every day-tripper scans in twice, so it can exceed ticket_sales.
        unique_attendees = min(s["mean_ever_admitted"], ticket_sales)
        gate_scans       = s["mean_ever_admitted"]
        ever_sub = (f"total gate scans: {gate_scans:,.0f} ({n_days}-day event)"
                    if n_days > 1 else "")

        card_peak.object  = metric_card("Peak On-Site",     f"{s['mean_peak_occ']:,.0f}",        C_PRIMARY).object
        card_ever.object  = metric_card("Unique Attendees", f"{unique_attendees:,.0f}",           C_ACCENT,
                                         sub=ever_sub).object
        card_safe.object  = metric_card("Safe Runs",        f"{s['frac_feasible']:.0%}",          safe_col,
                                         sub="% of MC runs with zero violations").object
        card_dens.object  = metric_card("Avg Crowd Density",f"{s['mean_D']:.3f} p/m²",           dens_col,
                                         sub="warn ≥1.5  ·  violation ≥1.8").object
        card_score.object = metric_card("Operator Score",   f"{s['E_uO']:,.0f}",                  C_TEXT,
                                         sub="revenue − cost − safety penalty").object
        card_warns.object = metric_card("Density Warnings", f"{s['mean_W_density']:.0f}",         warn_col,
                                         sub="avg soft crowding alerts / run").object

        # ── Scenario verdict ──────────────────────────────────────
        wl_map   = {"clear": ("☀ Clear", 1.00), "rain": ("🌧 Rain", 1.15), "heat": ("🔥 Heat", 1.40)}
        budget   = r["budget"]

        issues = []   # list of (severity, message)  severity: "error" | "warn"

        # 1. Safety: fraction of MC runs with zero violations
        if s["frac_feasible"] < 0.5:
            issues.append(("error", f"Safety: only {s['frac_feasible']:.0%} of MC runs were violation-free — this layout is unsafe"))
        elif s["frac_feasible"] < 0.8:
            issues.append(("warn",  f"Safety: {s['frac_feasible']:.0%} of MC runs were safe — consider reducing attendance or adding zones"))

        # 2. Density vs violations — explain the gap if they differ
        if s["mean_D"] >= 1.8:
            issues.append(("error", f"Crowd density {s['mean_D']:.2f} p/m² exceeds safety limit (1.8 p/m²)"))
        elif s["mean_D"] >= 1.5:
            issues.append(("warn",  f"Crowd density {s['mean_D']:.2f} p/m² is in the warning band (1.5–1.8 p/m²)"))
        if s["frac_feasible"] < 1.0 and s["mean_D"] < 1.5:
            # Violations are happening but NOT from crowd density — must be egress capacity
            issues.append(("warn",
                f"Safety violations detected despite low density ({s['mean_D']:.2f} p/m²). "
                f"This usually means total headcount exceeds the egress-based capacity limit "
                f"(how many people can exit safely in {r['t_evac']} min). "
                f"Try increasing evacuation time, adding exit width, or reducing attendance."))

        # 3. Budget — check ALL weather scenarios, not just baseline
        if budget:
            for wk, (wlabel, wmult) in wl_map.items():
                scenario_cost = report.baseline_cost * wmult
                if scenario_cost > budget:
                    issues.append(("error" if wk == "clear" else "warn",
                                   f"Cost {wlabel}: €{scenario_cost:,.0f} exceeds budget of €{budget:,.0f}/day"))

        # 4. Capacity: any weather where attendance exceeds A_max
        for wk, (wlabel, _) in wl_map.items():
            eff = int(ticket_sales * (1 - report.noshow_rates[wk]))
            amax = report.a_max_by_weather[wk]
            if eff > amax:
                issues.append(("error", f"Capacity {wlabel}: {eff:,} expected > {amax:,} max safe — overcapacity"))

        errors = [m for sev, m in issues if sev == "error"]
        warns  = [m for sev, m in issues if sev == "warn"]

        if errors:
            verdict_color   = C_RED
            verdict_bg      = C_RED_L
            verdict_border  = "#FECACA"
            verdict_icon    = "✗"
            verdict_label   = "Not feasible as configured"
        elif warns:
            verdict_color   = C_YELLOW
            verdict_bg      = C_YELLOW_L
            verdict_border  = "#FDE68A"
            verdict_icon    = "⚠"
            verdict_label   = "Feasible with caution"
        else:
            verdict_color   = C_GREEN
            verdict_bg      = C_GREEN_L
            verdict_border  = "#BBF7D0"
            verdict_icon    = "✓"
            verdict_label   = "Scenario looks good"

        # Revenue from actual tier definitions (not avg_price × count)
        tier_revenue = sum(t["price"] * t["qty"] for t in _ticket_tiers) if _ticket_tiers else ticket_sales * r["price"]
        margin       = tier_revenue - report.baseline_cost
        n_tiers      = len(_ticket_tiers)
        tier_note    = f"{n_tiers} ticket type{'s' if n_tiers != 1 else ''}" if n_tiers else "single tier"
        summary_line = (
            f"<b>{ticket_sales:,}</b> attendees ({tier_note}) · "
            f"avg <b>€{r['price']:.0f}</b>/ticket · "
            f"<b>{s['frac_feasible']:.0%}</b> safe runs · "
            f"peak density <b>{s['mean_D']:.2f} p/m²</b> · "
            f"revenue <b>€{tier_revenue:,.0f}</b> · "
            f"baseline cost <b>€{report.baseline_cost:,.0f}</b>/day · "
            f"gross margin <b>{'+'if margin>=0 else ''}€{margin:,.0f}</b>"
        )

        def _issue_row(sev, msg):
            ic = "✗" if sev == "error" else "⚠"
            c  = C_RED if sev == "error" else C_YELLOW
            return (f'<div style="display:flex;align-items:flex-start;gap:6px;'
                    f'margin-top:5px;font-size:12px;color:{C_TEXT}">'
                    f'<span style="color:{c};font-weight:700;flex-shrink:0">{ic}</span>'
                    f'<span>{msg}</span></div>')

        issue_html = "".join(_issue_row(sev, msg) for sev, msg in issues) if issues else (
            f'<div style="font-size:12px;color:{C_GREEN};margin-top:5px">'
            f'No issues detected across all weather scenarios.</div>'
        )

        pane_verdict.object = (
            f'<div style="background:{verdict_bg};border:1px solid {verdict_border};'
            f'border-radius:10px;padding:16px 20px">'
            f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:8px">'
            f'<span style="font-size:18px;font-weight:700;color:{verdict_color}">'
            f'{verdict_icon} {verdict_label}</span>'
            f'<span style="font-size:11px;color:{C_TEXT3};margin-left:4px">'
            f'— {name}</span></div>'
            f'<div style="font-size:12px;color:{C_TEXT2};margin-bottom:8px">{summary_line}</div>'
            f'{issue_html}'
            f'</div>'
        )

        # ── Occupancy chart ────────────────────────────────────────
        fig_occ = go.Figure()
        prev_w, prev_h = timeline["weather"].iloc[0], hours.iloc[0]
        for i in range(1, len(timeline)):
            w = timeline["weather"].iloc[i]
            if w != prev_w or i == len(timeline) - 1:
                fig_occ.add_vrect(x0=prev_h, x1=hours.iloc[i],
                                  fillcolor=W_COL[prev_w], opacity=0.06, line_width=0)
                prev_w, prev_h = w, hours.iloc[i]
        fig_occ.add_trace(go.Scatter(x=hours, y=timeline["total_a"], mode="lines",
                                     name="People on site",
                                     line=dict(color=C_PRIMARY, width=2.5),
                                     fill="tozeroy", fillcolor="rgba(26,86,219,0.07)"))
        fig_occ.add_trace(go.Scatter(x=hours, y=timeline["total_ever_admitted"], mode="lines",
                                     name="Cumulative through gates",
                                     line=dict(color=C_ACCENT, width=1.5, dash="dot")))
        viols = timeline[timeline["V"].diff().fillna(0) > 0]
        if not viols.empty:
            fig_occ.add_trace(go.Scatter(
                x=hours.iloc[viols.index], y=timeline.loc[viols.index, "total_a"],
                mode="markers", name="Safety violation",
                marker=dict(color=C_RED, size=9, symbol="x-thin",
                            line=dict(width=2.5, color=C_RED))))
        for d in timeline["day"].unique()[1:]:
            sep = hours.iloc[timeline[timeline["day"] == d].index[0]]
            fig_occ.add_vline(x=sep, line_dash="dot", line_color=C_BORDER, line_width=1.5)
            fig_occ.add_annotation(x=sep + 0.2, y=0.97, yref="paper", text=f"Day {d}",
                                   showarrow=False, font=dict(color=C_TEXT3, size=10))
        fig_occ.update_layout(**base_layout(320))
        fig_occ.update_xaxes(title_text="Hour of event")
        fig_occ.update_yaxes(title_text="Number of people")
        pane_occ.object = fig_occ

        # ── Capacity bars ──────────────────────────────────────────
        wl     = ["☀  Clear", "🌧  Rain", "🔥  Heat"]
        wk     = ["clear", "rain", "heat"]
        amax_v = [report.a_max_by_weather[w] for w in wk]
        noshow = report.noshow_rates
        eff_v  = [int(ticket_sales * (1 - noshow[w])) for w in wk]
        b_cols = [C_GREEN if eff_v[i] <= amax_v[i] else C_RED for i in range(3)]
        fig_cap = go.Figure()
        fig_cap.add_trace(go.Bar(name="Max safe capacity (A_max)", x=wl, y=amax_v,
                                  marker_color="#BFDBFE", width=0.5,
                                  text=[f"{v:,.0f}" for v in amax_v],
                                  textposition="outside", textfont=dict(color=C_TEXT2, size=10)))
        fig_cap.add_trace(go.Bar(name="Expected attendance", x=wl, y=eff_v,
                                  marker_color=b_cols, width=0.28,
                                  text=[f"{v:,.0f}" for v in eff_v],
                                  textposition="outside", textfont=dict(color=C_TEXT, size=10)))
        fig_cap.update_layout(**base_layout(320))
        fig_cap.update_layout(barmode="overlay",
                               legend=dict(orientation="h", y=-0.28, x=0))
        pane_feas.object = fig_cap

        # ── Zone density ───────────────────────────────────────────
        fig_dens = go.Figure()
        fig_dens.add_hline(y=THETA_WARN["clear"], line_dash="dot", line_color=C_YELLOW,
                           opacity=0.7,
                           annotation_text="Warning threshold (1.5 p/m²)",
                           annotation_font=dict(color=C_YELLOW, size=10))
        fig_dens.add_hline(y=THETA_VIOLATION["clear"], line_dash="dash", line_color=C_RED,
                           opacity=0.7,
                           annotation_text="Safety violation (1.8 p/m²)",
                           annotation_font=dict(color=C_RED, size=10))
        for i, zname in enumerate(zone_names):
            zdf = zone_tl[zone_tl["zone"] == zname]
            if zdf.empty:
                continue
            fig_dens.add_trace(go.Scatter(x=zdf["t"] * 0.25, y=zdf["density"],
                                          mode="lines", name=zname,
                                          line=dict(color=ZONE_PALETTE[i % len(ZONE_PALETTE)], width=2)))
        fig_dens.update_layout(**base_layout(300))
        fig_dens.update_xaxes(title_text="Hour of event")
        fig_dens.update_yaxes(title_text="People per m²")
        pane_dens.object = fig_dens

        # ── Vendor queues ──────────────────────────────────────────
        fig_q = go.Figure()
        fig_q.add_hline(y=10, line_dash="dot", line_color=C_YELLOW, opacity=0.6,
                        annotation_text="Extra stall deployed (10 people/stall)",
                        annotation_font=dict(color=C_YELLOW, size=10))
        for i, zname in enumerate(zone_names):
            zdf = zone_tl[zone_tl["zone"] == zname].copy()
            if zdf.empty:
                continue
            zdf["q_per_stall"] = zdf["q_vendor"] / (zdf["extra_stalls"] + 1).clip(lower=1)
            fig_q.add_trace(go.Scatter(x=zdf["t"] * 0.25, y=zdf["q_per_stall"],
                                       mode="lines", name=zname,
                                       line=dict(color=ZONE_PALETTE[i % len(ZONE_PALETTE)], width=2)))
        fig_q.update_layout(**base_layout(300))
        fig_q.update_xaxes(title_text="Hour of event")
        fig_q.update_yaxes(title_text="People waiting per stall")
        pane_q.object = fig_q

        # ── MC histogram ───────────────────────────────────────────
        mc_uo   = report.mc_results["u_O"].tolist()
        mean_uo = float(np.mean(mc_uo))
        fig_mc  = go.Figure()
        fig_mc.add_trace(go.Histogram(x=mc_uo, nbinsx=20, name="Operator Score",
                                      marker=dict(color=C_PRIMARY, opacity=0.80,
                                                  line=dict(color="white", width=0.5))))
        fig_mc.add_vline(x=mean_uo, line_dash="dash", line_color=C_RED, line_width=2)
        fig_mc.add_annotation(x=mean_uo, y=1, yref="paper",
                              text=f"  avg {mean_uo:,.0f}", showarrow=False,
                              font=dict(color=C_RED, size=11), xanchor="left")
        fig_mc.update_layout(**base_layout(300))
        fig_mc.update_xaxes(title_text="Operator Score  (revenue − cost − safety penalty)")
        fig_mc.update_yaxes(title_text="Number of simulated runs")
        pane_mc.object = fig_mc

        # ── Resource table ─────────────────────────────────────────
        resources = report.resources
        totals    = {k: 0 for k in ("attendance", "staff", "vendor_stalls", "toilets", "first_aid")}
        rows = []
        for zname, r in resources.items():
            for k in totals:
                totals[k] += r[k]
            col = ZONE_PALETTE[list(resources.keys()).index(zname) % len(ZONE_PALETTE)]
            rows.append([
                f'<span style="display:inline-flex;align-items:center;gap:6px">'
                f'<span style="width:8px;height:8px;border-radius:50%;background:{col};'
                f'display:inline-block"></span>{zname}</span>',
                f"{r['attendance']:,}", str(r["staff"]),
                str(r["vendor_stalls"]), str(r["toilets"]), str(r["first_aid"]),
            ])
        gate_lanes = report.total_lanes
        rows.append([
            f'<span style="color:{C_TEXT2}">🚪 Entrance gates</span>',
            f'{w_n_gates.value} gates',
            f'{gate_lanes} (gate staff)',
            "—", "—", "—",
        ])
        rows.append([
            f'<b style="color:{C_PRIMARY}">TOTAL</b>',
            f'<b>{totals["attendance"]:,}</b>',
            f'<b>{totals["staff"] + gate_lanes}</b>',
            f'<b>{totals["vendor_stalls"]}</b>',
            f'<b>{totals["toilets"]}</b>',
            f'<b>{totals["first_aid"]}</b>',
        ])
        pane_res.object = _table_html(
            rows, ["Zone", "Attendance", "Staff (incl. gates)", "Food Stalls", "Toilets", "First Aid"])

        # ── Feasibility detail ─────────────────────────────────────
        fhtml = ""
        for w, icon in [("clear", "☀"), ("rain", "🌧"), ("heat", "🔥")]:
            amax = report.a_max_by_weather[w]
            eff  = int(ticket_sales * (1 - noshow[w]))
            ok   = eff <= amax
            bg   = C_GREEN_L if ok else C_RED_L
            tc   = C_GREEN   if ok else C_RED
            pct  = min(eff / amax, 1.4) * 100 if amax > 0 else 0
            bar_w = min(pct, 100)
            bar_c = C_GREEN if ok else C_RED
            fhtml += (
                f'<div style="padding:12px 14px;border-radius:8px;background:{bg};margin-bottom:8px">'
                f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px">'
                f'<span style="font-size:13px;font-weight:600;color:{C_TEXT}">{icon} {w.capitalize()}</span>'
                f'<span style="font-size:11px;font-weight:700;color:{tc}">{"✓ FEASIBLE" if ok else "✗ OVER CAPACITY"}</span>'
                f'</div>'
                f'<div style="background:rgba(0,0,0,0.08);border-radius:4px;height:6px;margin-bottom:6px">'
                f'<div style="width:{bar_w:.0f}%;height:100%;background:{bar_c};border-radius:4px"></div>'
                f'</div>'
                f'<div style="font-size:11px;color:{C_TEXT2}">'
                f'Expected {eff:,} people &nbsp;·&nbsp; Safe limit {amax:,.0f}'
                f'&nbsp;·&nbsp; {pct:.0f}% utilisation</div>'
                f'</div>'
            )
        pane_pfeas.object = fhtml

        # ── Cost detail ────────────────────────────────────────────
        bc         = report.baseline_cost
        gate_cost  = report.gate_cost
        zone_staff_cost = bc - gate_cost
        chtml = (
            f'<div style="padding:10px 0;border-bottom:1px solid {C_BORDER}">'
            f'<div style="font-size:11px;color:{C_TEXT3};margin-bottom:3px">'
            f'Zone operations (staff, stalls, toilets, first aid, overhead)</div>'
            f'<div style="font-size:16px;font-weight:600;color:{C_TEXT}">'
            f'€{zone_staff_cost:,.0f}<span style="font-size:11px;font-weight:400;color:{C_TEXT3}"> / day</span>'
            f'</div></div>'
            f'<div style="padding:10px 0;border-bottom:1px solid {C_BORDER}">'
            f'<div style="font-size:11px;color:{C_TEXT3};margin-bottom:3px">'
            f'Gate infrastructure ({report.total_lanes} lanes × lane hire + 1 staff/lane)</div>'
            f'<div style="font-size:16px;font-weight:600;color:{C_TEXT}">'
            f'€{gate_cost:,.0f}<span style="font-size:11px;font-weight:400;color:{C_TEXT3}"> / day</span>'
            f'</div></div>'
        )
        for lbl, val, col in [("Total — clear weather (baseline)", bc, C_TEXT),
                               ("Total — rain (+15% ops cost)", bc * 1.15, "#1D4ED8"),
                               ("Total — heat (+40% ops cost)", bc * 1.40, C_RED)]:
            chtml += (
                f'<div style="padding:10px 0;border-bottom:1px solid {C_BORDER}">'
                f'<div style="font-size:11px;color:{C_TEXT3};margin-bottom:3px">{lbl}</div>'
                f'<div style="font-size:20px;font-weight:700;color:{col}">'
                f'€{val:,.0f}<span style="font-size:11px;font-weight:400;color:{C_TEXT3}"> / day</span>'
                f'</div></div>'
            )
        if budget:
            # Check every weather scenario — heat at +40% is the binding constraint
            scenarios_vs_budget = [
                ("Clear",  bc,        bc > budget),
                ("Rain",   bc * 1.15, bc * 1.15 > budget),
                ("Heat",   bc * 1.40, bc * 1.40 > budget),
            ]
            any_over = any(over for _, _, over in scenarios_vs_budget)
            all_over = all(over for _, _, over in scenarios_vs_budget)
            if all_over:
                summary_col, summary_bg = C_RED, C_RED_L
                summary_txt = f"✗ Over budget in all weather conditions"
            elif any_over:
                summary_col, summary_bg = C_YELLOW, C_YELLOW_L
                summary_txt = f"⚠ Within budget for clear weather but over in adverse conditions"
            else:
                summary_col, summary_bg = C_GREEN, C_GREEN_L
                summary_txt = f"✓ Within budget across all weather conditions"
            chtml += f'<div style="margin-top:12px;padding:10px 14px;border-radius:8px;background:{summary_bg}">'
            chtml += f'<div style="font-size:11px;color:{C_TEXT3};margin-bottom:6px">Budget: €{budget:,.0f} / day</div>'
            chtml += f'<div style="font-size:13px;font-weight:600;color:{summary_col};margin-bottom:8px">{summary_txt}</div>'
            for wname, wcost, wover in scenarios_vs_budget:
                wc = C_RED if wover else C_GREEN
                chtml += (
                    f'<div style="display:flex;justify-content:space-between;'
                    f'font-size:12px;padding:3px 0;border-top:1px solid {C_BORDER}">'
                    f'<span style="color:{C_TEXT2}">{wname}</span>'
                    f'<span style="color:{wc};font-weight:600">'
                    f'{"✗" if wover else "✓"} €{wcost:,.0f}</span></div>'
                )
            chtml += '</div>'
        pane_cost.object = chtml

        # ── Bayesian capacity rules ────────────────────────────────
        rain_cap = int(min(report.a_max_by_weather["rain"], ticket_sales))
        bhtml = ""
        for cond, action, col, bg in [
            ("☀  Clear forecast  (P(rain) < 20%)",
             f"Operate at full capacity: {ticket_sales:,} attendees", C_GREEN, C_GREEN_L),
            ("🌧  Rain forecast  (P(rain) > 85%)",
             f"Reduce admission to {rain_cap:,} (rain egress limit)", C_YELLOW, C_YELLOW_L),
            ("⛈  Strong rain signal  (P > 95%)",
             "Extend evacuation time or cancel — safety overrides revenue", C_RED, C_RED_L),
        ]:
            bhtml += (
                f'<div style="padding:10px 14px;border-radius:8px;background:{bg};margin-bottom:8px">'
                f'<div style="font-size:11px;color:{C_TEXT2};margin-bottom:3px">{cond}</div>'
                f'<div style="font-size:13px;font-weight:600;color:{col}">{action}</div>'
                f'</div>'
            )
        pane_bayes.object = bhtml

        run_status.object = f'<span style="color:{C_GREEN};font-weight:600">✓ Complete</span>'

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        run_status.object = f'<span style="color:{C_RED}">✗ {e}</span>'
    finally:
        _running = False
        run_spinner.value = False


# ── Gate/lane sweep ───────────────────────────────────────────────────────────
def _find_optimal_lanes(scenario, zones, cost_params, budget, seed=2026, n_runs=15):
    """
    Sweep total lane counts from a low bound to a ceiling.
    For each count we run a small MC and record the fraction of steps with
    entrance surge active. Returns a dict with the recommended layout and
    the full sweep table for the chart.
    """
    from main import monte_carlo as _mc, G_BASE_PER_MIN
    attendance = scenario.a_total
    # Throughput ceiling: all arrivals processed in gate_open→headliner window (≈4h=240min)
    min_lanes = max(4, int(attendance / (G_BASE_PER_MIN * 240)))
    max_lanes = min_lanes * 5
    step_size = max(1, (max_lanes - min_lanes) // 15)

    sweep = []
    for total in range(min_lanes, max_lanes + 1, step_size):
        # Distribute evenly across two gates
        el = {"main": (total + 1) // 2, "secondary": total // 2}
        df = _mc(scenario, n_runs=n_runs, seed=seed, zones=zones,
                 cost_params=cost_params, total_lanes=total, entrance_lanes=el)
        # surge_activations = number of 15-min steps in that run where any gate was surging
        # use mean surge steps as the primary signal; normalise to fraction of gate-open horizon
        gate_open_steps = max(1, scenario.horizon_steps // scenario.festival_type.n_days)
        surge_rate = (df["surge_activations"] / gate_open_steps).mean()
        cost_per_day = total * (cost_params.omega_lane + cost_params.omega_gate_staff)
        sweep.append({"total_lanes": total, "surge_rate": surge_rate,
                       "gate_cost": cost_per_day, "el": el})

    # Gate budget envelope: at most 25% of total daily budget for infrastructure
    gate_budget = (budget * 0.25) if budget else None

    # Tier 1: under surge threshold AND within budget — ideal
    tier1 = [s for s in sweep if s["surge_rate"] < 0.10 and
             (gate_budget is None or s["gate_cost"] <= gate_budget)]
    # Tier 2: within budget but surge rate may be higher — budget-constrained fallback
    tier2 = [s for s in sweep if gate_budget is None or s["gate_cost"] <= gate_budget]
    # Tier 3: ignore budget entirely, just find minimum surge — used when nothing fits
    tier3 = [s for s in sweep if s["surge_rate"] < 0.10]

    if tier1:
        recommended = tier1[0]           # cheapest option that is both safe and affordable
        budget_status = "ok"
    elif tier2:
        # Budget too tight to fully clear surge — pick lowest surge within budget
        recommended = min(tier2, key=lambda s: s["surge_rate"])
        budget_status = "constrained"    # surge not fully cleared but within budget
    elif tier3:
        recommended = tier3[0]           # over budget but operationally required
        budget_status = "over"
    else:
        recommended = sweep[-1]          # nothing good — use maximum tested
        budget_status = "over"

    recommended["budget_status"] = budget_status
    recommended["gate_budget"]   = gate_budget
    return recommended, sweep


# ── Optimiser ─────────────────────────────────────────────────────────────────
def _run_optimise():
    global _running
    _running = True
    run_spinner.value = True
    run_status.object = "Optimising…"

    try:
        r = _resolve_inputs()
        zones_data   = r["zones_data"]
        ticket_sales = r["tickets"]
        n_days       = r["n_days"]
        camping_frac = r["camping"]
        gate_hour    = r["gate_hour"]
        headliner_h  = r["headliner_h"]
        budget       = r["budget"] or 1e9
        n_runs       = max(10, r["n_runs"] // 2)
        name         = r["name"]
        total_area   = sum(z["area"] for z in zones_data)

        from main import CostParams, make_scenario as _make_scenario
        cost_params = CostParams(**r["cost_params_kw"])

        # ── Step 1: Gate/lane sweep ────────────────────────────────
        run_status.object = "Sizing gate infrastructure…"
        ft_sweep = FestivalType(n_days=n_days, is_camping=camping_frac > 0,
                                camping_fraction=camping_frac, sold_out_fraction=1.0)
        zones_sweep = _build_zones(zones_data, ticket_sales, r["t_evac"], r["camping_zone"])
        scenario_sweep = _make_scenario(
            name=name, a_total=ticket_sales, t_evac_min=r["t_evac"],
            ticket_price=r["price"], festival_type=ft_sweep,
            n_stages=1, staggered_end_times=False,
            gate_open_hour=gate_hour, headliner_start_hour=headliner_h,
            seed=2026,
        )
        recommended_gates, gate_sweep = _find_optimal_lanes(
            scenario_sweep, zones_sweep, cost_params,
            budget=r["budget"], seed=2026, n_runs=12,
        )
        entrance_lanes = recommended_gates["el"]
        total_lanes    = recommended_gates["total_lanes"]

        # ── Gate sweep chart ───────────────────────────────────────
        xs = [s["total_lanes"]  for s in gate_sweep]
        ys = [s["surge_rate"] * 100 for s in gate_sweep]
        cs = [s["gate_cost"]    for s in gate_sweep]
        rec_x = recommended_gates["total_lanes"]
        rec_y = recommended_gates["surge_rate"] * 100

        fig_gate = go.Figure()
        fig_gate.add_trace(go.Scatter(
            x=xs, y=ys,
            mode="lines+markers",
            line=dict(color=C_PRIMARY, width=2),
            marker=dict(size=7, color=cs, colorscale="Blues",
                        showscale=True,
                        colorbar=dict(title="Gate cost €/day", thickness=12, len=0.7)),
            name="Surge rate",
            hovertemplate="<b>%{x} lanes</b><br>Surge: %{y:.1f}%<br>Gate cost: €%{marker.color:,.0f}<extra></extra>",
        ))
        fig_gate.add_shape(type="line", x0=min(xs), x1=max(xs), y0=10, y1=10,
                           line=dict(color=C_YELLOW, width=1.5, dash="dash"))
        fig_gate.add_annotation(x=rec_x, y=rec_y, text="✓ Recommended",
                                 showarrow=True, arrowhead=2, arrowcolor=C_GREEN,
                                 font=dict(size=11, color=C_GREEN), bgcolor="white",
                                 bordercolor=C_GREEN, borderwidth=1)
        fig_gate.update_layout(**base_layout(260))
        fig_gate.update_layout(
            xaxis_title="Total lanes", yaxis_title="Surge rate (%)",
            yaxis=dict(rangemode="tozero"),
            shapes=[dict(type="rect", x0=min(xs), x1=max(xs), y0=0, y1=10,
                         fillcolor=C_GREEN_L, opacity=0.35, layer="below", line_width=0)],
        )
        pane_opt_gate_chart.object = fig_gate

        # ── Gate recommendation panel ─────────────────────────────
        n_gates_rec   = len(entrance_lanes)
        lanes_each    = list(entrance_lanes.values())
        lanes_desc    = " + ".join(f"{v} lanes" for v in lanes_each)
        gate_cost_d   = recommended_gates["gate_cost"]
        surge_pct     = recommended_gates["surge_rate"] * 100
        bstatus       = recommended_gates["budget_status"]
        gate_budget   = recommended_gates["gate_budget"]

        scol = C_GREEN if surge_pct < 5 else C_YELLOW if surge_pct < 15 else C_RED
        surge_bg = C_GREEN_L if surge_pct < 5 else C_YELLOW_L if surge_pct < 15 else C_RED_L

        if bstatus == "ok":
            bstatus_html = (
                f'<div style="padding:8px 12px;background:{C_GREEN_L};'
                f'border-radius:6px;margin-bottom:8px">'
                f'<span style="font-size:11px;color:{C_TEXT3}">Budget fit</span><br>'
                f'<span style="font-size:13px;font-weight:600;color:{C_GREEN}">'
                f'✓ Within gate budget (≤€{gate_budget:,.0f}/day)</span></div>'
            )
        elif bstatus == "constrained":
            min_safe_cost = next((s["gate_cost"] for s in gate_sweep if s["surge_rate"] < 0.10),
                                  gate_sweep[-1]["gate_cost"])
            bstatus_html = (
                f'<div style="padding:8px 12px;background:{C_YELLOW_L};'
                f'border-radius:6px;margin-bottom:8px">'
                f'<span style="font-size:11px;color:{C_TEXT3}">Budget constraint</span><br>'
                f'<span style="font-size:13px;font-weight:600;color:{C_YELLOW}">'
                f'⚠ Budget too tight to fully clear surge</span><br>'
                f'<span style="font-size:11px;color:{C_TEXT2}">'
                f'Minimum to clear surge would cost €{min_safe_cost:,.0f}/day — '
                f'increase budget or reduce attendance</span></div>'
            )
        else:
            bstatus_html = (
                f'<div style="padding:8px 12px;background:{C_RED_L};'
                f'border-radius:6px;margin-bottom:8px">'
                f'<span style="font-size:11px;color:{C_TEXT3}">Budget constraint</span><br>'
                f'<span style="font-size:13px;font-weight:600;color:{C_RED}">'
                f'✗ Gate infrastructure exceeds budget</span><br>'
                f'<span style="font-size:11px;color:{C_TEXT2}">'
                f'€{gate_cost_d:,.0f}/day required — '
                f'this is the operational minimum for safe entry flow</span></div>'
            )

        pane_opt_gates.object = (
            f'<div style="font-size:13px;color:{C_TEXT};line-height:1.8">'
            f'<div style="margin-bottom:10px">'
            f'<span style="font-size:22px;font-weight:700;color:{C_PRIMARY}">'
            f'{total_lanes} lanes</span>'
            f'<span style="font-size:12px;color:{C_TEXT3};margin-left:8px">'
            f'across {n_gates_rec} gate(s) ({lanes_desc})</span></div>'
            f'<div style="padding:8px 12px;background:{surge_bg};border-radius:6px;margin-bottom:8px">'
            f'<span style="font-size:11px;color:{C_TEXT3}">Entrance surge rate</span><br>'
            f'<span style="font-size:16px;font-weight:600;color:{scol}">{surge_pct:.1f}%</span>'
            f'<span style="font-size:11px;color:{C_TEXT3}"> of gate-open steps</span></div>'
            f'{bstatus_html}'
            f'<div style="padding:8px 12px;background:{C_BG};border-radius:6px;margin-bottom:8px">'
            f'<span style="font-size:11px;color:{C_TEXT3}">Gate infrastructure cost</span><br>'
            f'<span style="font-size:16px;font-weight:600;color:{C_TEXT}">€{gate_cost_d:,.0f}</span>'
            f'<span style="font-size:11px;color:{C_TEXT3}"> / day (lanes + 1 staff/lane)</span></div>'
            f'<div style="font-size:11px;color:{C_TEXT3};margin-top:6px">'
            f'Throughput: {total_lanes} lanes × 400 scans/hr = '
            f'<b>{total_lanes * 400:,} scans/hr</b> peak capacity</div>'
            f'</div>'
        )

        # Search grid
        t_evac_options  = [6, 8, 10, 12, 15]
        stage_options   = [1, 2, 3]
        stagger_options = [False, True]
        split_options   = [
            {"label": "Concentrated (60/25/15%)", "splits": [0.60, 0.25, 0.15]},
            {"label": "Balanced (45/35/20%)",      "splits": [0.45, 0.35, 0.20]},
            {"label": "Distributed (40/35/25%)",   "splits": [0.40, 0.35, 0.25]},
        ]
        n_zones = min(len(zones_data), 3)

        configs = list(itertools.product(t_evac_options, stage_options,
                                         stagger_options, split_options))
        total   = len(configs)
        results = []

        for idx, (tevac, nstage, stagger, split_def) in enumerate(configs):
            run_status.object = f"Testing config {idx+1}/{total}…"

            splits = split_def["splits"][:n_zones]
            splits = [s / sum(splits) for s in splits]  # normalise

            zone_areas = {zones_data[i]["name"]: total_area * splits[i]
                          for i in range(n_zones)}

            plan = FestivalPlan(
                name=name,
                zone_areas=zone_areas,
                entrance_lanes=entrance_lanes,
                ticket_sales=ticket_sales, ticket_price=r["price"],
                n_days=n_days, t_evac_min=tevac,
                is_camping=camping_frac > 0, camping_fraction=camping_frac,
                sold_out_fraction=1.0,
                multiday_ticket_fraction=r["multiday_ticket_frac"],
                total_budget=float(budget),
                n_stages=nstage, staggered_end_times=stagger,
                gate_open_hour=gate_hour, headliner_start_hour=headliner_h,
                n_runs=n_runs, seed=2026,
            )
            try:
                report = generate_plan(plan, cost_params=cost_params)
                s      = report.summary
                bc     = report.baseline_cost
                cost_rain = bc * 1.15
                cost_heat = bc * 1.40
                # Budget filter: reject if ANY weather scenario exceeds budget
                worst_cost = cost_heat if budget else 0
                if budget and worst_cost > budget:
                    continue
                composite = s["E_uO"] * s["frac_feasible"]
                # Capacity feasibility per weather
                noshow = report.noshow_rates
                cap_ok = {w: int(ticket_sales * (1 - noshow[w])) <= report.a_max_by_weather[w]
                          for w in ("clear", "rain", "heat")}
                results.append({
                    "t_evac":      tevac,
                    "n_stages":    nstage,
                    "stagger":     stagger,
                    "split":       split_def["label"],
                    "E_uO":        s["E_uO"],
                    "feasible":    s["frac_feasible"],
                    "density":     s["mean_D"],
                    "cost":        bc,
                    "cost_rain":   cost_rain,
                    "cost_heat":   cost_heat,
                    "cap_ok":      cap_ok,
                    "composite":   composite,
                    "report":      report,
                })
            except Exception:
                continue

        if not results:
            pane_opt_summary.object = (
                f'<div style="padding:14px;background:{C_RED_L};border-radius:8px;color:{C_RED}">'
                f'No feasible configurations found within budget. '
                f'Try increasing the budget or reducing ticket count.</div>'
            )
            run_status.object = f'<span style="color:{C_YELLOW}">No feasible configs</span>'
            return

        results.sort(key=lambda x: x["composite"], reverse=True)
        top5   = results[:5]
        best   = top5[0]

        # ── Scatter chart ──────────────────────────────────────────
        x_all  = [r["feasible"] * 100 for r in results]
        y_all  = [r["E_uO"]           for r in results]
        c_all  = [r["composite"]       for r in results]
        x_top  = [r["feasible"] * 100  for r in top5]
        y_top  = [r["E_uO"]            for r in top5]

        fig_opt = go.Figure()
        fig_opt.add_trace(go.Scatter(
            x=x_all, y=y_all, mode="markers", name="All configs",
            marker=dict(color=c_all, colorscale="Blues", size=7,
                        colorbar=dict(title="Score×Safety", thickness=12),
                        opacity=0.65),
            hovertemplate="Safe runs: %{x:.0f}%<br>Operator Score: %{y:,.0f}<extra></extra>",
        ))
        fig_opt.add_trace(go.Scatter(
            x=x_top, y=y_top, mode="markers+text", name="Top 5",
            text=[f"#{i+1}" for i in range(len(top5))],
            textposition="top center",
            textfont=dict(color=C_PRIMARY, size=10, family="Inter"),
            marker=dict(color=C_PRIMARY, size=13, symbol="star",
                        line=dict(color="white", width=1.5)),
        ))
        fig_opt.update_layout(**base_layout(340))
        fig_opt.update_xaxes(title_text="% of runs with zero safety violations",
                              range=[-2, 105])
        fig_opt.update_yaxes(title_text="Operator Score (higher = more profitable)")
        pane_opt_chart.object = fig_opt

        # ── Weather cost comparison chart (top 5) ─────────────────
        cfg_labels = [
            f"#{i+1} {r['t_evac']}min · {r['n_stages']}stage · {r['split'].split('(')[0].strip()}"
            for i, r in enumerate(top5)
        ]
        fig_wx = go.Figure()
        wx_colors = {"☀ Clear": "#93C5FD", "🌧 Rain": "#60A5FA", "🔥 Heat": "#EF4444"}
        for wlabel, wkey, wmult in [("☀ Clear", "cost", 1.0),
                                     ("🌧 Rain",  "cost_rain", 1.15),
                                     ("🔥 Heat",  "cost_heat", 1.40)]:
            ys = [r[wkey] for r in top5]
            fig_wx.add_trace(go.Bar(
                name=wlabel, x=cfg_labels, y=ys,
                marker_color=wx_colors[wlabel],
                text=[f"€{v:,.0f}" for v in ys],
                textposition="outside",
                textfont=dict(size=9),
            ))
        if budget:
            fig_wx.add_hline(y=budget, line_dash="dash", line_color=C_RED, line_width=2,
                             annotation_text=f"Budget €{budget:,.0f}",
                             annotation_font=dict(color=C_RED, size=10),
                             annotation_position="top right")
        fig_wx.update_layout(**base_layout(300))
        fig_wx.update_layout(
            barmode="group",
            legend=dict(orientation="h", y=-0.28, x=0),
            xaxis=dict(tickangle=-15, tickfont=dict(size=9)),
            yaxis_title="Daily cost (€)",
        )
        pane_opt_weather.object = fig_wx

        # ── Best config summary ────────────────────────────────────
        stagger_str = "staggered set times" if best["stagger"] else "simultaneous close"
        pane_opt_summary.object = (
            f'<div style="background:{C_BLUE_L};border-radius:8px;padding:16px 18px">'
            f'<div style="font-size:14px;font-weight:700;color:#1E40AF;margin-bottom:12px">'
            f'🏆  Recommended Configuration</div>'
            f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:12px">'
            f'<div><div style="font-size:10px;color:#3B82F6;text-transform:uppercase;'
            f'letter-spacing:0.08em;margin-bottom:3px">Evacuation Time</div>'
            f'<div style="font-size:18px;font-weight:700;color:#1E3A8A">{best["t_evac"]} min</div></div>'
            f'<div><div style="font-size:10px;color:#3B82F6;text-transform:uppercase;'
            f'letter-spacing:0.08em;margin-bottom:3px">Stages</div>'
            f'<div style="font-size:18px;font-weight:700;color:#1E3A8A">{best["n_stages"]}</div></div>'
            f'<div><div style="font-size:10px;color:#3B82F6;text-transform:uppercase;'
            f'letter-spacing:0.08em;margin-bottom:3px">Zone Layout</div>'
            f'<div style="font-size:13px;font-weight:600;color:#1E3A8A">{best["split"]}</div></div>'
            f'<div><div style="font-size:10px;color:#3B82F6;text-transform:uppercase;'
            f'letter-spacing:0.08em;margin-bottom:3px">Set Times</div>'
            f'<div style="font-size:13px;font-weight:600;color:#1E3A8A">{stagger_str.capitalize()}</div></div>'
            f'</div>'
            f'<div style="margin-top:14px;padding-top:12px;border-top:1px solid #BFDBFE;'
            f'display:flex;gap:24px">'
            f'<div><span style="font-size:11px;color:#3B82F6">Safe runs</span>'
            f'<div style="font-size:16px;font-weight:700;color:{C_GREEN}">'
            f'{best["feasible"]:.0%}</div></div>'
            f'<div><span style="font-size:11px;color:#3B82F6">Operator score</span>'
            f'<div style="font-size:16px;font-weight:700;color:#1E3A8A">'
            f'{best["E_uO"]:,.0f}</div></div>'
            f'<div><span style="font-size:11px;color:#3B82F6">Avg density</span>'
            f'<div style="font-size:16px;font-weight:700;color:#1E3A8A">'
            f'{best["density"]:.3f} p/m²</div></div>'
            f'</div>'
            f'<div style="margin-top:12px;padding-top:12px;border-top:1px solid #BFDBFE">'
            f'<div style="font-size:10px;color:#3B82F6;text-transform:uppercase;'
            f'letter-spacing:0.08em;margin-bottom:8px">Cost by weather scenario</div>'
            f'<div style="display:flex;gap:12px;flex-wrap:wrap">'
            + "".join(
                f'<div style="flex:1;min-width:80px;padding:8px 10px;background:{"#DBEAFE" if wk=="cost" else "#EFF6FF" if wk=="cost_rain" else C_RED_L};border-radius:6px">'
                f'<div style="font-size:10px;color:{"#1D4ED8" if wk!="cost_heat" else C_RED};margin-bottom:2px">{wl}</div>'
                f'<div style="font-size:14px;font-weight:700;color:{"#1E3A8A" if wk!="cost_heat" else C_RED}">'
                f'€{best[wk]:,.0f}</div>'
                f'{"" if not budget else f"""<div style="font-size:10px;color:{C_GREEN if best[wk]<=budget else C_RED}">{"✓" if best[wk]<=budget else "✗"} budget</div>"""}'
                f'</div>'
                for wl, wk in [("☀ Clear", "cost"), ("🌧 Rain", "cost_rain"), ("🔥 Heat", "cost_heat")]
            ) +
            f'</div></div></div>'
        )

        # ── Top 5 table ────────────────────────────────────────────
        rows = []
        for i, r in enumerate(top5):
            rank_icon = ["🥇", "🥈", "🥉", "4th", "5th"][i]
            safe_col  = C_GREEN if r["feasible"] >= 0.8 else C_YELLOW

            def _cost_cell(val):
                if not budget:
                    return f'€{val:,.0f}'
                ok = val <= budget
                c  = C_GREEN if ok else C_RED
                return f'<span style="color:{c};font-weight:600">{"✓" if ok else "✗"} €{val:,.0f}</span>'

            def _cap_cell(wk):
                ok = r["cap_ok"][wk]
                c  = C_GREEN if ok else C_RED
                return f'<span style="color:{c}">{"✓" if ok else "✗"}</span>'

            rows.append([
                f'<b>{rank_icon}</b>',
                f'{r["t_evac"]} min',
                str(r["n_stages"]),
                "Yes" if r["stagger"] else "No",
                r["split"],
                f'<span style="color:{safe_col};font-weight:600">{r["feasible"]:.0%}</span>',
                f'{r["E_uO"]:,.0f}',
                _cost_cell(r["cost"]),
                _cost_cell(r["cost_rain"]),
                _cost_cell(r["cost_heat"]),
                _cap_cell("clear"),
                _cap_cell("rain"),
                _cap_cell("heat"),
                f'{r["density"]:.3f}',
            ])
        pane_opt_detail.object = _table_html(
            rows,
            ["Rank", "Evac.", "Stages", "Stagger", "Zone Layout",
             "Safe Runs", "Score",
             "Cost ☀", "Cost 🌧", "Cost 🔥",
             "Cap ☀", "Cap 🌧", "Cap 🔥",
             "Density"],
        )

        run_status.object = (
            f'<span style="color:{C_GREEN};font-weight:600">'
            f'✓ Tested {len(results)} configs</span>'
        )

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        run_status.object = f'<span style="color:{C_RED}">✗ {e}</span>'
    finally:
        _running = False
        run_spinner.value = False


def _on_run(_):
    if not _running:
        threading.Thread(target=_run_simulation, daemon=True).start()


def _on_optimise(_):
    if not _running:
        threading.Thread(target=_run_optimise, daemon=True).start()


btn_run.on_click(_on_run)
btn_optimise.on_click(_on_optimise)

if __name__ == "__main__":
    pn.serve(layout, port=5006, show=True, title="FestivalSim")
