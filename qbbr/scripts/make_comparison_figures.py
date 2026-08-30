"""Bar/point/line comparison figures for every remaining result set that has
NOT yet been turned into a figure, complementing make_cdf_figures.py (which
covers the datasets that still have raw per-seed/per-grid-point arrays on
disk). Everything here only has an already-aggregated median (and
sometimes IQR or p-value) per location -- not a raw distribution -- so a
bar/point/line chart is the honest representation, not a CDF.

Covers:
1. RQ1a/RQ1b headline (rq1_report_final_pacingonly_classical.json,
   rq1_report_final_multihead_classical.json) -- retransmit reduction %,
   pacing_gain-only vs. the extended action space, both under final
   dynamics (matches Table rq1b-extension in main.tex).
2. RQ2 (rq2_report_extended.json + rq2_extended_action_space_report.json)
   -- alpha-fair efficiency ratio, qbbr vs. stock, across the full alpha
   sweep, per location, with the extended-action-space checkpoint overlaid
   as a robustness check.
3. RQ3 (rq3_report.json) -- risk_on vs. risk_off, retransmit rate and
   throughput, per location.
4. RQ4 (quantum_vs_classical_final.json) -- classical vs. quantum retransmit
   reduction (median, quantum with IQR), per location.
5. Shifted-freeze decoy-phase control (shifted_freeze_report.json) -- real
   vs. pooled-decoy phase retransmit rate, per location.
6. ECN/exploit behavior-shift v2 (ecn_behavior_shift_report.json) --
   mean inflight level minus each arm's own default, exploit vs. fixed,
   per location (visualizes the finding added to main.tex sec:rq1b-extensions).
7. Freeze-window retransmit attribution (freeze_window_retransmit_attribution
   .json) -- share of retransmits vs. share of time inside the freeze
   window, per location (the "concentration_ratio" claim).
8. Reward decomposition under sustained throttle
   (reward_decomposition_sustained_throttle.json) -- utility / delay_term /
   beta*l_t, stock vs. throttle, per location (the reward "blind spot"
   argument in sec:rq1b-reward-episode).
9. Anticipatory lead-time sweep vs. decoy-lag control
   (anticipatory_throttle_test.json + decoy_lag_throttle_test.json) --
   retransmit reduction vs. stock as a function of lead/lag time, per
   location (predecessor sanity check to the shifted-freeze design).
10. RQ1 raw boxplot (rq1_raw_for_boxplot.json, produced by
    eval_rq1_raw_for_boxplot.py) -- stock BBR-v3 vs. qbbr (pacing_gain-only
    and extended), Throughput/RTT/Retransmissions, boxplotted from the real
    per-seed (qbbr) / per-trace-file (BBR-v3) raw values -- the direct
    analogue of a reference boxplot-grid figure, restricted to the three
    metrics this project's fluid simulator actually tracks (it does not
    model congestion window or receiver-advertised window).

Output -> figures/*.png (same consolidated directory as make_cdf_figures.py
and every other paper figure).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # .../Codebase
OUT_DIR = PROJECT_ROOT / "outputs"
FIG_DIR = PROJECT_ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

LOCATIONS = ["London", "Mumbai", "Ohio", "SaoPaulo", "Sydney", "Tokyo"]
C1, C2, C3 = "#1f77b4", "#ff7f0e", "#2ca02c"


def _save(fig, name, title, top=0.90):
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, top))
    out = FIG_DIR / name
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved -> {out.relative_to(PROJECT_ROOT)}")


def _pstr(p):
    return f"p={p:.3f}" if p >= 0.001 else "p<0.001"


# ---------------------------------------------------------------- 1. RQ1a/b
def fig_rq1_headline():
    pacing = {r["location"]: r for r in json.load(open(OUT_DIR / "rq1_report_final_pacingonly_classical.json"))["rq1"]}
    ext = {r["location"]: r for r in json.load(open(OUT_DIR / "rq1_report_final_multihead_classical.json"))["rq1"]}
    x = np.arange(len(LOCATIONS))
    w = 0.35
    fig, ax = plt.subplots(figsize=(11, 5))
    v1 = [pacing[loc]["retransmit_reduction"] * 100 for loc in LOCATIONS]
    v2 = [ext[loc]["retransmit_reduction"] * 100 for loc in LOCATIONS]
    b1 = ax.bar(x - w / 2, v1, w, label="pacing_gain only", color=C1)
    b2 = ax.bar(x + w / 2, v2, w, label="extended (inflight_hi/lo + freeze + EMA)", color=C2)
    for i, loc in enumerate(LOCATIONS):
        ax.text(x[i] - w / 2, v1[i] + 1, _pstr(pacing[loc]["p_retransmits"]), ha="center", fontsize=7, rotation=90)
        ax.text(x[i] + w / 2, v2[i] + 1, _pstr(ext[loc]["p_retransmits"]), ha="center", fontsize=7, rotation=90)
    ax.axhline(20, color="gray", linestyle=":", linewidth=1.3, label="RQ1a pre-registered threshold (20%)")
    ax.set_xticks(x)
    ax.set_xticklabels(LOCATIONS)
    ax.set_ylabel("retransmit reduction vs. stock BBR-v3 (%)")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    _save(fig, "bar_rq1_headline.png",
          "RQ1a/RQ1b Headline: Retransmit Reduction, pacing_gain-only vs. Extended Action Space (10 seeds/location)")


# ---------------------------------------------------------------- 2. RQ2
def fig_rq2_alpha_sweep():
    main_rq2 = {r["location"]: r for r in json.load(open(OUT_DIR / "rq2_report_extended.json"))["rq2"]}
    ext_rq2 = {r["location"]: r for r in json.load(open(OUT_DIR / "rq2_extended_action_space_report.json"))["rq2_extended"]}
    alpha_labels = ["0.0", "0.25", "0.5", "0.75", "1.0", "inf"]
    alpha_x_labels = ["0", "0.25", "0.5", "0.75", "1", "∞"]
    ext_alpha_labels = ["0.0", "0.5", "1.0", "inf"]
    ext_x_idx = [alpha_labels.index(a) for a in ext_alpha_labels]

    fig, axes = plt.subplots(2, 3, figsize=(15, 7.5))
    axes = axes.flatten()
    for i, loc in enumerate(LOCATIONS):
        ax = axes[i]
        row = main_rq2[loc]
        x = np.arange(len(alpha_labels))
        qbbr_y = [row[f"rho_{a}_qbbr_median"] for a in alpha_labels]
        stock_y = [row[f"rho_{a}_stock_median"] for a in alpha_labels]
        ax.plot(x, qbbr_y, "o-", color=C1, label="qbbr (pacing-only)")
        ax.plot(x, stock_y, "s--", color="gray", label="stock BBR-v3")
        ext_row = ext_rq2[loc]
        ext_y = [ext_row[f"rho_{a}_qbbr_median"] for a in ext_alpha_labels]
        ax.plot(ext_x_idx, ext_y, "^:", color=C2, label="qbbr (extended action space)")
        ax.set_xticks(x)
        ax.set_xticklabels(alpha_x_labels)
        ax.set_xlabel(r"$\alpha$")
        ax.set_ylabel(r"$\rho_\alpha$ (efficiency ratio)")
        ax.set_title(loc)
        ax.grid(alpha=0.3)
        if i == 0:
            ax.legend(fontsize=8)
    _save(fig, "line_rq2_alpha_sweep.png",
          r"RQ2: $\alpha$-Fair Efficiency Ratio vs. $\alpha$, qbbr vs. Stock (+ extended-action-space robustness check)",
          top=0.92)


# ---------------------------------------------------------------- 3. RQ3
def fig_rq3_risk_toggle():
    rows = {r["location"]: r for r in json.load(open(OUT_DIR / "rq3_report.json"))["rq3"]}
    x = np.arange(len(LOCATIONS))
    w = 0.35
    fig, axs = plt.subplots(1, 2, figsize=(13, 5))

    ax = axs[0]
    on = [rows[loc]["risk_on_retransmits_per_s_median"] for loc in LOCATIONS]
    off = [rows[loc]["risk_off_retransmits_per_s_median"] for loc in LOCATIONS]
    ax.bar(x - w / 2, off, w, label="risk-state off", color=C1)
    ax.bar(x + w / 2, on, w, label="risk-state on", color=C2)
    for i, loc in enumerate(LOCATIONS):
        ax.text(x[i], max(on[i], off[i]) * 1.03, _pstr(rows[loc]["p_retransmits"]), ha="center", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(LOCATIONS, rotation=15)
    ax.set_ylabel("retransmits / s")
    ax.set_yscale("log")
    ax.set_title("Retransmit rate")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    ax = axs[1]
    on_t = [rows[loc]["risk_on_throughput_mbps_median"] for loc in LOCATIONS]
    off_t = [rows[loc]["risk_off_throughput_mbps_median"] for loc in LOCATIONS]
    ax.bar(x - w / 2, off_t, w, label="risk-state off", color=C1)
    ax.bar(x + w / 2, on_t, w, label="risk-state on", color=C2)
    for i, loc in enumerate(LOCATIONS):
        ax.text(x[i], max(on_t[i], off_t[i]) * 1.02, _pstr(rows[loc]["p_throughput"]), ha="center", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(LOCATIONS, rotation=15)
    ax.set_ylabel("throughput (Mbps)")
    ax.set_title("Throughput")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    _save(fig, "bar_rq3_risk_toggle.png",
          "RQ3: Risk-Predictive State On vs. Off (10 seeds/location)", top=0.90)


# ---------------------------------------------------------------- 4. RQ4
def fig_rq4_quantum_vs_classical():
    d = json.load(open(OUT_DIR / "quantum_vs_classical_final.json"))
    locs = [loc for loc in LOCATIONS if loc in d]
    x = np.arange(len(locs))
    fig, ax = plt.subplots(figsize=(10, 5))
    classical_y = [d[loc]["classical_reduction"] for loc in locs]
    q_y = [d[loc]["q_reduction"] for loc in locs]
    q_lo = [d[loc]["q_reduction"] - d[loc]["q_iqr"][0] for loc in locs]
    q_hi = [d[loc]["q_iqr"][1] - d[loc]["q_reduction"] for loc in locs]
    ax.scatter(x - 0.08, classical_y, marker="o", s=70, color=C1, label="classical", zorder=3)
    ax.errorbar(x + 0.08, q_y, yerr=[q_lo, q_hi], fmt="^", color=C2, markersize=9,
                capsize=4, label="quantum (median, IQR)", zorder=3)
    for i, loc in enumerate(locs):
        n = d[loc]["n_seeds"]
        marker = "" if n >= 10 else f" (n={n})"
        ax.text(x[i], max(classical_y[i], d[loc]["q_iqr"][1]) + 2, f"{_pstr(d[loc]['p'])}{marker}",
                ha="center", fontsize=8)
    ax.axhline(0, color="gray", linestyle=":", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(locs)
    ax.set_ylabel("retransmit reduction vs. stock BBR-v3 (%)")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    _save(fig, "point_rq4_quantum_vs_classical.png",
          "RQ4: Classical vs. Quantum Core, Retransmit Reduction (Sydney: n=3 seeds, all others n=10)")


# ---------------------------------------------------------------- 5. Shifted-freeze
def fig_shifted_freeze():
    rows = {r["location"]: r for r in json.load(open(OUT_DIR / "shifted_freeze_report.json"))["report"]}
    x = np.arange(len(LOCATIONS))
    w = 0.35
    fig, ax = plt.subplots(figsize=(11, 5))
    real_y = [rows[loc]["real_phase_rtx_mean"] for loc in LOCATIONS]
    decoy_y = [rows[loc]["decoy_pooled_rtx_mean"] for loc in LOCATIONS]
    ax.bar(x - w / 2, real_y, w, label="real calibrated phase", color=C1)
    ax.bar(x + w / 2, decoy_y, w, label="pooled decoy phase (5 offsets)", color=C2)
    for i, loc in enumerate(LOCATIONS):
        ax.text(x[i], max(real_y[i], decoy_y[i]) * 1.03,
                f"{rows[loc]['delta']*100:+.1f}%\n{_pstr(rows[loc]['p'])}", ha="center", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(LOCATIONS, rotation=15)
    ax.set_ylabel("retransmits / s")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    _save(fig, "bar_shifted_freeze.png",
          "Shifted-Freeze Decoy-Phase Control: Real Phase vs. Pooled Decoy Phase (10 seeds/location)")


# ---------------------------------------------------------------- 6. ECN behavior-shift
def fig_ecn_behavior_shift():
    d = json.load(open(OUT_DIR / "ecn_behavior_shift_report.json"))
    x = np.arange(len(LOCATIONS))
    w = 0.35
    fig, axs = plt.subplots(1, 2, figsize=(13, 5))
    for ax, head, head_label in [(axs[0], "inflight_hi_mult", "inflight_hi"), (axs[1], "inflight_lo_mult", "inflight_lo")]:
        exploit_y = [d["exploit"][loc]["mean_level_minus_default"][head] for loc in LOCATIONS]
        fixed_y = [d["fixed"][loc]["mean_level_minus_default"][head] for loc in LOCATIONS]
        ax.bar(x - w / 2, exploit_y, w, label="exploit (native symmetric range)", color=C1)
        ax.bar(x + w / 2, fixed_y, w, label="fixed (native expansion-only range)", color=C2)
        ax.axhline(0, color="gray", linestyle=":", linewidth=1.3, label="each arm's own default")
        ax.set_xticks(x)
        ax.set_xticklabels(LOCATIONS, rotation=15)
        ax.set_ylabel("mean level $-$ own default (multiples of $\\bar{B}_{DP}$)")
        ax.set_title(head_label)
        ax.grid(axis="y", alpha=0.3)
    axs[0].legend(fontsize=8)
    _save(fig, "bar_ecn_behavior_shift.png",
          "Exploit vs. Fixed Checkpoints, Each Replayed Under Its Own Native Action Space (5 seeds/location)")


# ---------------------------------------------------------------- 7. Freeze-window attribution
def fig_freeze_attribution():
    d = json.load(open(OUT_DIR / "freeze_window_retransmit_attribution.json"))
    x = np.arange(len(LOCATIONS))
    w = 0.35
    fig, ax = plt.subplots(figsize=(11, 5))
    share_rtx = [d[loc]["frozen_share_of_retransmits"] * 100 for loc in LOCATIONS]
    share_time = [d[loc]["frozen_share_of_time"] * 100 for loc in LOCATIONS]
    ax.bar(x - w / 2, share_time, w, label="share of episode TIME inside freeze window", color=C1)
    ax.bar(x + w / 2, share_rtx, w, label="share of RETRANSMITS inside freeze window", color=C2)
    for i, loc in enumerate(LOCATIONS):
        ax.text(x[i], share_rtx[i] + 1, f"{d[loc]['concentration_ratio']:.2f}x", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(LOCATIONS, rotation=15)
    ax.set_ylabel("% of episode")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    _save(fig, "bar_freeze_window_attribution.png",
          "Retransmit Concentration Inside the Boundary-Freeze Window (label: concentration ratio)")


# ---------------------------------------------------------------- 8. Reward decomposition
def fig_reward_decomposition():
    d = json.load(open(OUT_DIR / "reward_decomposition_sustained_throttle.json"))
    locs = [loc for loc in LOCATIONS if loc in d]
    x = np.arange(len(locs))
    w = 0.35
    fig, axs = plt.subplots(1, 2, figsize=(13, 5))

    ax = axs[0]
    stock_dt = [d[loc]["stock"]["delay_term"] for loc in locs]
    throttle_dt = [d[loc]["throttle"]["delay_term"] for loc in locs]
    ax.bar(x - w / 2, stock_dt, w, label="stock (no throttle)", color=C1)
    ax.bar(x + w / 2, throttle_dt, w, label="sustained throttle", color=C2)
    ax.set_xticks(x)
    ax.set_xticklabels(locs, rotation=15)
    ax.set_ylabel(r"delay term ($-\delta\log(\mathrm{RTT}/\mathrm{RTT}_{\min})$, per step)")
    ax.set_title("Delay term (rewarded every step)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    ax = axs[1]
    stock_bl = [d[loc]["stock"]["beta_l_t"] for loc in locs]
    throttle_bl = [d[loc]["throttle"]["beta_l_t"] for loc in locs]
    ax.bar(x - w / 2, stock_bl, w, label="stock (no throttle)", color=C1)
    ax.bar(x + w / 2, throttle_bl, w, label="sustained throttle", color=C2)
    ax.set_xticks(x)
    ax.set_xticklabels(locs, rotation=15)
    ax.set_ylabel(r"$\beta \cdot l_t$ (retransmit-rate-delta loss, per step)")
    ax.set_title(r"Loss term $\beta l_t$ (delta-only -- cannot see a stable elevated rate)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    _save(fig, "bar_reward_decomposition.png",
          "Reward Decomposition Under Sustained Throttle: Stock vs. Throttle, Delay Term vs. Loss Term", top=0.90)


# ---------------------------------------------------------------- 9. Anticipatory lead vs. decoy lag
def fig_anticipatory_vs_decoy_lag():
    d_lead = json.load(open(OUT_DIR / "anticipatory_throttle_test.json"))
    d_lag = json.load(open(OUT_DIR / "decoy_lag_throttle_test.json"))
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.5))
    axes = axes.flatten()
    for i, loc in enumerate(LOCATIONS):
        ax = axes[i]
        lead_entries = sorted(d_lead[loc].items(), key=lambda kv: float(kv[0]))
        lead_x = [float(k) for k, _ in lead_entries]
        lead_y = [v["retransmit_reduction_vs_stock"] * 100 for _, v in lead_entries]
        ax.plot(lead_x, lead_y, "o-", color=C1, label="anticipatory (lead)")
        lag_entries = sorted(d_lag[loc].items(), key=lambda kv: float(kv[0]))
        lag_x = [float(k) for k, _ in lag_entries]
        lag_y = [v["retransmit_reduction_vs_stock"] * 100 for _, v in lag_entries]
        ax.plot(lag_x, lag_y, "s--", color=C2, label="decoy control (lag)")
        ax.set_xlabel("scripted throttle lead/lag time (s)")
        ax.set_ylabel("retransmit reduction vs. stock (%)")
        ax.set_title(loc)
        ax.grid(alpha=0.3)
        if i == 0:
            ax.legend(fontsize=8)
    _save(fig, "line_anticipatory_vs_decoy_lag.png",
          "Scripted Throttle Sanity Check: Anticipatory Lead-Time Sweep vs. Decoy-Lag Control", top=0.92)


# ---------------------------------------------------------------- 10. RQ1 boxplot (BBR-v3 vs qbbr, raw per-seed)
def fig_rq1_boxplot_vs_bbr():
    """The direct analogue of the reference "Competitive/Dedicated ... over
    Starlink" boxplot grids: real per-seed (qbbr) / per-trace-file (stock
    BBR-v3) raw values, boxplotted per location, for every metric this
    project's fluid simulator actually tracks (throughput, RTT, retransmit
    rate -- it does not model congestion window or receiver-advertised
    window, so this grid has 3 panels, not 6). Source: outputs/
    rq1_raw_for_boxplot.json (eval_rq1_raw_for_boxplot.py), n=10 both sides."""
    d = json.load(open(OUT_DIR / "rq1_raw_for_boxplot.json"))
    arms = [("bbr", "stock BBR-v3", "#7f7f7f"), ("pacing_only", "qbbr (pacing_gain only)", C1),
            ("extended", "qbbr (extended action space)", C2)]
    metrics = [("throughput_mbps", "Throughput (Mbps)"), ("rtt_ms", "RTT (ms)"),
               ("retransmits_per_s", "Retransmissions (count/s)")]

    fig, axes = plt.subplots(3, 1, figsize=(13, 13))
    group_width = len(arms) + 1
    for ax, (metric_key, metric_label) in zip(axes, metrics):
        all_positions, all_data, all_colors = [], [], []
        for i, loc in enumerate(LOCATIONS):
            for j, (arm_key, _arm_label, color) in enumerate(arms):
                src = d["real_cca"] if arm_key == "bbr" else d["qbbr"]
                key = f"{loc}|{arm_key}"
                all_data.append(src[key][metric_key])
                all_positions.append(i * group_width + j)
                all_colors.append(color)
        bp = ax.boxplot(all_data, positions=all_positions, widths=0.8, patch_artist=True, showfliers=False)
        for patch, color in zip(bp["boxes"], all_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.85)
        for median in bp["medians"]:
            median.set_color("black")
        tick_positions = [i * group_width + (len(arms) - 1) / 2 for i in range(len(LOCATIONS))]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(LOCATIONS)
        ax.set_ylabel(metric_label)
        if metric_key == "retransmits_per_s":
            ax.set_yscale("symlog", linthresh=1)
            ax.set_ylim(bottom=0)
            ax.set_yticks([t for t in ax.get_yticks() if t >= 0])
        ax.grid(axis="y", alpha=0.3)
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=color, alpha=0.85) for _, _, color in arms]
    axes[0].legend(handles, [label for _, label, _ in arms], fontsize=9, loc="upper right")
    _save(fig, "boxplot_rq1_vs_bbr.png",
          "qbbr vs. Stock BBR-v3: Throughput / RTT / Retransmissions (10 seeds / 10 trace-files per box)",
          top=0.95)


def main():
    fig_rq1_headline()
    fig_rq1_boxplot_vs_bbr()
    fig_rq2_alpha_sweep()
    fig_rq3_risk_toggle()
    fig_rq4_quantum_vs_classical()
    fig_shifted_freeze()
    fig_ecn_behavior_shift()
    fig_freeze_attribution()
    fig_reward_decomposition()
    fig_anticipatory_vs_decoy_lag()


if __name__ == "__main__":
    main()
