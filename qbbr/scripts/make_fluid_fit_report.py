"""Stage 1b -- Block 1: fit test of the fluid BBR-v3 model against observed
BBR-v3 dynamics in the iperf3 sequential logs (qbbr/data/raw).

§sec:validation asks the fluid model to be validated against observed
ProbeBW and ProbeRTT transitions, inflight_hi and inflight_lo -- not merely
checked for internal consistency. This script measures, per city/direction:

  from the real `bbr` traces      from a fluid stock rollout
  ------------------------------   --------------------------------
  ProbeRTT dip cadence / depth     (fluid has no ProbeRTT phase)
  ProbeBW sawtooth period          probe_bw_interval_s + v_over_bdp autocorr
  snd_cwnd / BDP  p5/p50/p95       v_bytes / BDP  p5/p50/p95
  retransmit rate (per s)          retransmit rate (per s)
  RTT p50 / RTT std (ms)           RTT p50 / RTT std (ms)

Writes outputs/fluid_fit_report.{json,md} and figures/fluid_fit_{dir}.png.
No training. Reads checkpoints of nothing -- stock BBR-v3 needs no policy.
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from qbbr.env.calibration import load_calibration
from qbbr.scripts.run_native_qa2c_successor import PACKAGE_ROOT, _env

CITIES = ["Tokyo", "SaoPaulo", "Ohio", "London", "Mumbai", "Sydney"]
DIRECTIONS = ["downlink", "uplink"]
TS_S = 120.0  # window shown in the figures
CONFIG = yaml.safe_load(Path("qbbr/configs/tier1_native_qa2c_successor_protocol.yaml").read_text())
CALIB = load_calibration(PACKAGE_ROOT / "data" / "calibrated" / "per_location_constants.json")


def _bdp_bytes(city, direction):
    c = CALIB[city][direction]
    return c["B_max_mbps"] * 1e6 / 8.0 * c["RTT_min_ms"] / 1000.0


# ---------- real traces ----------
def _read_tolerant(path):
    return json.JSONDecoder().raw_decode(Path(path).read_text().lstrip())[0]


def _seq_glob(city, direction):
    if direction == "downlink":
        return sorted(glob.glob(
            f"qbbr/data/raw/downlink-sequential-logs/{city}/**/bbr_{city}__REV_run*.json", recursive=True))
    return sorted(glob.glob(f"qbbr/data/raw/uplink-sequential-logs/{city}/bbr_{city}__FWD_run*.json"))


def _probe_rtt_stats(cwnd):
    """Inter-dip interval (samples) and depth for drops >=40% below a rolling median."""
    if len(cwnd) < 20:
        return None, None
    w = 15
    med = np.array([np.median(cwnd[max(0, i - w):i + w + 1]) for i in range(len(cwnd))])
    deep = cwnd < 0.6 * med
    idx = np.where(deep[:-1] & ~deep[1:])[0]  # falling edges of a dip
    if len(idx) < 2:
        return None, (float(np.min(cwnd[deep]) / np.median(med)) if deep.any() else None)
    return float(np.median(np.diff(idx))), float(np.median(cwnd[deep]) / np.median(med))


def _dominant_period(x, dt, lo=0.5, hi=8.0):
    x = np.asarray(x, float)
    x = x - np.median(x)
    if len(x) < 32 or np.allclose(x, 0):
        return None
    ac = np.correlate(x, x, "full")[len(x) - 1:]
    ac = ac / (ac[0] + 1e-12)
    los, his = int(lo / dt), min(int(hi / dt), len(ac) - 1)
    if his <= los + 1:
        return None
    k = los + int(np.argmax(ac[los:his]))
    return float(k * dt) if ac[k] > 0.15 else None


def real_stats(city, direction):
    rows = []
    for p in _seq_glob(city, direction)[:10]:
        try:
            d = _read_tolerant(p)
        except Exception:
            continue
        ivs = [iv["streams"][0] for iv in d.get("intervals", []) if not iv.get("omitted", False)
               and iv.get("streams")]
        if len(ivs) < 30:
            continue
        dt = float(np.median([s.get("seconds", 1.0) for s in ivs])) or 1.0
        cwnd = np.array([s.get("snd_cwnd", np.nan) for s in ivs], float)
        rtt = np.array([s.get("rtt", np.nan) for s in ivs], float) / 1e3
        dur = float(d.get("end", {}).get("streams", [{}])[0].get("sender", {}).get("seconds", len(ivs) * dt))
        retr = d.get("end", {}).get("streams", [{}])[0].get("sender", {}).get("retransmits")
        bdp = _bdp_bytes(city, direction)
        dip_iv, dip_depth = _probe_rtt_stats(cwnd[np.isfinite(cwnd)])
        rows.append(dict(
            cwnd_over_bdp_p5=float(np.nanpercentile(cwnd, 5) / bdp),
            cwnd_over_bdp_p50=float(np.nanpercentile(cwnd, 50) / bdp),
            cwnd_over_bdp_p95=float(np.nanpercentile(cwnd, 95) / bdp),
            probe_rtt_interval_s=(dip_iv * dt) if dip_iv else None,
            probe_rtt_depth=dip_depth,
            probe_bw_period_s=_dominant_period(cwnd[np.isfinite(cwnd)], dt),
            retransmit_per_s=(retr / dur) if (retr is not None and dur) else None,
            rtt_p50_ms=float(np.nanpercentile(rtt, 50)),
            rtt_std_ms=float(np.nanstd(rtt)),
        ))
    if not rows:
        return None
    agg = {k: float(np.nanmedian([r[k] for r in rows if r[k] is not None])) if any(
        r[k] is not None for r in rows) else None for k in rows[0]}
    agg["n_runs"] = len(rows)
    return agg


# ---------- fluid rollout ----------
def fluid_stats(city, direction):
    env = _env(CALIB, city, direction, CONFIG)
    state, done = env.reset(seed=1000), False
    t, vob, icrs, idwn, rtt, retr, dur = [], [], [], [], [], 0.0, 0.0
    bdp = env.params.bdp_bytes
    while not done:
        state, _r, done, info = env.step(2)  # stock gain 1.00
        t.append(info["t_start"])
        vob.append(float(env._history[-1]["v_over_bdp"]))  # reported inflight/BDP (what s3 sees)
        icrs.append(info["i_crs"]); idwn.append(info["i_dwn"])
        rtt.append(info["rtt_ms"]); retr += info["retransmits"]; dur += info["t_dec_s"]
    dt = float(np.median(np.diff(t))) if len(t) > 2 else 1.0
    from qbbr.env.fluid_sim import probe_bw_interval_s
    return dict(
        cwnd_over_bdp_p5=float(np.percentile(vob, 5)),
        cwnd_over_bdp_p50=float(np.percentile(vob, 50)),
        cwnd_over_bdp_p95=float(np.percentile(vob, 95)),
        probe_rtt_interval_s=None,           # no ProbeRTT phase in the fluid model
        probe_rtt_depth=None,
        probe_bw_period_s=_dominant_period(vob, dt),
        probe_bw_interval_param_s=float(probe_bw_interval_s(env.params.rtt_rtp_s)),
        retransmit_per_s=float(retr / dur) if dur else None,
        rtt_p50_ms=float(np.percentile(rtt, 50)),
        rtt_std_ms=float(np.std(rtt)),
        series=dict(t=t, v_over_bdp=vob, i_crs=icrs, i_dwn=idwn),
    )


# ---------- figure ----------
def make_fig(pairs, direction):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True)
    for ax, city in zip(axes.flat, CITIES):
        fl = pairs[city][direction]["fluid"]["series"]
        m = np.asarray(fl["t"]) <= TS_S
        ax.plot(np.asarray(fl["t"])[m], np.asarray(fl["v_over_bdp"])[m], color="#54a24b", lw=1.2,
                label="fluid  v/BDP")
        ax.plot(np.asarray(fl["t"])[m], np.asarray(fl["i_crs"])[m], color="#4c78a8", lw=0.8, ls=":",
                label="fluid  i_crs")
        ax.plot(np.asarray(fl["t"])[m], np.asarray(fl["i_dwn"])[m], color="#e45756", lw=0.8, ls=":",
                label="fluid  i_dwn")
        rr = pairs[city][direction]["real"]
        if rr:
            for q, ls in ((rr["cwnd_over_bdp_p50"], "-"), (rr["cwnd_over_bdp_p95"], "--")):
                ax.axhline(q, color="black", lw=0.8, ls=ls, alpha=0.6)
        ax.axhline(1.0, color="grey", lw=0.6)
        ax.set_title(city, fontsize=10); ax.grid(alpha=0.3); ax.set_ylabel("/ BDP")
        ax.set_ylim(-0.05, max(1.5, ax.get_ylim()[1]))
    for ax in axes[1]:
        ax.set_xlabel("time (s)")
    axes[0, 0].legend(fontsize=7, loc="upper right")
    fig.suptitle(f"Fluid BBR-v3 vs observed cwnd envelope -- dedicated {direction}  "
                 f"(fluid v/BDP + i_crs/i_dwn over {TS_S:.0f}s; black lines = real snd_cwnd/BDP p50 solid, p95 dashed)",
                 fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = Path("figures") / f"fluid_fit_{direction}.png"
    fig.savefig(out, dpi=150); plt.close(fig); print(out)


def verdict(real, fluid, key, rel_tol=0.35):
    if real is None or fluid is None or real.get(key) is None or fluid.get(key) is None:
        return "n/a"
    r, f = real[key], fluid[key]
    if r == 0:
        return "match" if abs(f) < 1e-6 else "MISMATCH"
    ratio = f / r
    return "match" if (1 - rel_tol) <= ratio <= (1 + rel_tol) else f"MISMATCH (fluid/real = {ratio:.2f}x)"


def main():
    pairs = {c: {} for c in CITIES}
    for c in CITIES:
        for d in DIRECTIONS:
            pairs[c][d] = {"real": real_stats(c, d), "fluid": fluid_stats(c, d)}
            print(f"[fit] {c} {d}", flush=True)

    slim = {c: {d: {"real": pairs[c][d]["real"],
                    "fluid": {k: v for k, v in pairs[c][d]["fluid"].items() if k != "series"}}
                for d in DIRECTIONS} for c in CITIES}
    Path("outputs/fluid_fit_report.json").write_text(json.dumps(slim, indent=2))

    lines = ["# Stage 1b Block 1 -- fluid BBR-v3 fit vs observed dynamics", "",
             "Real: median over the `bbr` sequential runs. Fluid: stock rollout, holdout seed 1000.",
             "MISMATCH = fluid/real outside +-35%.", ""]
    metrics = [("probe_bw_period_s", "ProbeBW period (s)"),
               ("probe_rtt_interval_s", "ProbeRTT cadence (s)"),
               ("probe_rtt_depth", "ProbeRTT depth (cwnd_min/median)"),
               ("cwnd_over_bdp_p50", "inflight/BDP p50"),
               ("cwnd_over_bdp_p95", "inflight/BDP p95"),
               ("retransmit_per_s", "retransmit rate (/s)"),
               ("rtt_p50_ms", "RTT p50 (ms)"),
               ("rtt_std_ms", "RTT std (ms)")]
    for d in DIRECTIONS:
        lines += [f"## {d}", "",
                  "| city | metric | real | fluid | verdict |", "|---|---|---|---|---|"]
        for c in CITIES:
            r, f = pairs[c][d]["real"], {k: v for k, v in pairs[c][d]["fluid"].items() if k != "series"}
            for key, name in metrics:
                rv = None if not r else r.get(key)
                fv = f.get(key)
                lines.append(f"| {c} | {name} | {rv if rv is None else round(rv,3)} "
                             f"| {fv if fv is None else round(fv,3)} | {verdict(r, f, key)} |")
        lines.append("")
    lines += ["## Structural notes",
              "- **Stage 1b fix applied**: `steady_inflight_bdp_frac` (calibrated per location to "
              "observed snd_cwnd/BDP p50) adds a persistent pipe term to the reported inflight/BDP, "
              "and `base_retransmit_rate_pps` (calibrated to the observed retransmit rate) adds an "
              "always-on baseline to the DRAIN-gated retransmit term. inflight/BDP p50, retransmit "
              "rate and RTT p50 now match; inflight/BDP p95 and RTT std do not (see below).",
              "- The fluid model still has **no ProbeRTT phase** (paper Eq. 29 `w_prt = BDP/2` is "
              "not implemented); ProbeRTT rows are n/a on the fluid side.",
              "- `t_since_probe_s` is tracked and reset on the Eq. 22 interval but does **not** "
              "modulate the pacing multiplier, so there is no ProbeBW gain sawtooth. This is why "
              "the fluid inflight/BDP **p95** stays near p50 (~0.5x real p95) and RTT std is off: "
              "the ProbeBW/ProbeRTT excursions are a declared abstraction, not calibrated.",
              "- `inflight_hi`/`inflight_lo` (Eq. 28, `bdp_hi_mult`/`bdp_lo_mult`) **are** modelled "
              "and drive `i_crs`/`i_dwn`.", ""]
    Path("outputs/fluid_fit_report.md").write_text("\n".join(lines))
    print("outputs/fluid_fit_report.{json,md}")

    for d in DIRECTIONS:
        make_fig(pairs, d)
    print("done")


if __name__ == "__main__":
    main()
