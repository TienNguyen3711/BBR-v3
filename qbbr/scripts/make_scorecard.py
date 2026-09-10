"""Regenerate the 4-RQ x 6-site scorecard used on slide 8 of the status deck.

Replaces the 28 Aug image (RQ4 row = "6-qubit core", p .064) with the current
seven-input RQ4 result (exact 126-parameter full-agent match, n=10,
Mann-Whitney p >= 0.47).  -> figures/scorecard.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG_DIR = Path(__file__).resolve().parent.parent.parent / "figures"
SITES = ["London", "Mumbai", "Ohio", "SaoPaulo", "Sydney", "Tokyo"]

GREEN, AMBER, GREY = "#d7ead7", "#fbe6c8", "#e4e6ea"

ROWS = [
    ("RQ1a  pacing_gain  (pre-registered)\nretransmit reduction vs stock", [
        ("37.1% fewer", "p .024", GREEN),
        ("7.3% fewer", "p 1.00", AMBER),
        ("3.6% fewer", "p .14", AMBER),
        ("6.2% fewer", "p .23", AMBER),
        ("4.9% fewer", "p .14", AMBER),
        ("10.0% fewer", "p .68", AMBER),
    ]),
    ("RQ2  alpha-fair reward  (coexistence)\nDelta rho1  as % of rho1", [
        ("+0.17%", "sig, negligible", GREY),
        ("+0.09%", "sig, negligible", GREY),
        ("+0.16%", "sig, negligible", GREY),
        ("+0.14%", "sig, negligible", GREY),
        ("-0.05%", "sig, negligible", GREY),
        ("+0.12%", "sig, negligible", GREY),
    ]),
    ("RQ3  risk state s5/s6\nrisk-on minus off   (+ = worse)", [
        ("+0.8%", "p 1.00", GREY),
        ("+5.0%", "p .43", GREY),
        ("+3.4%", "p .73", GREY),
        ("+3.9%", "p .52", GREY),
        ("+0.2%", "p .43", GREY),
        ("+1.7%", "p .52", GREY),
    ]),
    ("RQ4  seven-input core\n126p full-agent match", [
        ("~ classical", "p .47", GREY),
        ("~ classical", "p .97", GREY),
        ("~ classical", "p .73", GREY),
        ("~ classical", "p .62", GREY),
        ("~ classical", "p .68", GREY),
        ("~ classical", "p .79", GREY),
    ]),
]

CAPTION = ("classical core, 10 seeds / site.   RQ1a = retransmit-rate reduction vs stock "
           "BBR-v3;   RQ3 = risk-on minus risk-off retransmit rate.")


def main():
    fig, ax = plt.subplots(figsize=(13.6, 4.05))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6.2)
    ax.axis("off")

    label_x, col0_x, col_w, cell_h = 0.15, 3.1, 1.46, 0.92
    top = 5.55

    for j, site in enumerate(SITES):
        ax.text(col0_x + j * col_w + col_w / 2, top + 0.30, site, ha="center", va="center",
                fontsize=13, fontweight="bold")

    for i, (label, cells) in enumerate(ROWS):
        y = top - 0.55 - i * (cell_h + 0.18)
        ax.text(label_x, y, label, ha="left", va="center", fontsize=10.5, fontweight="bold")
        for j, (big, sub, colour) in enumerate(cells):
            x = col0_x + j * col_w
            ax.add_patch(plt.Rectangle((x, y - cell_h / 2), col_w - 0.12, cell_h,
                                       facecolor=colour, edgecolor="none"))
            cx = x + (col_w - 0.12) / 2
            ax.text(cx, y + 0.16, big, ha="center", va="center", fontsize=11, fontweight="bold")
            ax.text(cx, y - 0.24, sub, ha="center", va="center", fontsize=8.5, color="#444")

    leg_y = top - 0.55 - len(ROWS) * (cell_h + 0.18) - 0.15
    for lx, colour, txt in [(0.15, GREEN, "clears pre-registered bar"),
                            (4.3, AMBER, "direction correct, fails magnitude / significance"),
                            (9.5, GREY, "null / negligible")]:
        ax.add_patch(plt.Rectangle((lx, leg_y - 0.16), 0.5, 0.32, facecolor=colour, edgecolor="none"))
        ax.text(lx + 0.68, leg_y, txt, ha="left", va="center", fontsize=9.5)

    ax.text(6, leg_y - 0.65, CAPTION, ha="center", va="center", fontsize=8.5, color="#666")

    fig.tight_layout(pad=0.4)
    out = FIG_DIR / "scorecard.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
