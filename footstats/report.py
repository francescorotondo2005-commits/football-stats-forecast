"""Report di valutazione: tabella in Markdown e due grafici PNG per il README."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .config import DEFAULT_LEVEL, REPORTS_DIR, STAT_LABELS_IT, TEST_SEASONS  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
BAND = "#b7d3f6"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "text.color": INK, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
})


def _pct(x):
    return f"{x * 100:.1f}%"


def write_markdown(results: dict) -> str:
    lvl = int(DEFAULT_LEVEL * 100)
    seasons = ", ".join(f"{y}-{(y + 1) % 100:02d}" for y in TEST_SEASONS)
    lines = [
        f"# Risultati sul test ({seasons}, partite mai viste in training)",
        "",
        "## Errore medio assoluto (MAE) del valore atteso",
        "",
        "| Statistica | Media campionato | Media mobile | GLM Poisson | Gradient Boosting | Scelto (in validazione) | Miglioramento vs media mobile |",
        "|---|---|---|---|---|---|---|",
    ]
    for stat, r in results.items():
        f = r["families"]
        mae = {k: v["test"]["mae"] for k, v in f.items()}
        chosen = r["chosen"]
        gain = 1 - mae[chosen] / mae["media mobile"]
        lines.append(
            f"| {STAT_LABELS_IT[stat]} | {mae['media campionato']:.3f} | {mae['media mobile']:.3f} | "
            f"{mae['GLM Poisson']:.3f} | {mae['Gradient Boosting']:.3f} | {chosen} | {gain:.1%} |"
        )
    lines += [
        "",
        f"## Intervalli di previsione al {lvl}% (modello scelto)",
        "",
        "Con statistiche intere l'intervallo non può contenere esattamente l'80% di probabilità:",
        "la colonna *Probabilità dichiarata* è quella che il modello assegna davvero all'intervallo.",
        "Il modello è ben calibrato se la *Copertura reale* è vicina alla probabilità dichiarata.",
        "",
        "| Statistica | Ampiezza media | Probabilità dichiarata | Copertura reale | Distribuzione |",
        "|---|---|---|---|---|",
    ]
    for stat, r in results.items():
        fam = r["families"][r["chosen"]]
        t = fam["test"]
        dist = "Poisson" if fam["k"] >= 9900 else f"Binomiale Negativa (k = {fam['k']:.0f})"
        lines.append(
            f"| {STAT_LABELS_IT[stat]} | {t[f'width_{lvl}']:.1f} | {_pct(t[f'claimed_{lvl}'])} | "
            f"{_pct(t[f'cov_{lvl}'])} | {dist} |"
        )
    text = "\n".join(lines) + "\n"
    (REPORTS_DIR / "results.md").write_text(text)
    return text


def plot_calibration(results: dict) -> None:
    """Probabilità dichiarata vs copertura reale, per più livelli: sulla diagonale = calibrato."""
    fig, ax = plt.subplots(figsize=(6.4, 5.2), dpi=150)
    ax.plot([0.45, 1.0], [0.45, 1.0], color=INK_2, lw=1, ls=(0, (4, 3)), zorder=1, label="Calibrazione perfetta")
    for color, (stat, r) in zip(SERIES, results.items()):
        t = r["families"][r["chosen"]]["test"]
        levels = sorted(int(k.split("_")[1]) for k in t if k.startswith("cov_"))
        x = [t[f"claimed_{l}"] for l in levels]
        y = [t[f"cov_{l}"] for l in levels]
        ax.plot(x, y, color=color, lw=2, marker="o", ms=5, mec=SURFACE, mew=1.5,
                label=STAT_LABELS_IT[stat], zorder=3)
    ax.set_xlim(0.45, 1.0)
    ax.set_ylim(0.45, 1.0)
    ax.set_xlabel("Probabilità dichiarata dal modello per l'intervallo")
    ax.set_ylabel("Copertura reale sul test")
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    ax.set_title("Calibrazione degli intervalli (livelli 50–95%)", loc="left", fontsize=12, color=INK)
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "calibration.png")
    plt.close(fig)


def plot_team_example(results: dict, team="Inter", stat="shots") -> None:
    """Le previsioni partita per partita di una squadra in una stagione di test."""
    frame = results[stat]["_test_frame"]
    season = max(TEST_SEASONS)
    d = frame[(frame.team == team) & (frame.season == season)].sort_values("date")
    if d.empty:
        return
    x = np.arange(len(d))
    inside = (d.y >= d.lo) & (d.y <= d.hi)

    fig, ax = plt.subplots(figsize=(10, 4.4), dpi=150)
    ax.vlines(x, d.lo, d.hi, color=BAND, lw=6, zorder=1, label=f"Range previsto ({int(DEFAULT_LEVEL * 100)}%)")
    ax.plot(x, d.mu, color=SERIES[0], lw=2, zorder=2, label="Valore atteso")
    ax.scatter(x[inside], d.y[inside], s=26, color=INK, zorder=4, label="Valore reale (dentro il range)",
               edgecolor=SURFACE, linewidth=1.2)
    ax.scatter(x[~inside], d.y[~inside], s=34, color=SERIES[1], marker="D", zorder=4,
               label="Valore reale (fuori dal range)", edgecolor=SURFACE, linewidth=1.2)
    ticks = x[::4]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{r.opponent}\n{'C' if r.is_home else 'T'}" for r in d.iloc[::4].itertuples()], fontsize=8)
    ax.set_ylabel(STAT_LABELS_IT[stat])
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    cov = inside.mean()
    ax.set_title(f"{team}, {STAT_LABELS_IT[stat].lower()} per partita, Serie A {season}-{(season + 1) % 100:02d} "
                 f"(dentro il range: {inside.sum()}/{len(d)}, {cov:.0%})", loc="left", fontsize=12, color=INK)
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.18), fontsize=9)
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / f"example_{team.lower()}_{stat}.png")
    plt.close(fig)


def write_report(results: dict) -> None:
    write_markdown(results)
    plot_calibration(results)
    plot_team_example(results)
