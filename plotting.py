import pandas as pd
import matplotlib.pyplot as plt
from Utils import normalize_data_to_100, SMA

# Validated categorical palette, used in fixed slot order -- never cycled.
# Slots 3 and 4 sit below 3:1 on the light surface, so every line carries a direct
# label as well as a legend entry: identity is never colour-alone.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
          "#e87ba4", "#008300", "#4a3aa7", "#e34948"]

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

# Real font-family names only. matplotlib has no CSS-style generic families, so a
# "system-ui" entry just emits a findfont warning for every text object drawn.
FONT = ["Segoe UI", "DejaVu Sans"]


def _style_axes(ax: plt.Axes) -> None:
    """Recessive chrome: hairline grid, no top/right spines, muted ticks."""
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8, alpha=1.0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(colors=INK_MUTED, labelsize=9, length=0)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily(FONT)


def _place_end_labels(ax: plt.Axes, entries: list[tuple[float, str]]) -> None:
    """
    Direct-label every line at its right end, nudging labels apart where series
    finish at similar values so they never overlap.
    """
    if not entries:
        return

    y_low, y_high = ax.get_ylim()
    min_gap = (y_high - y_low) * 0.042           # smallest legible vertical separation

    placed: list[tuple[float, str]] = []
    for value, text in sorted(entries, key=lambda e: e[0]):
        y = value
        if placed and y - placed[-1][0] < min_gap:
            y = placed[-1][0] + min_gap          # push up off the label below
        placed.append((y, text))

    x_right = ax.get_xlim()[1]
    for y, text in placed:
        ax.annotate(
            text,
            xy=(x_right, y),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            ha="left",
            fontsize=9,
            fontfamily=FONT,
            color=INK_SECONDARY,
            annotation_clip=False,
        )


def equity_vs_benchmarks_plot(
    strat_equity: pd.Series,
    benchmarks: dict[str, pd.Series],
    title: str = "Portfolio vs Benchmarks",
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """
    Growth of 100 for the strategy and each benchmark.

    Everything is indexed to 100 at the start so a dollar equity curve and index
    levels share one axis -- the alternative would be two y-scales, which is never
    correct.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(11, 5.5))
    else:
        fig = ax.figure

    _style_axes(ax)

    labels: list[tuple[float, str]] = []

    strat = normalize_data_to_100(strat_equity.dropna())
    name = strat_equity.name or "Strategy"
    ax.plot(strat.index, strat, linewidth=2.4, color=SERIES[0], label=name, zorder=5)
    labels.append((float(strat.iloc[-1]), name))

    for i, (bench_name, series) in enumerate(benchmarks.items(), start=1):
        norm = normalize_data_to_100(series.dropna())
        ax.plot(norm.index, norm, linewidth=1.6, color=SERIES[i % len(SERIES)],
                label=bench_name, zorder=4)
        labels.append((float(norm.iloc[-1]), bench_name))

    ax.axhline(y=100, linewidth=1.0, color=AXIS, zorder=1)      # the break-even reference

    ax.set_title(title, fontsize=13, fontfamily=FONT, color=INK, loc="left", pad=12)
    ax.set_ylabel("Growth of 100", fontsize=10, fontfamily=FONT, color=INK_SECONDARY)
    legend = ax.legend(loc="upper left", frameon=False, fontsize=9, ncol=2)
    for text in legend.get_texts():
        text.set_fontfamily(FONT)
        text.set_color(INK_SECONDARY)

    ax.margins(x=0.01)
    _place_end_labels(ax, labels)
    fig.set_facecolor(SURFACE)
    return fig


def drawdown_plot(
    curves: dict[str, pd.Series],
    title: str = "Drawdown from peak",
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """
    Percentage below the running peak, for each curve.

    Its own panel with its own axis rather than a second scale on the equity chart.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(11, 2.8))
    else:
        fig = ax.figure

    _style_axes(ax)

    # Deliberately at most two series here. Four overlapping drawdown traces are noise --
    # the panel exists to answer "how deep did this get, and versus what", not to rank
    # every index.
    for i, (name, series) in enumerate(list(curves.items())[:2]):
        clean = series.dropna()
        drawdown = (clean / clean.cummax() - 1.0) * 100
        colour = SERIES[i % len(SERIES)]
        if i == 0:
            ax.fill_between(drawdown.index, drawdown, 0, color=colour, alpha=0.18, zorder=3)
        ax.plot(drawdown.index, drawdown, linewidth=2.0 if i == 0 else 1.4, color=colour,
                label=name, zorder=4)

    ax.set_title(title, fontsize=11, fontfamily=FONT, color=INK_SECONDARY, loc="left", pad=8)
    ax.set_ylabel("%", fontsize=10, fontfamily=FONT, color=INK_SECONDARY)
    legend = ax.legend(loc="lower left", frameon=False, fontsize=9, ncol=2)
    for text in legend.get_texts():
        text.set_fontfamily(FONT)
        text.set_color(INK_SECONDARY)
    ax.margins(x=0.01)
    fig.set_facecolor(SURFACE)
    return fig


def performance_figure(
    strat_equity: pd.Series,
    benchmarks: dict[str, pd.Series],
    title: str = "Portfolio vs Benchmarks",
    save_path: str | None = None,
) -> plt.Figure:
    """
    Two stacked panels sharing the x-axis: growth of 100 above, drawdown below.

    Two panels rather than two y-scales on one panel.
    """
    fig, (ax_equity, ax_drawdown) = plt.subplots(
        2, 1, figsize=(11, 7.5), sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.18},
    )

    equity_vs_benchmarks_plot(strat_equity, benchmarks, title=title, ax=ax_equity)

    curves = {strat_equity.name or "Strategy": strat_equity}
    curves.update(benchmarks)
    drawdown_plot(curves, ax=ax_drawdown)

    fig.set_facecolor(SURFACE)
    # Explicit margins rather than tight_layout: the direct labels sit outside the axes
    # (annotation_clip=False), which tight_layout cannot account for and warns about.
    fig.subplots_adjust(left=0.07, right=0.86, top=0.93, bottom=0.08, hspace=0.18)
    if save_path:
        fig.savefig(save_path, dpi=160, facecolor=SURFACE, bbox_inches="tight")
        print(f"Saved chart to {save_path}")
    return fig


def equity_plot_with_moving_average(
    strat_equity: pd.Series,
    title: str = "Strategy Equity Curve with SMAs",
    ax: plt.Axes | None = None,
) -> plt.Figure:
    if ax is None:
        fig, ax = plt.subplots(figsize=(11, 4))
    else:
        fig = ax.figure

    _style_axes(ax)

    name = strat_equity.name or "Equity"
    ax.plot(strat_equity.index, strat_equity, linewidth=2.4, color=SERIES[0], label=name)
    for i, window in enumerate((12, 50), start=1):
        sma = SMA(strat_equity, window)
        ax.plot(sma.index, sma, linewidth=1.6, color=SERIES[i], label=f"{window}-Day SMA")

    ax.set_title(title, fontsize=13, fontfamily=FONT, color=INK, loc="left", pad=12)
    ax.set_ylabel("Price", fontsize=10, fontfamily=FONT, color=INK_SECONDARY)
    legend = ax.legend(loc="upper left", frameon=False, fontsize=9)
    for text in legend.get_texts():
        text.set_fontfamily(FONT)
        text.set_color(INK_SECONDARY)

    fig.set_facecolor(SURFACE)
    return fig
