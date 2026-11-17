"""Chart builders. Each returns a matplotlib Figure; to_png() turns it into bytes
for st.download_button, so the downloaded image is exactly what is on screen.
No Streamlit and no pyplot (global state) in here.
"""
import io

from matplotlib.figure import Figure

COLORS = {"Pollard's Rho": "#1f77b4", "Baby-step Giant-step": "#d62728"}


def _valid(y):
    return y is not None and y == y and y > 0  # not None, not NaN, positive (log axis)


def _style(ax, title, xlabel, ylabel, log_y):
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if log_y:
        ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)


def line_chart(x, series, title, xlabel, ylabel, log_y=True):
    """series: {label: [y per x]} (None for missing points)."""
    fig = Figure(figsize=(7, 4))
    ax = fig.subplots()
    for label, ys in series.items():
        pts = [(xi, yi) for xi, yi in zip(x, ys) if _valid(yi)]
        if pts:
            xs, vals = zip(*pts)
            ax.plot(xs, vals, marker="o", label=label, color=COLORS.get(label))
    _style(ax, title, xlabel, ylabel, log_y)
    if ax.get_legend_handles_labels()[0]:
        ax.legend()
    fig.tight_layout()
    return fig


def bar_chart(labels, values, title, ylabel, fmt, log_y=True):
    """One bar per label; None means 'not available' and is annotated."""
    fig = Figure(figsize=(5, 4))
    ax = fig.subplots()
    for i, (label, value) in enumerate(zip(labels, values)):
        if _valid(value):
            ax.bar(i, value, color=COLORS.get(label))
            ax.annotate(fmt(value), (i, value), ha="center", va="bottom", fontsize=9)
        else:
            ax.annotate("n/a", (i, 0.5), ha="center", va="bottom", fontsize=9, transform=ax.get_xaxis_transform())
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8)
    _style(ax, title, "", ylabel, log_y)
    ax.margins(y=0.2)
    fig.tight_layout()
    return fig


def histogram(values, title, xlabel, vlines=()):
    """vlines: [(x, label)] drawn as dashed vertical lines."""
    fig = Figure(figsize=(7, 4))
    ax = fig.subplots()
    ax.hist(values, bins=min(40, max(5, len(values) // 5)), color="#1f77b4", alpha=0.8)
    for x, label in vlines:
        ax.axvline(x, linestyle="--", label=label, color="#d62728" if "theory" in label.lower() else "black")
    _style(ax, title, xlabel, "Number of runs", log_y=False)
    if vlines:
        ax.legend()
    fig.tight_layout()
    return fig


def to_png(fig, dpi=200):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    return buf.getvalue()
