"""Diagrams as code: roofline (A2), RoPE frequencies (B3), attention weights (B4)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def plot_attention(weights, labels, path):
    """Plot one head's (N, N) attention weights. Rows query, columns key."""
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(weights.detach().float().cpu().numpy(), vmin=0, vmax=1, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("key token (looked at)")
    ax.set_ylabel("query token (asking)")
    ax.set_title("Causal attention: future weights are zero")
    fig.colorbar(image, ax=ax, label="attention weight")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_rope(inv_freq, num_positions, path):
    """Left: wavelength (positions per full turn) of each rotation pair. Right: cos(position x frequency)."""
    inv_freq = np.asarray(inv_freq, dtype=np.float64)
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4))
    left.semilogy(np.arange(len(inv_freq)), 2 * np.pi / inv_freq, marker="o", markersize=3)
    left.set_xlabel("pair index i")
    left.set_ylabel("wavelength (positions per turn)")
    left.set_title("rotation wavelength per pair")
    left.grid(True, which="both", alpha=0.3)
    image = right.imshow(np.cos(np.outer(np.arange(num_positions), inv_freq)), aspect="auto", cmap="RdBu", vmin=-1, vmax=1)
    right.set_xlabel("pair index i")
    right.set_ylabel("position")
    right.set_title("cos(position x frequency)")
    fig.colorbar(image, ax=right)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_roofline(peak_flops, peak_bandwidth, points, path, title="Roofline", burst_flops=None):
    """points: list of (label, intensity FLOPs/byte, achieved FLOP/s). Saves a PNG to `path`.
    burst_flops (optional) draws a dashed line for the short-burst compute peak."""
    ridge = peak_flops / peak_bandwidth
    x_values = [10 ** (e / 10) for e in range(-10, 41)]
    roof = [min(peak_flops, x * peak_bandwidth) / 1e12 for x in x_values]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.loglog(x_values, roof, color="black", label="roofline (peaks)")
    ax.axvline(ridge, color="gray", linestyle=":", label="ridge %.0f FLOPs/byte" % ridge)
    if burst_flops:
        ax.axhline(burst_flops / 1e12, color="tab:red", linestyle="--", linewidth=1, label="burst peak (before throttling)")
    for label, intensity, flops in points:
        ax.scatter([intensity], [flops / 1e12], zorder=3)
        ax.annotate(label, (intensity, flops / 1e12), textcoords="offset points", xytext=(4, 4), fontsize=8)
    ax.set_xlabel("arithmetic intensity (FLOPs / byte)")
    ax.set_ylabel("achieved TFLOPS")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
