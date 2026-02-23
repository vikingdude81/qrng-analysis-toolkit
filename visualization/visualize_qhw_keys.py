"""
Quantum Hardware QRNG Key Visualization
========================================

Generates visual analysis of 48K quantum hardware 256-bit keys:
  1. Distribution histogram vs ideal uniform
  2. 2D phase space (x[n] vs x[n+1]) — delay embedding
  3. 3D phase space trajectory
  4. Recurrence plot (proximity matrix)
  5. Hurst scaling (log-log R/S)
  6. Autocorrelation function
  7. Spectral density (FFT power spectrum)
  8. Comparison: QHW vs other QRNG sources phase space
  9. Temporal drift (rolling mean/std)
 10. Byte-level bitmap (visual randomness test)
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
from scipy.fft import rfft, rfftfreq
from scipy.stats import gaussian_kde

OUT_DIR = Path("qrng_visualizations")
OUT_DIR.mkdir(exist_ok=True)

STREAM_DIR = Path("qrng_streams")


def load_qhw_floats():
    """Load all quantum hardware floats."""
    floats = []
    for f in sorted(STREAM_DIR.glob("qrng_stream_qhw_*.json")):
        d = json.loads(f.read_text())
        floats.extend(d["floats"])
    return np.array(floats)


def load_source_floats(source_prefix):
    """Load floats from a specific source by filename prefix."""
    floats = []
    for f in sorted(STREAM_DIR.glob(f"{source_prefix}*.json")):
        d = json.loads(f.read_text())
        fvals = d.get("floats", d.get("values", d.get("data", [])))
        if fvals:
            floats.extend(fvals)
    return np.array(floats) if floats else None


def load_all_sources():
    """Load all sources grouped by source tag."""
    sources = {}
    for f in sorted(STREAM_DIR.glob("qrng_stream_*.json")):
        try:
            d = json.loads(f.read_text())
            src = d.get("source", "unknown")
            fvals = d.get("floats", d.get("values", d.get("data", [])))
            if fvals:
                sources.setdefault(src, []).extend(fvals)
        except Exception:
            pass
    return {k: np.array(v) for k, v in sources.items() if len(v) > 100}


# ── Style ──
plt.style.use("dark_background")
COLORS = {
    "quantum_hardware_256bit": "#00d4ff",
    "outshift_qrng_api": "#ff6b6b",
    "anu_qrng": "#51cf66",
    "cpu_rdrand": "#ffd43b",
    "prng_mt": "#868e96",
    "ibm_quantum_superconducting": "#da77f2",
    "cipherstone_m1": "#ff922b",
}

def get_color(src):
    for key, col in COLORS.items():
        if key in src.lower().replace(" ", "_"):
            return col
    return "#adb5bd"


def plot_distribution(data):
    """1. Distribution histogram vs ideal uniform."""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(data, bins=200, density=True, alpha=0.7, color="#00d4ff",
            edgecolor="none", label=f"QHW ({len(data):,} samples)")
    ax.axhline(1.0, color="#ff6b6b", ls="--", lw=1.5, label="Ideal Uniform")
    ax.set_xlabel("Value", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Quantum Hardware Key Distribution vs Uniform[0,1]", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.set_xlim(0, 1)
    ax.set_ylim(0.9, 1.1)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "01_distribution.png", dpi=150)
    plt.close(fig)
    print("  [1/10] Distribution histogram")


def plot_phase_space_2d(data):
    """2. 2D delay embedding phase space."""
    n = min(50000, len(data) - 1)
    x = data[:n]
    y = data[1:n+1]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Scatter
    axes[0].scatter(x[:5000], y[:5000], s=0.3, alpha=0.3, c="#00d4ff")
    axes[0].set_xlabel("x[n]")
    axes[0].set_ylabel("x[n+1]")
    axes[0].set_title("Delay Embedding (scatter, 5K)")
    axes[0].set_xlim(0, 1); axes[0].set_ylim(0, 1)
    axes[0].set_aspect("equal")

    # 2D histogram (heatmap)
    h, xedges, yedges = np.histogram2d(x, y, bins=200, range=[[0,1],[0,1]])
    im = axes[1].imshow(h.T, origin="lower", extent=[0,1,0,1],
                        cmap="inferno", aspect="equal", interpolation="gaussian")
    axes[1].set_xlabel("x[n]")
    axes[1].set_ylabel("x[n+1]")
    axes[1].set_title(f"Phase Space Density ({n:,} pts)")
    plt.colorbar(im, ax=axes[1], shrink=0.8, label="Count")

    # Lag-2
    x2 = data[:n]
    y2 = data[2:n+2] if len(data) > n+2 else data[2:]
    mn = min(len(x2), len(y2))
    h2, _, _ = np.histogram2d(x2[:mn], y2[:mn], bins=200, range=[[0,1],[0,1]])
    im2 = axes[2].imshow(h2.T, origin="lower", extent=[0,1,0,1],
                         cmap="inferno", aspect="equal", interpolation="gaussian")
    axes[2].set_xlabel("x[n]")
    axes[2].set_ylabel("x[n+2]")
    axes[2].set_title("Lag-2 Phase Space")
    plt.colorbar(im2, ax=axes[2], shrink=0.8, label="Count")

    fig.suptitle("Phase Space Analysis — Quantum Hardware Keys", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "02_phase_space_2d.png", dpi=150)
    plt.close(fig)
    print("  [2/10] 2D phase space")


def plot_phase_space_3d(data):
    """3. 3D delay embedding."""
    n = min(20000, len(data) - 2)
    x = data[:n]
    y = data[1:n+1]
    z = data[2:n+2]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(x[:5000], y[:5000], z[:5000], s=0.5, alpha=0.3,
                    c=np.arange(5000), cmap="cool")
    ax.set_xlabel("x[n]")
    ax.set_ylabel("x[n+1]")
    ax.set_zlabel("x[n+2]")
    ax.set_title("3D Phase Space Trajectory (5K pts, color=time)", fontsize=13, fontweight="bold")
    fig.colorbar(sc, shrink=0.5, label="Time step")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "03_phase_space_3d.png", dpi=150)
    plt.close(fig)
    print("  [3/10] 3D phase space")


def plot_recurrence(data):
    """4. Recurrence plot (proximity matrix)."""
    n = 1000  # small for visualization
    seg = data[:n]
    # Distance matrix
    dist = np.abs(seg[:, None] - seg[None, :])
    threshold = 0.05
    recurrence = (dist < threshold).astype(float)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].imshow(recurrence, cmap="binary", origin="lower", interpolation="none")
    axes[0].set_xlabel("Time i")
    axes[0].set_ylabel("Time j")
    axes[0].set_title(f"Recurrence Plot (ε={threshold}, n={n})")

    axes[1].imshow(dist[:500, :500], cmap="viridis", origin="lower", interpolation="none")
    axes[1].set_xlabel("Time i")
    axes[1].set_ylabel("Time j")
    axes[1].set_title("Distance Matrix (first 500)")
    plt.colorbar(axes[1].images[0], ax=axes[1], shrink=0.8, label="|x[i] - x[j]|")

    fig.suptitle("Recurrence Analysis — Quantum Hardware", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "04_recurrence.png", dpi=150)
    plt.close(fig)
    print("  [4/10] Recurrence plot")


def plot_hurst_scaling(data):
    """5. Hurst R/S log-log scaling."""
    n = len(data)
    max_k = int(np.log2(n))
    ns = [int(2**i) for i in range(3, max_k + 1)]
    rs_means = []
    for nn in ns:
        rs_vals = []
        for start in range(0, n - nn + 1, nn):
            chunk = data[start:start+nn]
            m = chunk.mean()
            y = np.cumsum(chunk - m)
            r = y.max() - y.min()
            s = chunk.std(ddof=1)
            if s > 0:
                rs_vals.append(r / s)
        if rs_vals:
            rs_means.append((nn, np.mean(rs_vals)))

    log_n = np.log([r[0] for r in rs_means])
    log_rs = np.log([r[1] for r in rs_means])
    slope, intercept = np.polyfit(log_n, log_rs, 1)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(log_n, log_rs, c="#00d4ff", s=40, zorder=5, label="R/S data")
    fit_x = np.linspace(log_n[0], log_n[-1], 100)
    ax.plot(fit_x, slope * fit_x + intercept, "--", c="#ff6b6b", lw=2,
            label=f"H = {slope:.4f}")
    ax.plot(fit_x, 0.5 * fit_x + intercept, ":", c="#51cf66", lw=1.5,
            label="H = 0.5 (pure random)")
    ax.set_xlabel("log(n)", fontsize=12)
    ax.set_ylabel("log(R/S)", fontsize=12)
    ax.set_title(f"Hurst Exponent Scaling — H = {slope:.4f}", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "05_hurst_scaling.png", dpi=150)
    plt.close(fig)
    print(f"  [5/10] Hurst scaling (H={slope:.4f})")


def plot_autocorrelation(data):
    """6. Autocorrelation function."""
    max_lag = 100
    dm = data - data.mean()
    c0 = np.sum(dm ** 2)
    acf = [np.sum(dm[:-lag] * dm[lag:]) / c0 if lag > 0 else 1.0
           for lag in range(max_lag + 1)]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(range(max_lag + 1), acf, width=1, color="#00d4ff", alpha=0.7)
    ci = 1.96 / np.sqrt(len(data))
    ax.axhline(ci, color="#ff6b6b", ls="--", lw=1, label=f"95% CI (±{ci:.5f})")
    ax.axhline(-ci, color="#ff6b6b", ls="--", lw=1)
    ax.axhline(0, color="white", lw=0.5)
    ax.set_xlabel("Lag", fontsize=12)
    ax.set_ylabel("Autocorrelation", fontsize=12)
    ax.set_title("Autocorrelation Function — Quantum Hardware (386K samples)", fontsize=14, fontweight="bold")
    ax.set_xlim(0, max_lag)
    ax.set_ylim(-0.01, 0.01)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "06_autocorrelation.png", dpi=150)
    plt.close(fig)
    print("  [6/10] Autocorrelation")


def plot_spectrum(data):
    """7. Power spectral density."""
    n = min(2**18, len(data))
    seg = data[:n]
    spectrum = np.abs(rfft(seg)) ** 2
    freqs = rfftfreq(n)

    # Smooth for visibility
    window = 50
    smoothed = np.convolve(spectrum, np.ones(window) / window, mode="valid")
    freqs_s = freqs[:len(smoothed)]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.semilogy(freqs_s[1:], smoothed[1:], color="#00d4ff", lw=0.8, alpha=0.7, label="QHW PSD")
    ax.axhline(np.median(smoothed[1:]), color="#ff6b6b", ls="--", lw=1.5,
               label=f"Median level")
    ax.set_xlabel("Frequency", fontsize=12)
    ax.set_ylabel("Power (log)", fontsize=12)
    ax.set_title("Power Spectral Density — Quantum Hardware", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "07_spectrum.png", dpi=150)
    plt.close(fig)
    print("  [7/10] Spectral density")


def plot_source_comparison(qhw, all_sources):
    """8. Phase space comparison across QRNG sources."""
    # Pick sources with enough data
    compare = {}
    for src, vals in all_sources.items():
        if len(vals) >= 1000 and "cipherstone_m2" not in src.lower():
            compare[src] = vals

    n_sources = len(compare)
    if n_sources < 2:
        print("  [8/10] Skipped (not enough sources)")
        return

    cols = min(4, n_sources)
    rows = (n_sources + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    axes = np.atleast_2d(axes)

    for idx, (src, vals) in enumerate(sorted(compare.items())):
        r, c = divmod(idx, cols)
        ax = axes[r, c]
        n = min(20000, len(vals) - 1)
        x, y = vals[:n], vals[1:n+1]
        h, _, _ = np.histogram2d(x, y, bins=150, range=[[0,1],[0,1]])
        ax.imshow(h.T, origin="lower", extent=[0,1,0,1], cmap="inferno",
                  aspect="equal", interpolation="gaussian")
        short = src.replace("_", " ").title()
        if len(short) > 25:
            short = short[:22] + "..."
        ax.set_title(f"{short}\n({len(vals):,} pts)", fontsize=10)
        ax.set_xlabel("x[n]", fontsize=8)
        ax.set_ylabel("x[n+1]", fontsize=8)

    # Hide unused axes
    for idx in range(n_sources, rows * cols):
        r, c = divmod(idx, cols)
        axes[r, c].set_visible(False)

    fig.suptitle("Phase Space Comparison Across QRNG Sources", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "08_source_comparison.png", dpi=150)
    plt.close(fig)
    print(f"  [8/10] Source comparison ({n_sources} sources)")


def plot_temporal_drift(data):
    """9. Rolling mean/std to check for temporal drift."""
    window = 5000
    n = len(data)
    steps = range(0, n - window, window // 2)
    means = [data[i:i+window].mean() for i in steps]
    stds = [data[i:i+window].std() for i in steps]
    xs = [i + window // 2 for i in steps]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    ax1.plot(xs, means, color="#00d4ff", lw=1)
    ax1.axhline(0.5, color="#ff6b6b", ls="--", lw=1.5, label="Ideal 0.5")
    ax1.fill_between(xs,
                     [0.5 - 1.96 / np.sqrt(window)] * len(xs),
                     [0.5 + 1.96 / np.sqrt(window)] * len(xs),
                     alpha=0.15, color="#ff6b6b", label="95% CI")
    ax1.set_ylabel("Rolling Mean", fontsize=12)
    ax1.set_title("Temporal Stability — Quantum Hardware Keys", fontsize=14, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.2)

    ax2.plot(xs, stds, color="#51cf66", lw=1)
    ax2.axhline(1 / np.sqrt(12), color="#ff6b6b", ls="--", lw=1.5,
                label=f"Ideal {1/np.sqrt(12):.4f}")
    ax2.set_ylabel("Rolling Std", fontsize=12)
    ax2.set_xlabel("Sample Index", fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "09_temporal_drift.png", dpi=150)
    plt.close(fig)
    print("  [9/10] Temporal drift")


def plot_bitmap(data):
    """10. Byte-level bitmap (visual randomness test)."""
    # Convert floats to bytes and render as bitmap
    n = 256 * 256  # 65536 pixels
    seg = data[:n]
    bytes_arr = (seg * 256).astype(np.uint8)
    img = bytes_arr.reshape(256, 256)

    # Also make a PRNG comparison
    prng_data = np.random.RandomState(42).random(n)
    prng_bytes = (prng_data * 256).astype(np.uint8).reshape(256, 256)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    ax1.imshow(img, cmap="gray", interpolation="none", vmin=0, vmax=255)
    ax1.set_title("Quantum Hardware (first 65K values)", fontsize=12, fontweight="bold")
    ax1.axis("off")

    ax2.imshow(prng_bytes, cmap="gray", interpolation="none", vmin=0, vmax=255)
    ax2.set_title("PRNG (Mersenne Twister)", fontsize=12)
    ax2.axis("off")

    fig.suptitle("Byte-Level Bitmap — Visual Randomness Test", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "10_bitmap.png", dpi=150)
    plt.close(fig)
    print(" [10/10] Byte bitmap")


def main():
    print("Loading quantum hardware data...")
    qhw = load_qhw_floats()
    print(f"  {len(qhw):,} values loaded\n")

    print("Loading all QRNG sources for comparison...")
    all_sources = load_all_sources()
    for src, vals in sorted(all_sources.items()):
        print(f"  {src}: {len(vals):,}")

    print(f"\nGenerating visualizations → {OUT_DIR}/")
    plot_distribution(qhw)
    plot_phase_space_2d(qhw)
    plot_phase_space_3d(qhw)
    plot_recurrence(qhw)
    plot_hurst_scaling(qhw)
    plot_autocorrelation(qhw)
    plot_spectrum(qhw)
    plot_source_comparison(qhw, all_sources)
    plot_temporal_drift(qhw)
    plot_bitmap(qhw)

    print(f"\nDone! All plots saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
