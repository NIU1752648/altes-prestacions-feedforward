"""
plot_openmp_stats.py

Llegeix els logs d'execució d'OpenMP de logs/openmp/ i genera les figures de
l'informe a .docs/figures/:

  1. time_vs_threads.png        Temps d'execució vs fils (OMP vs SIMD, mitjana d'execucions)
  2. speedup_vs_reference.png   Acceleració de cada pas d'implementació vs seqüencial
  3. speedup_vs_threads.png     Acceleració vs fils, amb l'acceleració lineal ideal
  4. efficiency_vs_threads.png  Eficiència paral·lela
  5. accuracy_per_run.png       Encerts del test (/946) de cada execució, per comprovar la correcció

També escriu figures/summary.csv amb les dades agregades.

Línies base:
  - Acceleració S(n) = T_seq / T(n), amb T_seq el temps de la versió seqüencial
    original (logs/openmp/sequential/01.out).
  - Eficiència E(n) = T_1 / (n * T(n)), amb T_1 el temps del nostre codi amb 1 fil
    (mitjana de les execucions d'1 fil de la mateixa versió, OMP o SIMD). Així
    l'eficiència mesura només el que perdem en paral·lelitzar, sense barrejar-hi
    les millores del codi reestructurat.

Temps utilitzat arreu: el valor "Goodbye! (X sec)" que imprimeix el programa, que
és l'únic present també al log seqüencial.

Ús (des de qualsevol lloc):  python .docs/plot_openmp_stats.py
"""

import csv
import re
import statistics
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

# Algunes instal·lacions de matplotlib no tenen la font LastResort i fallen amb qualsevol text
mpl.rcParams["font.enable_last_resort"] = False

ROOT = Path(__file__).resolve().parent.parent.parent
LOGS = ROOT / "logs" / "openmp"
FINAL = LOGS / "actually-final"
OUT = Path(__file__).resolve().parent / "figures"

# Paleta (ranures categòriques 1-3 validades) i tinta neutra
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK_2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
VERSION_COLOR = {"OMP": BLUE, "SIMD": ORANGE}

TIME_RE = re.compile(r"Goodbye! \(([\d.]+) sec\)")
HITS_RE = re.compile(r"Total encerts = (\d+)")
FINAL_RE = re.compile(r"(omp|simd)_(\d+)threads_(\d+)\.out")

# Passos d'implementació en l'ordre en què es van desenvolupar (vegeu OPENMP_PARALLELIZATION.md)
REFERENCES = [
    ("Seqüencial", "sequential/01.out", "Base"),
    ("init_deltas", "parallel-batch1/init-deltas.out", "Lot 1"),
    ("FP static implícit", "parallel-batch1/fp-static-implicit.out", "Lot 1"),
    ("FP static explícit", "parallel-batch1/fp-static-explicit.out", "Lot 1"),
    ("FP static tot", "parallel-batch1/fp-static-all.out", "Lot 1"),
    ("BP atomic", "parallel-batch1/bp-atomic.out", "Lot 1"),
    ("BP reduction", "parallel-batch1/bp-reduction.out", "Lot 1"),
    ("UW v1", "parallel-batch2/uw-v1.out", "Lot 2"),
    ("FP simd", "parallel-batch2/fp-simd.out", "Lot 2"),
    ("FP + UW collapse", "parallel-batch2/fp-uw-collapse.out", "Lot 2"),
    ("BP separat + simd", "parallel-batch2/bp-separated-simd.out", "Lot 2"),
]
GROUP_COLOR = {"Base": MUTED, "Lot 1": BLUE, "Lot 2": ORANGE, "Final (fils màx.)": AQUA}


def parse_log(path):
    text = path.read_text(encoding="utf-8", errors="replace")
    t = TIME_RE.search(text)
    h = HITS_RE.search(text)
    if not t:
        raise ValueError(f"No s'ha trobat el temps d'execució a {path}")
    return float(t.group(1)), int(h.group(1)) if h else None


def load_final_stats():
    """Retorna {versió: {fils: [(temps, encerts), ...]}} des de logs/openmp/actually-final/."""
    data = {}
    for f in sorted(FINAL.glob("*.out")):
        m = FINAL_RE.fullmatch(f.name)
        if not m:
            continue
        version, threads = m.group(1).upper(), int(m.group(2))
        data.setdefault(version, {}).setdefault(threads, []).append(parse_log(f))
    # OMP abans que SIMD, i fils en ordre numèric
    return {v: dict(sorted(data[v].items())) for v in sorted(data)}


def mean_times(stats):
    """Retorna {versió: {fils: temps mitjà}}."""
    return {v: {p: statistics.mean(t for t, _ in runs) for p, runs in bt.items()}
            for v, bt in stats.items()}


def own_baseline(means):
    """T_1 del nostre codi amb 1 fil, per versió: base de l'eficiència."""
    return {v: m[1] for v, m in means.items()}


def style_axes(ax, title, xlabel, ylabel):
    ax.set_title(title, loc="left", fontsize=13, color=INK, pad=12)
    ax.set_xlabel(xlabel, color=INK_2)
    ax.set_ylabel(ylabel, color=INK_2)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=INK_2)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=200)
    plt.close(fig)
    print(f"  desat figures/{name}")


def plot_time_vs_threads(stats, seq_time):
    fig, ax = plt.subplots(figsize=(8, 5))
    threads = sorted(next(iter(stats.values())))
    ideal = [seq_time / p for p in threads]
    ax.plot(threads, ideal, "--", color=MUTED, linewidth=1.5, label="Ideal (T_seqüencial / p)")
    for version, by_threads in stats.items():
        means = [statistics.mean(t for t, _ in by_threads[p]) for p in threads]
        lo = [m - min(t for t, _ in by_threads[p]) for m, p in zip(means, threads)]
        hi = [max(t for t, _ in by_threads[p]) - m for m, p in zip(means, threads)]
        ax.errorbar(threads, means, yerr=[lo, hi], marker="o", markersize=7, linewidth=2,
                    capsize=4, color=VERSION_COLOR.get(version), label=version)
        ax.annotate(f"{means[-1]:.2f} s", (threads[-1], means[-1]),
                    textcoords="offset points", xytext=(8, -4 if version == "SIMD" else 6),
                    color=INK_2, fontsize=9)
    ax.set_xticks(threads)
    style_axes(ax, "Temps d'execució versus fils", "Fils", "Temps [s]")
    ax.legend(frameon=False)
    save(fig, "time_vs_threads.png")


def plot_speedup_vs_reference(seq_time, final_max, max_threads, n_runs):
    labels, speedups, groups = [], [], []
    for name, rel, group in REFERENCES:
        t, _ = parse_log(LOGS / rel)
        labels.append(name)
        speedups.append(seq_time / t)
        groups.append(group)
    final_group = "Final (fils màx.)"
    for version, t in final_max.items():
        labels.append(f"Final {version}")
        speedups.append(seq_time / t)
        groups.append(final_group)
    colors = [GROUP_COLOR[g] for g in groups]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = range(len(labels))
    bars = ax.bar(x, speedups, color=colors, width=0.7, edgecolor="white", linewidth=2)
    ax.axhline(1, color=MUTED, linewidth=1, linestyle="--")
    for bar, s in zip(bars, speedups):
        ax.annotate(f"{s:.2f}x", (bar.get_x() + bar.get_width() / 2, s),
                    textcoords="offset points", xytext=(0, 3), ha="center",
                    fontsize=8.5, color=INK_2)
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"#{i}\n{l}" for i, l in enumerate(labels)], rotation=45,
                       ha="right", fontsize=8.5)
    style_axes(ax, "Acceleració de cada pas d'implementació vs seqüencial",
               "Referència d'implementació (en ordre de desenvolupament)",
               "Acceleració (T_seq / T)")
    ax.set_ylim(0, max(speedups) * 1.12)
    handles = [plt.Rectangle((0, 0), 1, 1, color=GROUP_COLOR[g]) for g in GROUP_COLOR]
    ax.legend(handles, GROUP_COLOR.keys(), frameon=False, loc="upper left")
    ax.text(0.01, 0.62,
            "Els logs dels lots són acumulatius, d'una sola execució.\n"
            f"Final = mitjana de {n_runs} execucions amb {max_threads} fils.",
            transform=ax.transAxes, ha="left", va="top", fontsize=8, color=MUTED)
    save(fig, "speedup_vs_reference.png")


def plot_speedup_and_efficiency(stats, seq_time):
    threads = sorted(next(iter(stats.values())))
    means = mean_times(stats)
    t1 = own_baseline(means)

    # Acceleració respecte al seqüencial original
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot([0] + threads, [0] + threads, "--", color=MUTED, linewidth=1.5, label="Ideal (lineal)")
    for version, m in means.items():
        sp = [seq_time / m[p] for p in threads]
        ax.plot(threads, sp, marker="o", markersize=7, linewidth=2,
                color=VERSION_COLOR.get(version), label=version)
        ax.annotate(f"{sp[-1]:.2f}x", (threads[-1], sp[-1]), textcoords="offset points",
                    xytext=(8, -4 if version == "OMP" else 4), color=INK_2, fontsize=9)
    ax.set_xticks(threads)
    ax.set_xlim(0, threads[-1] + 1.5)
    ax.set_ylim(0, threads[-1] + 1.5)
    style_axes(ax, "Acceleració versus fils", "Fils", "Acceleració (T_seq / T)")
    ax.legend(frameon=False, loc="upper left")
    save(fig, "speedup_vs_threads.png")

    # Eficiència respecte al temps d'1 fil del nostre codi
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axhline(1, color=MUTED, linewidth=1.5, linestyle="--", label="Ideal (100%)")
    for version, m in means.items():
        eff = [t1[version] / (m[p] * p) for p in threads]
        ax.plot(threads, eff, marker="o", markersize=7, linewidth=2,
                color=VERSION_COLOR.get(version), label=version)
        ax.annotate(f"{eff[-1]:.1%}", (threads[-1], eff[-1]), textcoords="offset points",
                    xytext=(8, -4 if version == "OMP" else 4), color=INK_2, fontsize=9)
    ax.set_xticks(threads)
    ax.set_xlim(0, threads[-1] + 2)
    ax.set_ylim(0, 1.1)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    style_axes(ax, "Eficiència versus fils", "Fils", "Eficiència (T₁ / (n·T))")
    ax.legend(frameon=False, loc="lower left")
    save(fig, "efficiency_vs_threads.png")


def plot_accuracy(stats, seq_hits):
    fig, ax = plt.subplots(figsize=(8, 5))
    threads = sorted(next(iter(stats.values())))
    offset = {"OMP": -0.17, "SIMD": 0.17}
    for version, by_threads in stats.items():
        xs, ys = [], []
        for i, p in enumerate(threads):
            runs = by_threads[p]
            for r, (_, h) in enumerate(runs):
                # reparteix horitzontalment les execucions d'un grup perquè les puntuacions iguals no s'amaguin
                spread = (r - (len(runs) - 1) / 2) * 0.03
                xs.append(i + offset.get(version, 0) + spread)
                ys.append(h)
            low = [h for _, h in runs if h < seq_hits - 20]
            if low:
                ax.annotate(f"{len(low)}/{len(runs)} execucions a {min(low)}-{max(low)}",
                            (i + offset.get(version, 0), min(low)), textcoords="offset points",
                            xytext=(0, 10), ha="center", fontsize=8.5, color=INK_2)
        ax.scatter(xs, ys, s=40, color=VERSION_COLOR.get(version), edgecolors="white",
                   linewidths=1.2, label=version, zorder=3)
    ax.axhline(seq_hits, color=MUTED, linestyle="--", linewidth=1.2,
               label=f"Seqüencial ({seq_hits})")
    ax.set_xticks(range(len(threads)))
    ax.set_xticklabels([str(p) for p in threads])
    style_axes(ax, "Prediccions correctes del test per execució (sobre 946)",
               "Fils", "Prediccions correctes")
    ax.legend(frameon=False, loc="lower left")
    save(fig, "accuracy_per_run.png")


def write_summary(stats, seq_time):
    means = mean_times(stats)
    t1 = own_baseline(means)
    with open(OUT / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["version", "threads", "runs", "mean_s", "min_s", "max_s", "stdev_s",
                    "speedup", "efficiency", "min_hits", "max_hits"])
        for version, by_threads in stats.items():
            for p in sorted(by_threads):
                times = [t for t, _ in by_threads[p]]
                hits = [h for _, h in by_threads[p]]
                m = means[version][p]
                w.writerow([version, p, len(times), f"{m:.4f}", f"{min(times):.4f}",
                            f"{max(times):.4f}", f"{statistics.stdev(times):.4f}",
                            f"{seq_time / m:.3f}", f"{t1[version] / (m * p):.3f}",
                            min(hits), max(hits)])
    print("  desat figures/summary.csv")


def main():
    OUT.mkdir(exist_ok=True)
    seq_time, seq_hits = parse_log(LOGS / "sequential" / "01.out")
    stats = load_final_stats()
    if not stats or any(1 not in bt for bt in stats.values()):
        raise SystemExit(f"Falten logs d'1 fil a {FINAL}: no es pot calcular l'eficiència")
    means = mean_times(stats)
    t1 = own_baseline(means)

    print(f"Seqüencial original: {seq_time:.2f} s, {seq_hits} encerts")
    for version, by_threads in stats.items():
        print(f"  {version} T_1 (nostre codi, 1 fil) = {t1[version]:.3f} s")
        for p in sorted(by_threads):
            n = len(by_threads[p])
            print(f"  {version:5s} {p:2d} fils: mitjana {means[version][p]:.3f} s en {n} execucions")

    max_threads = max(next(iter(stats.values())))
    final_max = {v: m[max_threads] for v, m in means.items()}
    n_runs = len(next(iter(stats.values()))[max_threads])

    plot_time_vs_threads(stats, seq_time)
    plot_speedup_vs_reference(seq_time, final_max, max_threads, n_runs)
    plot_speedup_and_efficiency(stats, seq_time)
    plot_accuracy(stats, seq_hits)
    write_summary(stats, seq_time)


if __name__ == "__main__":
    main()
