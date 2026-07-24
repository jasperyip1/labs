#!/usr/bin/env python3
"""
MIT BWSI Autonomous Drone Racing Course - UAV Neo
GNU General Public License v3.0

Plot a flight log recorded by neo_lab.record().

    python3 plot_log.py <run.csv> [out.png]

Records are written when you run a lab with NEO_RECORD set, e.g.:
    NEO_RECORD=run.csv drone sim course/week3_controls/module3_pid/main.py
    python3 plot_log.py run.csv

Each numeric channel is drawn against time; if x and z columns are present, a top-down
trajectory (x right, z forward) is added. A PNG is always written; a window also opens
if your matplotlib backend is interactive.
"""
import csv
import sys

import matplotlib
if not sys.stdout.isatty():
    matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}
    cols = {}
    for key in rows[0]:
        values = []
        for r in rows:
            try:
                values.append(float(r[key]))
            except (ValueError, KeyError, TypeError):
                values.append(float("nan"))
        cols[key] = values
    return cols


def main():
    args = sys.argv[1:]
    title = None
    if "--title" in args:
        i = args.index("--title")
        title = args[i + 1]
        del args[i:i + 2]
    if not args:
        print("usage: python3 plot_log.py <run.csv> [out.png] [--title TEXT]")
        return
    path = args[0]
    out = args[1] if len(args) > 1 else path.rsplit(".", 1)[0] + ".png"
    cols = load(path)
    if not cols:
        print(f"no data in {path}")
        return

    t = cols.get("t", list(range(len(next(iter(cols.values()))))))
    has_xz = "x" in cols and "z" in cols
    series = [k for k in cols if k not in ("t", "x", "z")]

    panels = len(series) + (1 if has_xz else 0)
    ncol = 2
    nrow = (panels + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(11, 2.6 * nrow), squeeze=False)
    flat = axes.flatten()

    i = 0
    for key in series:
        ax = flat[i]
        i += 1
        ax.plot(t, cols[key])
        ax.set_title(key)
        ax.set_xlabel("t (s)")
        ax.grid(True, alpha=0.3)
        ax.ticklabel_format(axis="y", useOffset=False, style="plain")
        if key == "heading":
            # Yaw is 0-360 and wraps; a fixed axis reads as real degrees (jumps at 360/0).
            ax.set_ylim(0, 360)
            ax.set_yticks([0, 90, 180, 270, 360])
    if has_xz:
        ax = flat[i]
        i += 1
        ax.plot(cols["x"], cols["z"], marker=".", ms=3)
        ax.plot(cols["x"][0], cols["z"][0], "go", label="start")
        ax.plot(cols["x"][-1], cols["z"][-1], "rs", label="end")
        ax.set_title("trajectory")
        ax.set_xlabel("x right (m)")
        ax.set_ylabel("z forward (m)")
        ax.axis("equal")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    for j in range(i, len(flat)):
        flat[j].axis("off")

    fig.suptitle(title if title else path)
    fig.tight_layout()
    fig.savefig(out, dpi=90)
    print(f"wrote {out}")
    if sys.stdout.isatty():
        plt.show()


if __name__ == "__main__":
    main()
