# -*- coding: utf-8 -*-
"""Erzeugt ein synthetisches weCat3D-Laserprofil, speichert es als CSV
(immer mit neuem, eindeutigem Dateinamen) und plottet es.

Angelehnt an einen wenglor weCat3D (ca. 60 mm Messbreite, 2048 Punkte).
Sensorwerte sind Richtwerte, keine Datenblattwerte.
Punkte: gleichmäßig in X (Sensorspalten) + leichte Perspektive.
"""
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# ===================== EINSTELLUNGEN =====================
# Biegungen: (Ecke X [mm], Biegewinkel [Grad], Radius [mm])
# positiver Winkel = Biegung nach unten, negativer = nach oben
BENDS = [(-15.0, 5.0, 4.0),
         (5.0, -70.0, 2.5)]
START_ANGLE = -35.0      # Neigung des ersten Schenkels [Grad], 0 = waagrecht
X_START, X_END = -32.0, 30.0
N_POINTS = 2048
SEED = None             # None = jedes Mal neues Rauschen, Zahl = reproduzierbar
OUTPUT_DIR = Path(__file__).resolve().parent / "sim_profile"   # Unterordner neben dem Skript
SAVE_PNG = True         # Plot zusätzlich als PNG speichern
# =========================================================

rng = np.random.default_rng(SEED)


def build_path(step=0.002):
    """Polyline aus Geraden und Kreisbögen (tangentiale Übergänge)."""
    dirs = [np.radians(START_ANGLE)]
    for _, ang, _ in BENDS:
        dirs.append(dirs[-1] - np.radians(ang))
    # Ecken der scharfen Kontur
    corners = [np.array([BENDS[0][0], 0.0])]
    for i in range(1, len(BENDS)):
        dx = BENDS[i][0] - corners[-1][0]
        corners.append(corners[-1] + np.array([dx, dx * np.tan(dirs[i])]))
    d0 = np.array([np.cos(dirs[0]), np.sin(dirs[0])])
    pts = [corners[0] + d0 * (X_START - corners[0][0]) / d0[0]]
    for i, (c, (_, ang, R)) in enumerate(zip(corners, BENDS)):
        a0, a1 = dirs[i], dirs[i + 1]
        u0 = np.array([np.cos(a0), np.sin(a0)])
        u1 = np.array([np.cos(a1), np.sin(a1)])
        delta = abs(a1 - a0)
        tlen = R * np.tan(delta / 2)
        p_in = c - tlen * u0
        n = int(np.linalg.norm(p_in - pts[-1]) / step) + 2
        pts.extend(np.linspace(pts[-1], p_in, n)[1:])
        sign = np.sign(a1 - a0)
        center = p_in + R * np.array([-u0[1], u0[0]]) * sign
        phi0 = np.arctan2(*(p_in - center)[::-1])
        m = max(int(R * delta / step), 10)
        phi = phi0 + sign * np.linspace(0, delta, m)
        pts.extend(center + R * np.c_[np.cos(phi), np.sin(phi)])
    last = np.array([np.cos(dirs[-1]), np.sin(dirs[-1])])
    end = pts[-1] + last * (X_END - pts[-1][0]) / last[0]
    n = int(np.linalg.norm(end - pts[-1]) / step) + 2
    pts.extend(np.linspace(pts[-1], end, n)[1:])
    path = np.array(pts)
    keep = np.r_[True, np.diff(path[:, 0]) > 1e-12]   # Doppelpunkte an Übergängen
    if np.any(np.diff(path[:, 0]) < -1e-9):
        raise ValueError("Kontur läuft in X nicht monoton (Schenkel senkrecht "
                         "oder überhängend). START_ANGLE/Winkel anpassen.")
    return path[keep]


def simulate():
    path = build_path()
    margin = 0.5
    x_nom = np.linspace(X_START + margin, X_END - margin, N_POINTS)
    z_nom = np.interp(x_nom, path[:, 0], path[:, 1])
    x = x_nom * (1 + 0.0008 * z_nom)            # leichte Perspektive
    z = np.interp(x, path[:, 0], path[:, 1])

    slope = np.gradient(z, x)
    tilt = np.arctan(np.abs(slope))
    intensity = 820 * np.cos(tilt) ** 1.5 + rng.normal(0, 25, N_POINTS)
    curv = np.abs(np.gradient(slope, x))
    intensity += 180 * np.clip(curv * 2, 0, 1)
    sigma_z = 0.006 * (1 + 2.5 * np.sin(tilt)) * np.sqrt(600 / np.clip(intensity, 80, None))
    z_meas = z + rng.normal(0, 1, N_POINTS) * sigma_z
    z_meas += 0.003 * np.sin(x * 1.7 + 0.4) + 0.002 * np.sin(x * 5.3)

    for bx, _, R in BENDS:                       # Ausreißer in den Biegungen
        near = np.where(np.abs(x - bx) < 1.5 * R)[0]
        if len(near) >= 6:
            idx = rng.choice(near, 6, replace=False)
            z_meas[idx] += rng.uniform(0.05, 0.25, 6)

    invalid = np.zeros(N_POINTS, bool)           # Ausfälle
    invalid[:25] = invalid[-30:] = True
    invalid[rng.choice(N_POINTS, 40, replace=False)] = True
    gap = rng.integers(300, 1700)
    invalid[gap:gap + 12] = True

    intensity = np.clip(np.round(intensity), 0, 1023).astype(int)
    intensity[invalid] = 0
    x_out = np.round(x, 3)
    z_out = np.round(z_meas, 3)
    z_out[invalid] = 0.0
    return x_out, z_out, intensity, invalid, path


def unique_path():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = "_".join(f"{abs(a):g}g_R{R:g}" for _, a, R in BENDS).replace(".", "p") or "gerade"
    base = OUTPUT_DIR / f"wecat3d_sim_{stamp}_{tag}"
    p, k = Path(f"{base}.csv"), 1
    while p.exists():                            # niemals überschreiben
        p = Path(f"{base}_{k}.csv")
        k += 1
    return p


def save_csv(p, x, z, inten):
    soll = "; ".join(f"Biegung{i+1} {abs(a):g} Grad R={R:g} mm"
                     for i, (_, a, R) in enumerate(BENDS))
    with open(p, "w", encoding="utf-8") as f:
        f.write("# wenglor weCat3D (simuliert) - Profil 1 von 1\n")
        f.write(f"# Points: {len(x)}; Units: mm; Seed: {SEED}\n")
        f.write(f"# Soll: {soll}\n")
        f.write("X [mm],Z [mm],Intensity\n")
        for xi, zi, ii in zip(x, z, inten):
            f.write(f"{xi:.3f},{zi:.3f},{ii}\n")


def plot(p, x, z, inten, invalid, path):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1]})
    ok = ~invalid
    ax1.plot(path[:, 0], path[:, 1], "r-", lw=1, label="Soll-Kontur")
    ax1.plot(x[ok], z[ok], ".", ms=2, color="gray", label="Messpunkte")
    for bx, a, R in BENDS:
        ax1.annotate(f"{abs(a):g}° / R{R:g}", (bx, np.interp(bx, x[ok], z[ok])),
                     textcoords="offset points", xytext=(5, 10), fontsize=9)
    ax1.set_ylabel("Z [mm]"); ax1.set_aspect("equal", adjustable="datalim")
    ax1.grid(True); ax1.legend(); ax1.set_title(p.name)
    ax2.plot(x, inten, "-", lw=0.8)
    ax2.set_xlabel("X [mm]"); ax2.set_ylabel("Intensität"); ax2.grid(True)
    fig.tight_layout()
    if SAVE_PNG:
        fig.savefig(p.with_suffix(".png"), dpi=120)
    plt.show()


if __name__ == "__main__":
    x, z, inten, invalid, path = simulate()
    out = unique_path()
    save_csv(out, x, z, inten)
    print(f"CSV gespeichert: {out}")
    plot(out, x, z, inten, invalid, path)
