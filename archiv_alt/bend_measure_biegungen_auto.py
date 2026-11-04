# -*- coding: utf-8 -*-
"""Biegewinkel und Biegeradien (mehrere Biegungen) aus einem Laserlinien-Profil.

Vorgehen bei mehreren Biegungen:
    1. Grobe Ecken per rekursiver Sehnen-Teilung (Douglas-Peucker-Prinzip)
    2. Profil an den Schenkelmitten zwischen den Ecken in Abschnitte teilen
    3. Jeden Abschnitt wie eine Einzelbiegung auswerten (Geraden + Bezier)

Das Skript liest CSV-Dateien mit Komma, Semikolon oder Tabulator als
Spaltentrenner ein. Es überspringt Kopf- und Metadatenzeilen automatisch,
sofern sich anschließend numerische X-, Z- und Intensitätswerte finden lassen.

Benötigt:
    python -m pip install numpy matplotlib

Die CSV muss mindestens drei numerische Spalten für X, Z und Intensität
enthalten. Einheiten wie vom Sensor (z. B. mm), Winkel in Grad.
"""

from dataclasses import dataclass
from math import comb
from pathlib import Path
import csv
import re

import numpy as np

# ============================================================================
# EINSTELLUNGEN
# ============================================================================
# Mac-Beispiel. In Google Colab den Pfad z. B. auf
# "/content/wecat3d_scan_20261022_155323.csv" ändern.
PROFILE_PATH = Path(
    "/Users/Laurin/Documents/python/biegekante/wecat3d_sim_20261028_111144_5g_R4_70g_R2p5.csv"
)

PLOT = True

# Anzahl der Biegungen im Profil:
#   None      -> automatische Erkennung (adaptiv)
#   1, 2, 3.. -> genau so viele Biegungen suchen
N_BENDS = None
MIN_LEG_LENGTH = 3.0    # mm; Mindestabstand zwischen zwei groben Ecken

# Nur für die automatische Erkennung (N_BENDS = None):
MIN_DEVIATION = 0.10    # mm; minimale Sehnenabweichung, ab der eine Ecke zählt
                        # (deutlich über Rauschen + Welligkeit wählen)
MIN_BEND_ANGLE = 2.0    # Grad; kleinere "Biegungen" werden verworfen
MAX_BENDS = 20          # Sicherheitsgrenze
SMOOTH_WINDOW = 15      # Punkte; Medianfilter nur für die grobe Eckensuche

# Datenaufbereitung
I_MIN = 1.0             # Punkte mit Intensität <= I_MIN werden verworfen
X_MIN = -np.inf         # optionaler X-Bereich, z. B. -40.0
X_MAX = np.inf          # optionaler X-Bereich, z. B. 15.0

# Schenkel / Biegezone
EXCL_FRAC = 0.3         # Anteil nahe der Ecke, der nicht in den Geradenfit eingeht
K_SIGMA = 3.0           # Abweichungsschwelle für die Biegezone
MARGIN = 0.5            # Fitbereich um diesen Anteil der Biegezone erweitern
MIN_BEND_POINTS = 20    # darunter Warnung: Radius unsicher

# Bezier-Grad: höher = flexibler, aber empfindlicher für Messrauschen
BEZIER_DEGREE = 5       # verrauschte Daten: 4-5; 8 neigt zu Ueberschwingern
# ============================================================================


def _guess_delimiter(path):
    """Ermittelt Komma, Semikolon oder Tabulator anhand der Datei."""
    candidates = [",", ";", "\t", "|"]
    sample = []
    with open(path, "r", encoding="utf-8-sig", errors="replace", newline="") as f:
        for line in f:
            if line.strip():
                sample.append(line)
            if len(sample) >= 40:
                break

    if not sample:
        raise ValueError("Die CSV-Datei ist leer.")

    best_delimiter = None
    best_score = (-1, -1)
    for delimiter in candidates:
        widths = []
        for line in sample:
            try:
                widths.append(len(next(csv.reader([line], delimiter=delimiter))))
            except (csv.Error, StopIteration):
                widths.append(1)
        multi = sum(width >= 3 for width in widths)
        median_width = float(np.median(widths))
        score = (multi, median_width)
        if score > best_score:
            best_score = score
            best_delimiter = delimiter

    if best_score[0] == 0:
        preview = "\n".join(repr(line.rstrip("\r\n")) for line in sample[:3])
        raise ValueError(
            "Kein Trennzeichen für mindestens drei Spalten erkannt. "
            "Die ersten Zeilen der Datei sind:\n" + preview
        )
    return best_delimiter


def _number(cell, decimal_comma):
    """Wandelt ein CSV-Feld in float um; unterstützt auch Dezimalkommas."""
    text = str(cell).strip().replace("\ufeff", "")
    if not text:
        return None
    text = text.replace("\u00a0", "").replace(" ", "")
    if decimal_comma and "," in text:
        if "." in text:
            # z. B. 1.234,56 -> 1234.56
            text = text.replace(".", "").replace(",", ".")
        else:
            text = text.replace(",", ".")
    try:
        return float(text)
    except ValueError:
        return None


def _header_columns(row):
    """Sucht X-, Z- und Intensitätsspalten in einer möglichen Kopfzeile."""
    labels = [str(v).strip().lower().replace("\ufeff", "") for v in row]
    cleaned = [re.sub(r"[^a-z0-9äöüß]+", "", v) for v in labels]

    x_idx = next((i for i, s in enumerate(cleaned)
                  if s == "x" or s.startswith("xmm") or "xcoordinate" in s), None)
    z_idx = next((i for i, s in enumerate(cleaned)
                  if s == "z" or s.startswith("zmm") or "zcoordinate" in s), None)
    i_idx = next((i for i, s in enumerate(cleaned)
                  if s in {"i", "intensity", "intensitaet", "intensität", "signal"}
                  or s.startswith("intensity") or s.startswith("intensitaet")
                  or s.startswith("intensität")), None)

    if x_idx is not None and z_idx is not None:
        if i_idx is None and len(row) >= 3:
            remaining = [j for j in range(len(row)) if j not in (x_idx, z_idx)]
            i_idx = remaining[0] if remaining else None
        if i_idx is not None:
            return x_idx, z_idx, i_idx
    return None


def load_profile(path, i_min=I_MIN, x_min=X_MIN, x_max=X_MAX):
    """Lädt X, Z, Intensität aus CSV; erkennt Trennzeichen und Kopfzeile."""
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(
            f"CSV-Datei nicht gefunden:\n{path}\n"
            "Prüfe PROFILE_PATH am Anfang des Skripts."
        )

    delimiter = _guess_delimiter(path)
    with open(path, "r", encoding="utf-8-sig", errors="replace", newline="") as f:
        rows = list(csv.reader(f, delimiter=delimiter))

    # Bei Semikolon-/Tab-getrennten Dateien sind Dezimalkommas üblich.
    decimal_comma = delimiter in (";", "\t")
    columns = (0, 1, 2)  # Standard: erste drei Spalten sind X, Z, Intensität

    # Kopfzeile suchen; Metadatenzeilen davor werden dabei ignoriert.
    header_row_index = None
    for row_index, row in enumerate(rows[:100]):
        found = _header_columns(row)
        if found is not None:
            columns = found
            header_row_index = row_index
            break

    points = []
    intensities = []
    for row_index, row in enumerate(rows):
        if header_row_index is not None and row_index <= header_row_index:
            continue
        if not row or max(columns) >= len(row):
            continue
        values = [_number(row[j], decimal_comma) for j in columns]
        if any(value is None for value in values):
            continue
        x, z, intensity = values
        if np.isfinite(x) and np.isfinite(z) and np.isfinite(intensity):
            points.append((x, z))
            intensities.append(intensity)

    if not points:
        preview = "\n".join(repr(row) for row in rows[:5])
        raise ValueError(
            "Keine numerischen X-, Z- und Intensitätswerte gefunden. "
            f"Erkanntes Trennzeichen: {delimiter!r}.\n"
            "Prüfe, ob die Datei mindestens drei passende Spalten enthält. "
            "Anfang der Datei:\n" + preview
        )

    data = np.column_stack((np.asarray(points, dtype=float),
                            np.asarray(intensities, dtype=float)))
    keep = ((data[:, 2] > i_min)
            & (data[:, 0] >= x_min)
            & (data[:, 0] <= x_max))
    pts = data[keep, :2]
    if len(pts) < 10:
        raise ValueError(
            f"Nach dem Intensitäts- und X-Filter bleiben nur {len(pts)} Punkte. "
            "Prüfe I_MIN, X_MIN und X_MAX."
        )
    return pts[np.argsort(pts[:, 0])]


# ----------------------------------------------------------------------------
# Schenkel (Geraden)
# ----------------------------------------------------------------------------
def fit_line(pts):
    """Total-Least-Squares-Gerade: Schwerpunkt, Richtung, Normale, Residuen."""
    if len(pts) < 2:
        raise ValueError("Zu wenige Punkte für einen Geradenfit.")
    c = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
    d, n = vt[0], vt[1]
    return c, d, n, (pts - c) @ n


def fit_leg(side, apex, excl_frac=EXCL_FRAC, k=K_SIGMA):
    """Fit einer Schenkelgeraden; Richtung zeigt von der Ecke weg."""
    if len(side) < 5:
        raise ValueError("Zu wenige Punkte an einem Schenkel für den Geradenfit.")
    r = np.linalg.norm(side - apex, axis=1)
    outer = side[r > excl_frac * r.max()]
    if len(outer) < 3:
        raise ValueError("Zu wenige äußere Schenkelpunkte für den Geradenfit.")

    c, d, n, res = fit_line(outer)
    sigma0 = float(res.std())
    keep = np.abs(res) <= max(k * sigma0, np.finfo(float).eps)
    outer = outer[keep]
    if len(outer) < 3:
        raise ValueError("Ausreißerfilter lässt zu wenige Schenkelpunkte übrig.")
    c, d, n, res = fit_line(outer)
    sigma = float(res.std())
    if (c - apex) @ d < 0:
        d = -d

    off = np.abs((side - c) @ n) > max(k * sigma, np.finfo(float).eps)
    # Zahl der Punkte vom freien Ende bis zum Beginn des Bogens
    n_leg = len(side) - int(np.argmin(off[::-1]))
    return c, d, sigma, n_leg


def intersect(c1, d1, c2, d2):
    """Schnittpunkt zweier Geraden c + s*d."""
    matrix = np.column_stack([d1, -d2])
    if abs(np.linalg.det(matrix)) < 1e-10:
        raise ValueError("Die Schenkelgeraden sind nahezu parallel.")
    s = np.linalg.solve(matrix, c2 - c1)
    return c1 + s[0] * d1


# ----------------------------------------------------------------------------
# Bezierkurve vom Grad n
# ----------------------------------------------------------------------------
def bernstein(n, t):
    """Bernstein-Basis vom Grad n an den Stellen t."""
    i = np.arange(n + 1)
    binom = np.array([comb(n, j) for j in i], dtype=float)
    return binom * t[:, None] ** i * (1 - t[:, None]) ** (n - i)


def bezier(ctrl, t):
    """Bezierkurve sowie erste und zweite Ableitung."""
    n = len(ctrl) - 1
    t = np.asarray(t, dtype=float)
    B = bernstein(n, t) @ ctrl
    dB = n * bernstein(n - 1, t) @ np.diff(ctrl, axis=0)
    ddB = n * (n - 1) * bernstein(n - 2, t) @ np.diff(ctrl, 2, axis=0)
    return B, dB, ddB


def curvature(ctrl, t):
    """Krümmung der Bezierkurve."""
    _, dB, ddB = bezier(ctrl, t)
    speed = np.linalg.norm(dB, axis=1)
    denominator = np.maximum(speed, np.finfo(float).eps) ** 3
    return np.abs(dB[:, 0] * ddB[:, 1] - dB[:, 1] * ddB[:, 0]) / denominator


def project(ctrl, Q, t, steps=5):
    """Verbessert Kurvenparameter t durch Projektion der Messpunkte."""
    for _ in range(steps):
        B, dB, ddB = bezier(ctrl, t)
        r = B - Q
        f = np.sum(r * dB, axis=1)
        df = np.sum(dB * dB, axis=1) + np.sum(r * ddB, axis=1)
        safe_df = np.where(np.abs(df) > 1e-12, df, 1.0)
        t = np.clip(t - f / safe_df, 0, 1)
    return t


def fit_bend_bezier(Q, corner, d1, d2, n=BEZIER_DEGREE, n_iter=300, tol=1e-12):
    """Fit einer Bezierkurve, deren Endkontrollpunkte auf den Schenkeln liegen."""
    if n < 3:
        raise ValueError("BEZIER_DEGREE muss mindestens 3 sein.")
    if len(Q) < n + 1:
        raise ValueError(
            f"Für einen Beziergrad von {n} sind mindestens {n + 1} Fitpunkte nötig; "
            f"vorhanden: {len(Q)}. Verringere BEZIER_DEGREE oder erweitere den Fitbereich."
        )
    chord = np.r_[0, np.cumsum(np.linalg.norm(np.diff(Q, axis=0), axis=1))]
    if chord[-1] <= 0:
        raise ValueError("Die Fitpunkte haben keine messbare Ausdehnung.")
    t = chord / chord[-1]
    rhs = (Q - corner).ravel()
    ex, ey = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    rms_prev = np.inf

    for _ in range(n_iter):
        W = bernstein(n, t)
        cols = [W[:, 0, None] * d1, W[:, 1, None] * d1,
                W[:, n - 1, None] * d2, W[:, n, None] * d2]
        for i in range(2, n - 1):
            cols.extend([W[:, i, None] * ex, W[:, i, None] * ey])
        A = np.stack(cols, axis=2).reshape(-1, len(cols))
        x = np.linalg.lstsq(A, rhs, rcond=None)[0]

        ctrl = np.empty((n + 1, 2))
        ctrl[0], ctrl[1] = corner + x[0] * d1, corner + x[1] * d1
        ctrl[n - 1], ctrl[n] = corner + x[2] * d2, corner + x[3] * d2
        ctrl[2:n - 1] = corner + x[4:].reshape(-1, 2)

        t = project(ctrl, Q, t)
        rms = float(np.sqrt(np.mean(np.sum((bezier(ctrl, t)[0] - Q) ** 2, axis=1))))
        if abs(rms_prev - rms) < tol:
            break
        rms_prev = rms
    return ctrl, t, rms


# ----------------------------------------------------------------------------
# Gesamtauswertung
# ----------------------------------------------------------------------------
@dataclass
class BendResult:
    opening_angle: float
    bend_angle: float
    r_lokal: float
    r_ersatz: float
    corner: np.ndarray
    apex: np.ndarray
    r_lokal_at: np.ndarray
    legs: tuple
    ctrl: np.ndarray
    fit_points: np.ndarray
    n_bend: int
    sigma_legs: tuple
    rms_bezier: float


def measure_bend(pts):
    if len(pts) < 20:
        raise ValueError("Zu wenige Messpunkte für eine zuverlässige Auswertung.")

    # Grobe Ecke: größter Abstand zur Sehne zwischen erstem und letztem Punkt
    a, b = pts[0], pts[-1]
    chord = b - a
    chord_len = np.linalg.norm(chord)
    if chord_len <= 0:
        raise ValueError("Das Profil besitzt keine messbare Länge.")
    u = chord / chord_len
    i_apex = int(np.argmax(np.abs((pts - a) @ np.array([-u[1], u[0]]))))
    apex = pts[i_apex]
    if i_apex < 5 or len(pts) - i_apex < 5:
        raise ValueError("Die grob erkannte Ecke liegt zu nah an einem Profilende.")

    c1, d1, s1, n1 = fit_leg(pts[:i_apex + 1], apex)
    c2, d2, s2, n2 = fit_leg(pts[i_apex:][::-1], apex)

    cos_open = float(np.clip(d1 @ d2, -1, 1))
    if cos_open < -0.9998:
        raise ValueError("Schenkel nahezu parallel - keine Biegung im Profil gefunden.")
    corner = intersect(c1, d1, c2, d2)

    n_bend = len(pts) - n1 - n2
    if n_bend < 5:
        raise ValueError(f"Nur {n_bend} Punkte in der Biegezone - zu wenig für den Fit.")
    if n_bend < MIN_BEND_POINTS:
        print(f"WARNUNG: nur {n_bend} Punkte in der Biegezone, Radius unsicher.")

    margin_points = int(MARGIN * n_bend)
    q_start = max(n1 - margin_points, 0)
    q_end = min(len(pts) - n2 + margin_points, len(pts))
    Q = pts[q_start:q_end]
    ctrl, t_fit, rms_bez = fit_bend_bezier(Q, corner, d1, d2)

    # Robuster Nachfit: Ausreißer (z. B. Mehrfachreflexion im Radius) entfernen
    for _ in range(3):
        resid = np.linalg.norm(bezier(ctrl, t_fit)[0] - Q, axis=1)
        mad = 1.4826 * np.median(np.abs(resid - np.median(resid)))
        limit = np.median(resid) + K_SIGMA * max(mad, 1e-6)
        good = resid <= limit
        if good.all() or good.sum() < BEZIER_DEGREE + 1:
            break
        Q = Q[good]
        ctrl, t_fit, rms_bez = fit_bend_bezier(Q, corner, d1, d2)

    tt = np.linspace(0, 1, 4001)
    B = bezier(ctrl, tt)[0]
    covered = (tt >= t_fit.min()) & (tt <= t_fit.max())
    if not np.any(covered):
        raise ValueError("Der Messbereich überdeckt keinen Teil der angepassten Kurve.")
    covered_B = B[covered]
    kappa = curvature(ctrl, tt[covered])
    i_k = int(np.argmax(kappa))
    if kappa[i_k] <= 0:
        raise ValueError("Die Kurvenkrümmung konnte nicht zuverlässig bestimmt werden.")

    # Scheitel = Schnitt der Kurve mit der Winkelhalbierenden
    bisector = d1 + d2
    bisector_norm = np.linalg.norm(bisector)
    if bisector_norm <= 1e-12:
        raise ValueError("Die Winkelhalbierende ist nicht bestimmbar.")
    w = bisector / bisector_norm
    apex_b = B[np.argmin(np.abs((B - corner) @ np.array([-w[1], w[0]])))]
    sin_half = np.sqrt((1 - cos_open) / 2)
    opening = float(np.degrees(np.arccos(cos_open)))
    denom = 1 - sin_half
    if denom <= 1e-12:
        raise ValueError("Der Ersatzradius ist bei diesem Winkel nicht bestimmbar.")

    return BendResult(
        opening_angle=opening,
        bend_angle=180.0 - opening,
        r_lokal=1.0 / float(kappa[i_k]),
        r_ersatz=float(np.linalg.norm(apex_b - corner) * sin_half / denom),
        corner=corner,
        apex=apex_b,
        r_lokal_at=covered_B[i_k],
        legs=((c1, d1), (c2, d2)),
        ctrl=ctrl,
        fit_points=Q,
        n_bend=n_bend,
        sigma_legs=(s1, s2),
        rms_bezier=rms_bez,
    )


# ----------------------------------------------------------------------------
# Mehrere Biegungen
# ----------------------------------------------------------------------------
def _chord_deviation(pts, i0, i1):
    """Index und Betrag der größten Abweichung von der Sehne pts[i0]-pts[i1]."""
    if i1 - i0 < 3:
        return None, 0.0
    a, b = pts[i0], pts[i1]
    chord = b - a
    length = np.linalg.norm(chord)
    if length <= 0:
        return None, 0.0
    normal = np.array([-chord[1], chord[0]]) / length
    dist = np.abs((pts[i0 + 1:i1] - a) @ normal)
    k = int(np.argmax(dist))
    return i0 + 1 + k, float(dist[k])


def _median_smooth(pts, window=SMOOTH_WINDOW):
    """Medianfilter auf Z (nur für die Eckensuche), unterdrückt Ausreißer."""
    if window < 3 or len(pts) < window:
        return pts
    h = window // 2
    z = pts[:, 1]
    zp = np.pad(z, h, mode="edge")
    win = np.lib.stride_tricks.sliding_window_view(zp, window)
    out = pts.copy()
    out[:, 1] = np.median(win, axis=1)
    return out


def find_corners(pts, n_bends=N_BENDS, min_leg=MIN_LEG_LENGTH,
                 min_dev=MIN_DEVIATION):
    """Findet grobe Ecken durch wiederholtes Teilen an der größten
    Sehnenabweichung.

    n_bends = Zahl -> genau so viele Ecken.
    n_bends = None -> so lange teilen, wie die größte Abweichung > min_dev.
    Liefert die Indizes aufsteigend sortiert."""
    sm = _median_smooth(pts)
    breaks = [0, len(pts) - 1]
    limit = MAX_BENDS if n_bends is None else n_bends
    while len(breaks) - 2 < limit:
        best = (None, 0.0)
        for i0, i1 in zip(breaks[:-1], breaks[1:]):
            k, dev = _chord_deviation(sm, i0, i1)
            if k is None:
                continue
            # Ecke darf nicht zu nah an einer bestehenden Ecke/einem Ende liegen
            if (np.linalg.norm(sm[k] - sm[i0]) < min_leg
                    or np.linalg.norm(sm[k] - sm[i1]) < min_leg):
                continue
            if dev > best[1]:
                best = (k, dev)
        if n_bends is None:
            if best[0] is None or best[1] < min_dev:
                break
        elif best[0] is None:
            raise ValueError(
                f"Nur {len(breaks) - 2} von {n_bends} Biegungen gefunden. "
                "Prüfe N_BENDS, MIN_LEG_LENGTH sowie X_MIN/X_MAX."
            )
        breaks = sorted(breaks + [best[0]])
    if len(breaks) == 2:
        raise ValueError(
            "Keine Biegung erkannt. Ggf. MIN_DEVIATION verkleinern oder "
            "X_MIN/X_MAX prüfen."
        )
    return breaks[1:-1]


def _measure_at(pts, corners):
    """Wertet die Biegungen an den gegebenen Ecken aus; jeder Abschnitt reicht
    von der Mitte des vorherigen bis zur Mitte des folgenden Schenkels."""
    cuts = [0]
    for k0, k1 in zip(corners[:-1], corners[1:]):
        cuts.append((k0 + k1) // 2)
    cuts.append(len(pts))
    results, errors = [], []
    for i, k in enumerate(corners):
        sub = pts[cuts[i]:cuts[i + 1] + 1]
        try:
            results.append(measure_bend(sub))
            errors.append(None)
        except ValueError as exc:
            results.append(None)
            errors.append(f"Biegung {i + 1} (bei X = {pts[k, 0]:.2f} mm): {exc}")
    return results, errors


def measure_bends(pts, n_bends=N_BENDS):
    """Feste Anzahl: genau n_bends auswerten.
    Automatisch (None): Kandidaten suchen, Scheinbiegungen (zu kleiner Winkel
    oder nicht auswertbar) entfernen und neu auswerten, bis alle gültig sind."""
    corners = find_corners(pts, n_bends)
    if n_bends is not None:
        results, errors = _measure_at(pts, corners)
        for e in errors:
            if e:
                raise ValueError(e)
        return results

    while corners:
        results, errors = _measure_at(pts, corners)
        # schwächsten ungültigen Kandidaten entfernen
        bad = [i for i, (r, e) in enumerate(zip(results, errors))
               if e or r.bend_angle < MIN_BEND_ANGLE]
        if not bad:
            print(f"Automatisch erkannt: {len(results)} Biegung(en)")
            return results
        worst = min(bad, key=lambda i: -1.0 if results[i] is None
                    else results[i].bend_angle)
        reason = errors[worst] or (f"Winkel {results[worst].bend_angle:.2f} Grad "
                                   f"< MIN_BEND_ANGLE")
        print(f"Kandidat bei X = {pts[corners[worst], 0]:.2f} mm verworfen ({reason})")
        corners = corners[:worst] + corners[worst + 1:]
    raise ValueError("Keine gültige Biegung gefunden.")


def report(res):
    print(f"Innenwinkel     : {res.opening_angle:8.3f} Grad")
    print(f"Biegewinkel     : {res.bend_angle:8.3f} Grad")
    print(f"R_lokal         : {res.r_lokal:8.3f} mm   (kleinster lokaler Radius)")
    print(f"R_ersatz        : {res.r_ersatz:8.3f} mm   (Kreis mit gleichem Scheitelabstand)")
    print(f"Punkte Biegezone: {res.n_bend:8d}   (Fit mit Rand: {len(res.fit_points)})")
    print(f"Sigma Schenkel  : {res.sigma_legs[0]:8.4f} / {res.sigma_legs[1]:.4f} mm")
    print(f"RMS Bezier      : {res.rms_bezier:8.4f} mm")


def plot(pts, results):
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("Für die Grafik bitte installieren: python -m pip install matplotlib") from exc

    n = len(results)
    fig = plt.figure(figsize=(max(8, 5 * n), 10))
    grid = fig.add_gridspec(3, n, height_ratios=[1.3, 1.3, 1])
    ax_all = fig.add_subplot(grid[0, :])
    ax_all.plot(pts[:, 0], pts[:, 1], ".", ms=2, color="0.6", label="Messpunkte")
    tt = np.linspace(0, 1, 500)

    for j, res in enumerate(results):
        color = f"C{j}"
        B = bezier(res.ctrl, tt)[0]
        ax_all.plot(B[:, 0], B[:, 1], "-", color=color, lw=2,
                    label=f"Biegung {j + 1}: {res.bend_angle:.2f} Grad")
        ax_all.plot(*res.r_lokal_at, "m*", ms=10)

        # Detailansicht
        ax = fig.add_subplot(grid[1, j])
        ax.plot(pts[:, 0], pts[:, 1], ".", ms=2, color="0.6")
        ax.plot(res.fit_points[:, 0], res.fit_points[:, 1], ".", ms=3,
                color="tab:orange", label="Fitbereich")
        for c, d in res.legs:
            seg = res.corner + np.outer([0, 6 * max(res.r_ersatz, 1.0)], d)
            ax.plot(seg[:, 0], seg[:, 1], "r-", lw=1)
        ax.plot(B[:, 0], B[:, 1], "b-", lw=2, label=f"Bezier Grad {len(res.ctrl) - 1}")
        ax.plot(res.ctrl[:, 0], res.ctrl[:, 1], "bs--", lw=0.5, ms=3)
        ax.plot(*np.c_[res.corner, res.apex], "g-", lw=1, label="Scheitelabstand h")
        ax.plot(*res.r_lokal_at, "m*", ms=12, label="Max. Krümmung")
        span = 1.5 * max(res.r_ersatz, 1.0)
        ax.set_xlim(res.corner[0] - span, res.corner[0] + span)
        ax.set_ylim(res.corner[1] - span, res.corner[1] + span)
        ax.set_aspect("equal")
        ax.set_title(f"Biegung {j + 1}: R_lokal {res.r_lokal:.2f} / "
                     f"R_ersatz {res.r_ersatz:.2f} mm", fontsize="medium")
        ax.legend(fontsize="x-small")
        ax.grid(True)

        axk = fig.add_subplot(grid[2, j])
        arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(B, axis=0), axis=1))]
        axk.plot(arc, curvature(res.ctrl, tt), "b-")
        axk.axhline(1 / res.r_ersatz, color="g", ls="--", label="1 / R_ersatz")
        axk.set_xlabel("Bogenlänge [mm]")
        axk.set_ylabel("Krümmung [1/mm]")
        axk.legend(fontsize="x-small")
        axk.grid(True)

    ax_all.set_aspect("equal")
    ax_all.set_xlabel("X [mm]")
    ax_all.set_ylabel("Z [mm]")
    ax_all.set_title("Gesamtprofil")
    ax_all.legend(fontsize="small")
    ax_all.grid(True)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    try:
        points = load_profile(PROFILE_PATH)
        print(f"CSV geladen: {PROFILE_PATH}")
        print(f"Verwendete Messpunkte: {len(points)}")
        results = measure_bends(points, N_BENDS)
        for j, result in enumerate(results, start=1):
            print(f"\n=== Biegung {j}  (Ecke bei X = {result.corner[0]:.3f} mm, "
                  f"Z = {result.corner[1]:.3f} mm) ===")
            report(result)
        if len(results) >= 2:
            print("\n=== Abstände der Biegungen (Schnittpunkte der Schenkel) ===")
            for j in range(len(results) - 1):
                dist = np.linalg.norm(results[j + 1].corner - results[j].corner)
                print(f"Ecke {j + 1} -> Ecke {j + 2}: {dist:8.3f} mm")
        if PLOT:
            plot(points, results)
    except Exception as exc:
        raise SystemExit(f"FEHLER: {exc}") from exc
