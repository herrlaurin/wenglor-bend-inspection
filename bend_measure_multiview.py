#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Robuste Biegeauswertung aus Einzelprofilen oder registrierten Mehrprofil-CSV.

Winkel, Biegezentrum und sichtbarer Bogenradius werden aus den Rohpunkten
robust geschätzt, sofern Schenkel und genügend Bogenpunkte erkennbar sind.
"""

from __future__ import annotations

import argparse
import csv
import io
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class ScanData:
    profiles: dict[int, np.ndarray]
    angles: dict[int, float]
    center: np.ndarray | None
    thickness_mm: float | None
    expected_profiles: int | None
    sensor_positions: dict[int, np.ndarray] = field(default_factory=dict)
    center_source: str = "unbekannt"
    global_coordinates: bool = False


@dataclass
class LineGroup:
    direction: np.ndarray
    points: np.ndarray
    angle_mod_180: float
    span_mm: float
    support: int


@dataclass
class CircleFit:
    center: np.ndarray
    radius: float
    points: np.ndarray
    rms_mm: float
    coverage_deg: float


class NoBendDetected(RuntimeError):
    pass


def _norm_header(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def fit_circle_center(points: np.ndarray) -> tuple[np.ndarray | None, float | None, float | None]:
    """Algebraischer Kreisfit für bekannte Sensorpositionen auf einer Bahn."""
    q = np.unique(np.round(np.asarray(points, dtype=float), 4), axis=0)
    if len(q) < 3:
        return None, None, None
    A = np.column_stack((2.0 * q[:, 0], 2.0 * q[:, 1], np.ones(len(q))))
    b = np.sum(q * q, axis=1)
    try:
        coeff, *_ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return None, None, None
    center = coeff[:2]
    radius2 = float(coeff[2] + center @ center)
    if radius2 <= 0 or not np.all(np.isfinite(center)):
        return None, None, None
    radius = math.sqrt(radius2)
    residual = np.abs(np.linalg.norm(q - center, axis=1) - radius)
    rms = float(np.sqrt(np.mean(residual ** 2)))
    theta = np.sort(np.degrees(np.arctan2(q[:, 1] - center[1], q[:, 0] - center[0])) % 360.0)
    coverage = 360.0 - float(np.max(np.diff(np.r_[theta, theta[0] + 360.0])))
    if coverage < 8.0:
        return None, radius, rms
    if rms > max(0.10, radius * 0.002):
        return None, radius, rms
    return center, radius, rms


def read_scan(path: Path, center_x: float | None = None,
              center_z: float | None = None,
              thickness_override: float | None = None,
              i_min: float = 1.0,
              ignore_center_metadata: bool = False,
              assume_global_coordinates: bool = False) -> ScanData:
    """Liest CSV-Punkte, Profilnummern und Scanmetadaten."""
    if not path.is_file():
        raise FileNotFoundError(f"CSV nicht gefunden: {path}")
    comments, rows = [], []
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as f:
        for line in f:
            if line.lstrip().startswith("#"):
                comments.append(line.strip())
            elif line.strip():
                rows.append(line)
    if not rows:
        raise ValueError("Die CSV enthält keine Tabellenzeilen.")

    try:
        dialect = csv.Sniffer().sniff("".join(rows[:5]), delimiters=",;\t")
        delimiter = dialect.delimiter
    except csv.Error:
        delimiter = ","
    reader = csv.DictReader(io.StringIO("".join(rows)), delimiter=delimiter)
    if not reader.fieldnames:
        raise ValueError("CSV-Kopfzeile fehlt.")
    names = {_norm_header(name): name for name in reader.fieldnames}
    x_col = next((names[k] for k in names if k == "x" or k.startswith("xmm")), None)
    z_col = next((names[k] for k in names if k == "z" or k.startswith("zmm")), None)
    i_col = next((names[k] for k in names if k.startswith("intensity") or k in {"i", "signal"}), None)
    p_col = next((names[k] for k in names if k in {"profile", "profileid", "profil", "profilnummer"}), None)
    a_col = next((names[k] for k in names if "sensorangle" in k or "scanangle" in k), None)
    sx_col = next((names[k] for k in names if k.startswith("sensorx")), None)
    sz_col = next((names[k] for k in names if k.startswith("sensorz")), None)
    if not (x_col and z_col and i_col):
        raise ValueError("Benötigte Spalten X, Z und Intensity wurden nicht gefunden.")

    grouped: dict[int, list[tuple[float, float]]] = {}
    angles: dict[int, float] = {}
    sensor_positions: dict[int, np.ndarray] = {}
    malformed = 0
    for row in reader:
        try:
            x, z, intensity = float(row[x_col]), float(row[z_col]), float(row[i_col])
            profile = int(float(row[p_col])) if p_col and row.get(p_col) else 1
            angle = float(row[a_col]) if a_col and row.get(a_col) else math.nan
            sx = float(row[sx_col]) if sx_col and row.get(sx_col) else math.nan
            sz = float(row[sz_col]) if sz_col and row.get(sz_col) else math.nan
        except (TypeError, ValueError):
            malformed += 1
            continue
        if not (math.isfinite(x) and math.isfinite(z) and math.isfinite(intensity)):
            malformed += 1
            continue
        if math.isfinite(sx) and math.isfinite(sz):
            sensor_positions.setdefault(profile, np.array([sx, sz], dtype=float))
        if intensity <= i_min:
            continue
        grouped.setdefault(profile, []).append((x, z))
        if math.isfinite(angle):
            angles[profile] = angle
    if not grouped:
        raise ValueError("Keine gültigen Messpunkte nach dem Intensitätsfilter.")
    profiles = {pid: np.unique(np.round(np.asarray(points), 3), axis=0)
                for pid, points in grouped.items()}

    text = "\n".join(comments)
    m = re.search(r"Bend center X=([-+0-9.eE]+) mm, Z=([-+0-9.eE]+) mm", text)
    metadata_center = np.array([float(m.group(1)), float(m.group(2))]) if m else None
    global_coordinates = assume_global_coordinates or bool(
        re.search(r"Coordinates:\s*global X/Z", text, flags=re.IGNORECASE)
    )
    sensor_center, orbit_radius, orbit_rms = fit_circle_center(
        np.array(list(sensor_positions.values())) if sensor_positions else np.empty((0, 2))
    )
    if sensor_positions and sensor_center is None:
        if orbit_radius is not None and orbit_rms is not None:
            print(f"Warnung: Sensorbahnzentrum nicht übernommen (Scanbogen zu schmal oder Fit unsicher; "
                  f"Kreisfit R={orbit_radius:.3f} mm, RMS={orbit_rms:.4f} mm).")
        else:
            print("Warnung: Sensorbahnzentrum nicht schätzbar; mindestens drei verschiedene Sensorpositionen nötig.")
    if ignore_center_metadata:
        file_center = None
        center_source = "ignoriert; nicht aus Sensor-Drehpunkt abgeleitet"
    elif metadata_center is not None:
        file_center = metadata_center
        center_source = "Biegezentrum aus CSV-Metadaten"
    else:
        file_center = None
        center_source = "unbekannt; Sensor-Drehpunkt ist kein Biegezentrum"
    if center_x is not None or center_z is not None:
        if file_center is None:
            file_center = np.array([0.0, 0.0])
        else:
            file_center = file_center.copy()
        if center_x is not None:
            file_center[0] = center_x
        if center_z is not None:
            file_center[1] = center_z
        center_source = "manuell vorgegeben/überschrieben"
    if metadata_center is not None and sensor_center is not None:
        separation = float(np.linalg.norm(metadata_center - sensor_center))
        print(f"Abstand Sensor-Drehpunkt zu Biegezentrum: {separation:.3f} mm (CSV-Metadaten).")
    if sensor_center is not None:
        print(f"Sensorbahn-Fit: Zentrum ({sensor_center[0]:.3f}, {sensor_center[1]:.3f}) mm; "
              f"Radius {orbit_radius:.3f} mm; RMS {orbit_rms:.4f} mm.")
    tm = re.search(r"Dicke=([-+0-9.eE]+) mm", text)
    file_thickness = float(tm.group(1)) if tm else None
    thickness = thickness_override if thickness_override is not None else file_thickness
    pm = re.search(r"Kreisbahnscan mit (\d+) Profilen", text)
    expected = int(pm.group(1)) if pm else (max(profiles) if p_col else 1)
    if malformed:
        print(f"Hinweis: {malformed} fehlerhafte CSV-Zeile(n) übersprungen.")
    return ScanData(profiles, angles, file_center, thickness, expected,
                    sensor_positions, center_source, global_coordinates)


def _tls(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = points.mean(axis=0)
    _, _, vt = np.linalg.svd(points - center, full_matrices=False)
    direction = vt[0]
    direction /= np.linalg.norm(direction)
    return center, direction


def _best_line(points: np.ndarray, rng: np.random.Generator, tolerance: float,
               min_span: float, iterations: int, min_inliers: int,
               max_sample: int = 6000) -> tuple[np.ndarray, np.ndarray, np.ndarray, float] | None:
    """RANSAC-Kandidatensuche in Batches, danach TLS-Verfeinerung auf allen Punkten."""
    if len(points) < min_inliers:
        return None
    if len(points) > max_sample:
        sample = points[rng.choice(len(points), max_sample, replace=False)]
    else:
        sample = points
    best_score = -1.0
    best_p0 = best_d = None
    batch_size = 64
    for start in range(0, iterations, batch_size):
        count = min(batch_size, iterations - start)
        pair_ids = rng.integers(0, len(sample), size=(count, 2))
        p0 = sample[pair_ids[:, 0]]
        vec = sample[pair_ids[:, 1]] - p0
        lengths = np.linalg.norm(vec, axis=1)
        good = lengths >= max(2.0, min_span * 0.6)
        if not np.any(good):
            continue
        p0, vec, lengths = p0[good], vec[good], lengths[good]
        directions = vec / lengths[:, None]
        normals = np.column_stack((-directions[:, 1], directions[:, 0]))
        delta = sample[None, :, :] - p0[:, None, :]
        residual = np.abs(delta[:, :, 0] * normals[:, None, 0]
                          + delta[:, :, 1] * normals[:, None, 1])
        mask = residual <= tolerance
        inlier_counts = mask.sum(axis=1)
        projection = delta[:, :, 0] * directions[:, None, 0] + delta[:, :, 1] * directions[:, None, 1]
        mins = np.min(np.where(mask, projection, np.inf), axis=1)
        maxs = np.max(np.where(mask, projection, -np.inf), axis=1)
        spans = np.maximum(0.0, maxs - mins)
        scores = np.where(inlier_counts >= min_inliers, inlier_counts * spans, -1.0)
        if len(scores) and float(scores.max()) > best_score:
            k = int(np.argmax(scores))
            best_score = float(scores[k])
            best_p0 = p0[k].copy()
            best_d = directions[k].copy()
    if best_d is None or best_score <= 0:
        return None

    # Refit and reselect inliers twice to remove RANSAC sampling error.
    mask = np.zeros(len(points), dtype=bool)
    for _ in range(2):
        n = np.array([-best_d[1], best_d[0]])
        mask = np.abs((points - best_p0) @ n) <= tolerance
        if mask.sum() < min_inliers:
            return None
        center, best_d = _tls(points[mask])
        best_p0 = center
    n = np.array([-best_d[1], best_d[0]])
    mask = np.abs((points - best_p0) @ n) <= tolerance
    if mask.sum() < min_inliers:
        return None
    inliers = points[mask]
    projections = (inliers - best_p0) @ best_d
    span = float(np.ptp(projections))
    if span < min_span:
        return None
    return best_d, inliers, mask, span


def _angle_difference_undirected(a: float, b: float) -> float:
    return abs((a - b + 90.0) % 180.0 - 90.0)


def extract_line_groups(points: np.ndarray, tolerance: float = 0.05,
                        min_span: float = 8.0, iterations: int = 600,
                        max_lines: int = 8, min_inliers: int = 25,
                        seed: int = 2026) -> list[LineGroup]:
    """Findet lange Geraden robust und fasst parallele Blechflächen zusammen."""
    points = np.unique(np.round(np.asarray(points, dtype=float), 3), axis=0)
    remaining = points.copy()
    rng = np.random.default_rng(seed)
    candidates: list[tuple[float, np.ndarray]] = []
    for _ in range(max_lines):
        found = _best_line(remaining, rng, tolerance, min_span,
                           iterations, min_inliers)
        if found is None:
            break
        direction, inliers, mask, span = found
        if len(inliers) < min_inliers or span < min_span:
            break
        angle = math.degrees(math.atan2(direction[1], direction[0])) % 180.0
        candidates.append((angle, inliers))
        remaining = remaining[~mask]
        if len(remaining) < min_inliers:
            break

    grouped: list[dict] = []
    for angle, inliers in candidates:
        group = next((g for g in grouped
                      if _angle_difference_undirected(g["angle"], angle) <= 3.0), None)
        if group is None:
            grouped.append({"angle": angle, "points": inliers.copy()})
        else:
            group["points"] = np.vstack((group["points"], inliers))
    result = []
    for group in grouped:
        q = np.unique(np.round(group["points"], 3), axis=0)
        if len(q) < min_inliers:
            continue
        _, direction = _tls(q)
        angle = math.degrees(math.atan2(direction[1], direction[0])) % 180.0
        projections = (q - q.mean(axis=0)) @ direction
        span = float(np.ptp(projections))
        if span >= min_span:
            result.append(LineGroup(direction, q, angle, span, len(q)))
    return sorted(result, key=lambda g: g.support * g.span_mm, reverse=True)


def _circle_from_three(points: np.ndarray) -> tuple[np.ndarray, float] | None:
    p0, p1, p2 = points
    matrix = 2.0 * np.vstack((p1 - p0, p2 - p0))
    if abs(float(np.linalg.det(matrix))) < 1e-10:
        return None
    rhs = np.array([p1 @ p1 - p0 @ p0, p2 @ p2 - p0 @ p0])
    try:
        center = np.linalg.solve(matrix, rhs)
    except np.linalg.LinAlgError:
        return None
    radius = float(np.linalg.norm(p0 - center))
    if not np.all(np.isfinite(center)) or not math.isfinite(radius) or radius <= 0:
        return None
    return center, radius


def _least_squares_circle(points: np.ndarray) -> tuple[np.ndarray, float] | None:
    if len(points) < 3:
        return None
    x, z = points[:, 0], points[:, 1]
    matrix = np.column_stack((2.0 * x, 2.0 * z, np.ones(len(points))))
    rhs = x * x + z * z
    try:
        coeff, *_ = np.linalg.lstsq(matrix, rhs, rcond=None)
    except np.linalg.LinAlgError:
        return None
    center = coeff[:2]
    radius2 = float(coeff[2] + center @ center)
    if radius2 <= 0 or not np.all(np.isfinite(center)):
        return None
    return center, math.sqrt(radius2)


def _circle_coverage(points: np.ndarray, center: np.ndarray) -> float:
    angles = np.sort(np.degrees(np.arctan2(
        points[:, 1] - center[1], points[:, 0] - center[0]
    )) % 360.0)
    if len(angles) < 2:
        return 0.0
    return 360.0 - float(np.max(np.diff(np.r_[angles, angles[0] + 360.0])))


def _fit_circle_ransac(points: np.ndarray, tolerance: float = 0.05,
                       iterations: int = 400, seed: int = 2026) -> CircleFit | None:
    """Robuster Kreisbogenfit für Rohpunkte nach Entfernen der Schenkellinien."""
    points = np.unique(np.round(np.asarray(points, dtype=float), 3), axis=0)
    if len(points) < 12:
        return None
    rng = np.random.default_rng(seed)
    sample = points if len(points) <= 1800 else points[rng.choice(len(points), 1800, replace=False)]
    radial_tolerance = max(0.06, 2.0 * tolerance)
    min_inliers = max(12, min(30, int(0.08 * len(sample))))
    best_score = -1.0
    best_center = None
    best_radius = None
    best_mask = None

    for _ in range(iterations):
        ids = rng.choice(len(sample), 3, replace=False)
        candidate = _circle_from_three(sample[ids])
        if candidate is None:
            continue
        center, radius = candidate
        if radius > 1000.0:
            continue
        residual = np.abs(np.linalg.norm(sample - center, axis=1) - radius)
        mask = residual <= radial_tolerance
        count = int(mask.sum())
        if count < min_inliers:
            continue
        coverage = _circle_coverage(sample[mask], center)
        if coverage < 8.0:
            continue
        score = count * math.sqrt(coverage)
        if score > best_score:
            best_score, best_center, best_radius, best_mask = score, center, radius, mask

    if best_center is None or best_mask is None:
        return None
    center, radius = best_center, best_radius
    inliers = sample[best_mask]
    for _ in range(3):
        fit = _least_squares_circle(inliers)
        if fit is None:
            return None
        center, radius = fit
        residual = np.abs(np.linalg.norm(points - center, axis=1) - radius)
        inliers = points[residual <= radial_tolerance]
        if len(inliers) < min_inliers:
            return None
    residual = np.abs(np.linalg.norm(inliers - center, axis=1) - radius)
    rms = float(np.sqrt(np.mean(residual ** 2)))
    coverage = _circle_coverage(inliers, center)
    if coverage < 8.0 or rms > max(radial_tolerance, radius * 0.01):
        return None
    return CircleFit(center, radius, inliers, rms, coverage)


def extract_arc_candidate_points(points: np.ndarray, groups: list[LineGroup],
                                tolerance: float) -> np.ndarray:
    """Entfernt die endlichen geraden Schenkelbereiche und liefert Bogenkandidaten."""
    points = np.unique(np.round(np.asarray(points, dtype=float), 3), axis=0)
    arc_mask = np.ones(len(points), dtype=bool)
    line_tolerance = max(0.08, 1.5 * tolerance)
    for group in groups:
        direction = group.direction
        normal = np.array([-direction[1], direction[0]])
        line_projection = group.points @ direction
        normal_projection = group.points @ normal
        point_projection = points @ direction
        point_normal = points @ normal
        in_leg_span = (
            (point_projection >= float(line_projection.min()) - 0.25)
            & (point_projection <= float(line_projection.max()) + 0.25)
        )
        on_sheet_face_band = (
            (point_normal >= float(normal_projection.min()) - line_tolerance)
            & (point_normal <= float(normal_projection.max()) + line_tolerance)
        )
        arc_mask &= ~(in_leg_span & on_sheet_face_band)
    return points[arc_mask]


def estimate_raw_bend_circles(points: np.ndarray, groups: list[LineGroup],
                              tolerance: float = 0.05, iterations: int = 400,
                              seed: int = 2026) -> list[CircleFit]:
    """Findet einen oder zwei Bogenkreise direkt in den Profilpunkten."""
    points = np.unique(np.round(np.asarray(points, dtype=float), 3), axis=0)
    arc_mask = np.ones(len(points), dtype=bool)
    line_tolerance = max(0.08, 1.5 * tolerance)
    for group in groups:
        direction = group.direction
        normal = np.array([-direction[1], direction[0]])
        line_projection = group.points @ direction
        normal_projection = group.points @ normal
        point_projection = points @ direction
        point_normal = points @ normal
        in_leg_span = (
            (point_projection >= float(line_projection.min()) - 0.25)
            & (point_projection <= float(line_projection.max()) + 0.25)
        )
        on_sheet_face_band = (
            (point_normal >= float(normal_projection.min()) - line_tolerance)
            & (point_normal <= float(normal_projection.max()) + line_tolerance)
        )
        arc_mask &= ~(in_leg_span & on_sheet_face_band)
    remaining = points[arc_mask]
    fits: list[CircleFit] = []
    for index in range(2):
        fit = _fit_circle_ransac(remaining, tolerance, iterations, seed + index)
        if fit is None:
            break
        fits.append(fit)
        radial_distance = np.abs(np.linalg.norm(remaining - fit.center, axis=1) - fit.radius)
        remaining = remaining[radial_distance > max(0.06, 2.0 * tolerance)]
    if len(fits) == 2:
        center_gap = float(np.linalg.norm(fits[0].center - fits[1].center))
        concentric_limit = max(0.25, 0.03 * min(fits[0].radius, fits[1].radius))
        if center_gap > concentric_limit:
            fits = [max(fits, key=lambda fit: len(fit.points) * fit.coverage_deg)]
    return fits


def fit_bend_radius_from_tangents(points: np.ndarray, g1: LineGroup, g2: LineGroup,
                                 opening_deg: float, tolerance: float = 0.05
                                 ) -> CircleFit | None:
    """Fit a bend circle with its center constrained to the leg-angle bisector."""
    corner = _intersection_of_midlines(g1, g2)
    if corner is None:
        return None
    try:
        _, d1, d2 = opening_angle(g1, g2, None)
    except NoBendDetected:
        return None
    bisector = d1 + d2
    bisector_norm = float(np.linalg.norm(bisector))
    sine = math.sin(math.radians(opening_deg / 2.0))
    if bisector_norm <= 1e-9 or sine <= 1e-9:
        return None
    bisector /= bisector_norm
    arc_points = extract_arc_candidate_points(points, [g1, g2], tolerance)
    if len(arc_points) < 18:
        return None

    max_radius = max(50.0, 2.0 * max(g1.span_mm, g2.span_mm))
    coarse_radii = np.linspace(0.1, max_radius, 1200)

    def losses_for(radii: np.ndarray) -> np.ndarray:
        centers = corner[None, :] + (radii / sine)[:, None] * bisector[None, :]
        distances = np.linalg.norm(arc_points[None, :, :] - centers[:, None, :], axis=2)
        return np.median(np.abs(distances - radii[:, None]), axis=1)

    coarse_losses = losses_for(coarse_radii)
    best_index = int(np.argmin(coarse_losses))
    step = float(coarse_radii[1] - coarse_radii[0])
    low = max(0.1, coarse_radii[best_index] - 2.0 * step)
    high = min(max_radius, coarse_radii[best_index] + 2.0 * step)
    fine_radii = np.linspace(low, high, 801)
    fine_losses = losses_for(fine_radii)
    radius = float(fine_radii[int(np.argmin(fine_losses))])
    center = corner + (radius / sine) * bisector
    residuals = np.abs(np.linalg.norm(arc_points - center, axis=1) - radius)
    inlier_threshold = max(0.15, 3.0 * tolerance)
    inliers = arc_points[residuals <= inlier_threshold]
    if len(inliers) < 15:
        return None
    signed_residuals = np.linalg.norm(inliers - center, axis=1) - radius
    rms = float(np.sqrt(np.mean(signed_residuals ** 2)))
    coverage = _circle_coverage(inliers, center)
    max_coverage = (180.0 - opening_deg) + max(8.0, 0.25 * (180.0 - opening_deg))
    if coverage < 8.0 or coverage > max_coverage or rms > inlier_threshold:
        return None
    return CircleFit(center, radius, inliers, rms, coverage)


def validate_bend_circle_fits(fits: list[CircleFit], g1: LineGroup, g2: LineGroup,
                              opening_deg: float) -> list[CircleFit]:
    """Verwirft Kreise, die nicht zur Biegeecke und zum Biegebogen passen."""
    corner = _intersection_of_midlines(g1, g2)
    if corner is None:
        return []
    bend_deg = 180.0 - opening_deg
    accepted = []
    sine = math.sin(math.radians(opening_deg / 2.0))
    if sine <= 0:
        return []
    for fit in fits:
        expected_center_distance = fit.radius / sine
        actual_center_distance = float(np.linalg.norm(fit.center - corner))
        center_tolerance = max(0.5, 0.15 * expected_center_distance)
        max_coverage = bend_deg + max(8.0, 0.25 * bend_deg)
        if (abs(actual_center_distance - expected_center_distance) <= center_tolerance
                and fit.coverage_deg <= max_coverage):
            accepted.append(fit)
    return accepted


def _intersection_of_midlines(g1: LineGroup, g2: LineGroup) -> np.ndarray | None:
    d1, d2 = g1.direction, g2.direction
    n1, n2 = np.array([-d1[1], d1[0]]), np.array([-d2[1], d2[0]])
    q1, q2 = np.median(g1.points @ n1), np.median(g2.points @ n2)
    matrix = np.vstack((n1, n2))
    if abs(np.linalg.det(matrix)) < 1e-8:
        return None
    return np.linalg.solve(matrix, np.array([q1, q2]))


def opening_angle(g1: LineGroup, g2: LineGroup,
                 center: np.ndarray | None) -> tuple[float, np.ndarray, np.ndarray]:
    """Winkelt die beiden Geraden nach außen vom Biegebereich weg."""
    directions = []
    if center is None:
        corner = _intersection_of_midlines(g1, g2)
        if corner is None:
            raise NoBendDetected("Schenkelgeraden sind nahezu parallel; Ecke nicht bestimmbar.")
    for group in (g1, g2):
        d = group.direction.copy()
        n = np.array([-d[1], d[0]])
        q = float(np.median(group.points @ n))
        if center is not None:
            foot = center + (q - float(center @ n)) * n
        else:
            foot = corner
        side = float(np.median((group.points - foot) @ d))
        if abs(side) < max(0.05, 0.02 * group.span_mm):
            raise NoBendDetected("Geradenrichtung zur Biegezone ist mehrdeutig.")
        if side < 0:
            d = -d
        directions.append(d)
    cosang = float(np.clip(np.dot(directions[0], directions[1]), -1.0, 1.0))
    angle = math.degrees(math.acos(cosang))
    return angle, directions[0], directions[1]


def estimate_arc_radius(points: np.ndarray, center: np.ndarray | None,
                        thickness: float | None) -> tuple[float | None, float | None, str]:
    """Schätzt innere/äußere Bogenradien aus radialen Dichtemoden."""
    if center is None or len(points) < 100:
        return None, None, "Kreismittelpunkt oder genügend Bogenpunkte fehlen"
    points = np.unique(np.round(points, 3), axis=0)
    r = np.linalg.norm(points - center, axis=1)
    r = r[np.isfinite(r) & (r > 0.1)]
    if len(r) < 100:
        return None, None, "zu wenige radiale Messpunkte"
    width = 0.02
    upper = min(float(np.percentile(r, 99.5)), 100.0)
    edges = np.arange(0.0, upper + width, width)
    hist, _ = np.histogram(r, edges)
    smooth = np.convolve(hist.astype(float), np.array([1, 2, 3, 2, 1]) / 9.0, mode="same")
    peaks = [i for i in range(1, len(smooth) - 1)
             if smooth[i] >= smooth[i - 1] and smooth[i] > smooth[i + 1]]
    peaks.sort(key=lambda i: smooth[i], reverse=True)
    candidates: list[tuple[float, float, int]] = []
    min_peak = max(12, int(0.002 * len(r)))
    for i in peaks:
        if hist[i] < min_peak:
            continue
        nominal = 0.5 * (edges[i] + edges[i + 1])
        if any(abs(nominal - x[0]) < 0.35 for x in candidates):
            continue
        band = r[np.abs(r - nominal) <= 0.10]
        if len(band) >= min_peak:
            candidates.append((float(np.median(band)), float(smooth[i]), len(band)))
        if len(candidates) >= 8:
            break
    if not candidates:
        return None, None, "kein klarer Bogenradius-Peak gefunden"
    if thickness is not None and thickness > 0:
        pairs = []
        for i, a in enumerate(candidates):
            for b in candidates[i + 1:]:
                separation = abs(a[0] - b[0])
                error = abs(separation - thickness)
                if error <= max(0.30, 0.15 * thickness):
                    pairs.append((error, -(a[1] + b[1]), min(a[0], b[0]), max(a[0], b[0])))
        if pairs:
            pairs.sort()
            _, _, inner, outer = pairs[0]
            return inner, outer, f"zwei konzentrische Flächen; Abstand {outer-inner:.3f} mm"
    strongest = max(candidates, key=lambda x: x[1])
    return strongest[0], None, "nur eine Bogenfläche sicher sichtbar; Innen-/Außenradius nicht unterscheidbar"


def profile_diagnostics(profiles: dict[int, np.ndarray], tolerance: float,
                        min_span: float, iterations: int) -> dict[int, str]:
    statuses = {}
    for pid, points in sorted(profiles.items()):
        if len(points) < 25:
            statuses[pid] = f"nicht auswertbar: nur {len(points)} Treffer"
            continue
        groups = extract_line_groups(points, tolerance=tolerance, min_span=min_span,
                                     iterations=iterations, max_lines=4,
                                     min_inliers=max(15, min(25, len(points) // 20)),
                                     seed=2026 + pid)
        if len(groups) >= 2 and _angle_difference_undirected(
                groups[0].angle_mod_180, groups[1].angle_mod_180) >= 3.0:
            try:
                angle, _, _ = opening_angle(groups[0], groups[1], None)
                statuses[pid] = f"Einzelprofil verwertbar (Öffnung ca. {angle:.1f}°)"
            except NoBendDetected as exc:
                statuses[pid] = f"nicht einzeln auswertbar: {exc}"
        elif len(groups) == 1:
            statuses[pid] = "nicht einzeln auswertbar: nur ein Schenkel sichtbar"
        else:
            statuses[pid] = "nicht einzeln auswertbar: keine ausreichend langen Schenkel"
    return statuses


def summarize_raw_radii(estimates: list[tuple[int, CircleFit]],
                        thickness: float | None = None
                        ) -> tuple[float | None, float | None, str]:
    """Fasst Rohdaten-Radien robust zusammen; kleinere Kreise sind Innenradien."""
    if not estimates:
        return None, None, "Kein ausreichend großer, gekrümmter Bogen in den Rohdaten erkannt"
    ordered = sorted(estimates, key=lambda item: item[1].radius)
    radii = np.array([fit.radius for _, fit in ordered], dtype=float)
    if len(radii) >= 2:
        gaps = np.diff(radii)
        split = int(np.argmax(gaps))
        gap = float(gaps[split])
        reference = float(np.median(radii))
        threshold = max(0.35, 0.08 * reference,
                        0.25 * thickness if thickness is not None else 0.0)
        low_count, high_count = split + 1, len(radii) - split - 1
        same_profile_pair = (
            len(ordered) == 2 and ordered[0][0] == ordered[1][0]
        )
        if gap > threshold and ((low_count >= 2 and high_count >= 2) or same_profile_pair):
            inner = float(np.median(radii[:split + 1]))
            outer = float(np.median(radii[split + 1:]))
            return inner, outer, f"Innen- und Außenbogen aus Rohdaten erkannt; Abstand {outer-inner:.3f} mm"
    observed = float(np.median(radii))
    return observed, None, "Median des sichtbaren Bogenradius; Innen-/Außenseite nicht sicher zugeordnet"


def analyze_individual_profiles(data: ScanData, tolerance: float = 0.05,
                                min_span: float = 8.0, iterations: int = 800,
                                min_opening: float = 30.0):
    """Wertet Profile einzeln aus; gemeinsame Posen sind dafür nicht erforderlich."""
    diagnostics = profile_diagnostics(data.profiles, tolerance, min_span, 120)
    candidates = []
    for pid, points in sorted(data.profiles.items()):
        if len(points) < 25:
            continue
        groups = extract_line_groups(
            points, tolerance=tolerance, min_span=min_span,
            iterations=min(iterations, 300), max_lines=4,
            min_inliers=max(15, min(25, len(points) // 20)), seed=2026 + pid,
        )
        pair = None
        for i, first in enumerate(groups):
            for second in groups[i + 1:]:
                if _angle_difference_undirected(first.angle_mod_180,
                                                 second.angle_mod_180) < 3.0:
                    continue
                score = (first.support * first.span_mm) * (second.support * second.span_mm)
                if pair is None or score > pair[0]:
                    pair = (score, first, second)
        if pair is None:
            continue
        score, g1, g2 = pair
        try:
            # Im Einzelprofil sind Lage und Biegezentrum nicht nötig: der
            # Schnittpunkt der beiden Schenkel-Mittellinien orientiert den Winkel.
            opening, _, _ = opening_angle(g1, g2, None)
        except NoBendDetected:
            continue
        if opening < min_opening - 2.0:
            continue
        constrained_fit = fit_bend_radius_from_tangents(
            points, g1, g2, opening, tolerance=tolerance
        )
        circle_fits = [constrained_fit] if constrained_fit is not None else []
        candidates.append((score, pid, points, g1, g2, opening, circle_fits))

    if not candidates:
        raise NoBendDetected(
            "Kein einzelnes Profil zeigt zwei ausreichend lange, unterscheidbare "
            "Schenkel. Ohne gemeinsame globale Koordinaten können Teilprofile "
            "nicht sicher zusammengesetzt werden."
        )

    # Robuste Zusammenfassung der Winkel über vollständige Einzelprofile.
    openings = np.array([item[5] for item in candidates], dtype=float)
    median_opening = float(np.median(openings))
    opening_mad = float(np.median(np.abs(openings - median_opening)))
    # Ein Profil mit Kreisfit bevorzugen, damit Geometrieplot und Zentrum im selben Frame liegen.
    plot_candidates = [item for item in candidates if item[6]] or candidates
    _, pid, points, g1, g2, _, plot_circles = max(plot_candidates, key=lambda item: item[0])
    bend = 180.0 - median_opening

    circle_estimates = [(item[1], fit) for item in candidates for fit in item[6]]
    inner_r, outer_r, radius_note = summarize_raw_radii(circle_estimates, data.thickness_mm)
    profile_radius_values = np.array([fit.radius for _, fit in circle_estimates], dtype=float)
    radius_profile_count = len({profile_id for profile_id, _ in circle_estimates})
    radius_mad = (float(np.median(np.abs(profile_radius_values - np.median(profile_radius_values))))
                  if len(profile_radius_values) else None)
    if circle_estimates:
        best_center_profile, best_circle = max(
            circle_estimates,
            key=lambda item: len(item[1].points) * item[1].coverage_deg / max(item[1].rms_mm, 0.01),
        )
        if data.global_coordinates:
            center = np.median(np.vstack([fit.center for _, fit in circle_estimates]), axis=0)
            center_source = "aus Rohdaten geschätzt (globales X/Z)"
        else:
            if plot_circles:
                best_center_profile = pid
                best_circle = max(
                    plot_circles,
                    key=lambda fit: len(fit.points) * fit.coverage_deg / max(fit.rms_mm, 0.01),
                )
            center = best_circle.center
            center_source = f"aus Rohdaten geschätzt; Koordinaten von Profil {best_center_profile}"
    else:
        center = data.center if data.global_coordinates else None
        center_source = data.center_source if center is not None else "aus Rohdaten nicht sicher bestimmbar"
        if center is not None:
            inner_r, outer_r, radius_note = estimate_arc_radius(points, center, data.thickness_mm)
        else:
            radius_note = "Kein ausreichend großer, gekrümmter Bogen in den Rohdaten erkannt"
            plot_circles = []
    return {
        "mode": "individual_median", "selected_profile": pid,
        "points": len(points), "used_profiles": len(candidates),
        "opening_deg": median_opening, "bend_deg": bend,
        "opening_mad_deg": opening_mad,
        "radius_profile_count": radius_profile_count, "radius_mad_mm": radius_mad,
        "profile_radii": [(profile_id, fit.radius) for profile_id, fit in sorted(circle_estimates, key=lambda item: item[0])],
        "profile_openings": [(item[1], item[5]) for item in sorted(candidates, key=lambda item: item[1])],
        "center": center, "center_source": center_source,
        "line1": g1, "line2": g2, "plot_data": points, "plot_circles": plot_circles,
        "inner_radius": inner_r,
        "outer_radius": outer_r, "radius_note": radius_note,
        "diagnostics": diagnostics,
    }


def analyze(data: ScanData, tolerance: float = 0.05, min_span: float = 8.0,
            iterations: int = 800, min_opening: float = 30.0):
    """Führt Einzelprofil- oder gemeinsamen globalen Mehrprofil-Fit aus."""
    if not data.global_coordinates:
        return analyze_individual_profiles(data, tolerance, min_span,
                                           iterations, min_opening)
    usable = {pid: pts for pid, pts in data.profiles.items() if len(pts) >= 25}
    if not usable:
        raise NoBendDetected("Keine Profile mit mindestens 25 gültigen Treffern.")
    points = np.unique(np.round(np.vstack(list(usable.values())), 3), axis=0)
    groups = extract_line_groups(points, tolerance=tolerance, min_span=min_span,
                                 iterations=iterations, max_lines=10,
                                 min_inliers=30, seed=2026)
    if len(groups) < 2:
        diagnostics = profile_diagnostics(data.profiles, tolerance, min_span, 120)
        single = sum("nur ein Schenkel" in s for s in diagnostics.values())
        reason = (f"Nur {len(groups)} dominante Schenkellinie(n) gefunden; "
                  f"{single} Profil(e) zeigen einzeln nur einen Schenkel. "
                  "Scanbogen könnte für verdeckte Schenkel zu eng sein.")
        raise NoBendDetected(reason)

    # Zwei geometrisch längste/dichteste Linien repräsentieren die Blech-Schenkel.
    pair = None
    for i, first in enumerate(groups):
        for second in groups[i + 1:]:
            separation = _angle_difference_undirected(first.angle_mod_180,
                                                       second.angle_mod_180)
            if separation < 3.0:
                continue
            score = (first.support * first.span_mm) * (second.support * second.span_mm)
            if pair is None or score > pair[0]:
                pair = (score, first, second)
    if pair is None:
        diagnostics = profile_diagnostics(data.profiles, tolerance, min_span, 120)
        raise NoBendDetected("Keine zwei unterscheidbaren Schenkelrichtungen; "
                             "möglicherweise gerade Kontur oder unzureichende Sicht.")
    _, g1, g2 = pair
    opening, d1, d2 = opening_angle(g1, g2, data.center)
    if opening < min_opening - 2.0:
        raise NoBendDetected(f"Erkannte Öffnung {opening:.2f}° liegt deutlich unter "
                             f"der eingestellten Untergrenze {min_opening:.1f}°.")
    bend = 180.0 - opening
    try:
        per_profile_result = analyze_individual_profiles(
            data, tolerance, min_span, iterations, min_opening
        )
    except NoBendDetected:
        per_profile_result = None
    if per_profile_result and per_profile_result["radius_profile_count"]:
        center = per_profile_result["center"]
        center_source = per_profile_result["center_source"]
        inner_r = per_profile_result["inner_radius"]
        outer_r = per_profile_result["outer_radius"]
        radius_note = per_profile_result["radius_note"]
        plot_circles = per_profile_result["plot_circles"]
        radius_profile_count = per_profile_result["radius_profile_count"]
        radius_mad = per_profile_result["radius_mad_mm"]
        profile_radii = per_profile_result["profile_radii"]
    else:
        plot_circles = estimate_raw_bend_circles(
            points, [g1, g2], tolerance=tolerance,
            iterations=min(iterations, 500), seed=3026,
        )
        plot_circles = validate_bend_circle_fits(plot_circles, g1, g2, opening)
        if plot_circles:
            center = np.median(np.vstack([fit.center for fit in plot_circles]), axis=0)
            center_source = "aus Rohdaten geschätzt (gemeinsames globales X/Z)"
            inner_r, outer_r, radius_note = summarize_raw_radii(
                [(-1, fit) for fit in plot_circles], data.thickness_mm
            )
        else:
            center = data.center
            center_source = data.center_source
            inner_r, outer_r, radius_note = estimate_arc_radius(
                points, center, data.thickness_mm
            )
        radius_profile_count = 0
        radius_mad = None
        profile_radii = []
    diagnostics = profile_diagnostics(data.profiles, tolerance, min_span, 120)
    return {
        "mode": "combined", "selected_profile": None,
        "points": len(points), "used_profiles": len(usable),
        "opening_deg": opening, "bend_deg": bend,
        "center": center, "center_source": center_source,
        "line1": g1, "line2": g2, "plot_data": points, "plot_circles": plot_circles,
        "inner_radius": inner_r, "outer_radius": outer_r, "radius_note": radius_note,
        "radius_profile_count": radius_profile_count, "radius_mad_mm": radius_mad,
        "profile_radii": profile_radii, "diagnostics": diagnostics,
    }


def _draw_fitted_line(ax, group: LineGroup, label: str, color: str) -> None:
    points = group.points
    direction = group.direction
    anchor = points.mean(axis=0)
    projection = (points - anchor) @ direction
    pad = max(0.5, 0.03 * float(np.ptp(projection)))
    ends = anchor + np.array([projection.min() - pad, projection.max() + pad])[:, None] * direction
    ax.plot(ends[:, 0], ends[:, 1], color=color, linewidth=2.2, label=label)


def _draw_circle_arc(ax, fit: CircleFit, index: int) -> None:
    points = fit.points
    angles = np.sort(np.mod(np.arctan2(
        points[:, 1] - fit.center[1], points[:, 0] - fit.center[0]
    ), 2.0 * math.pi))
    gaps = np.diff(np.r_[angles, angles[0] + 2.0 * math.pi])
    gap_index = int(np.argmax(gaps))
    start = angles[(gap_index + 1) % len(angles)]
    end = angles[gap_index]
    if end < start:
        end += 2.0 * math.pi
    theta = np.linspace(start, end, 120)
    x = fit.center[0] + fit.radius * np.cos(theta)
    z = fit.center[1] + fit.radius * np.sin(theta)
    color = "#009E73" if index == 0 else "#CC79A7"
    ax.plot(x, z, linestyle="--", color=color, linewidth=2.0,
            label=f"Bogenfit R={fit.radius:.2f} mm")
    if index == 0:
        ax.scatter([fit.center[0]], [fit.center[1]], marker="x", s=65,
                   c="#7A3E9D", linewidths=2, label="Biegezentrum (Rohdatenfit)")


def save_analysis_plot(result: dict, output_path: Path) -> None:
    """Speichert Messpunkte, erkannte Schenkelgeraden und Winkelübersicht."""
    import matplotlib.pyplot as plt

    plot_data = result["plot_data"]
    if result["mode"] == "individual_median":
        fig, (ax, ax_angles, ax_radii) = plt.subplots(
            1, 3, figsize=(15.5, 5.5), gridspec_kw={"width_ratios": [1.3, 1.0, 1.0]}
        )
        ax.scatter(plot_data[:, 0], plot_data[:, 1], s=5, c="#555555",
                   alpha=0.45, linewidths=0, label="Profilpunkte")
        ax.set_title(f"Ausgewähltes Profil {result['selected_profile']}")
        profile_ids = [item[0] for item in result["profile_openings"]]
        openings = [item[1] for item in result["profile_openings"]]
        positions = np.arange(len(profile_ids))
        ax_angles.scatter(positions, openings, s=28, c="#0072B2", zorder=3,
                          label="Einzelprofil")
        ax_angles.axhline(result["opening_deg"], color="#D55E00", linewidth=2,
                          label=f"Median {result['opening_deg']:.3f}°")
        ax_angles.set_title(
            f"Öffnungswinkel je Profil\nMAD: {result['opening_mad_deg']:.3f}°"
        )
        ax_angles.set_xlabel("Profilnummer")
        ax_angles.set_ylabel("Öffnungswinkel [°]")
        ax_angles.set_xticks(positions)
        ax_angles.set_xticklabels(profile_ids)
        if len(profile_ids) > 10:
            ax_angles.tick_params(axis="x", labelrotation=45)
        ax_angles.grid(True, alpha=0.3)
        ax_angles.legend(loc="lower left")

        radius_pairs = result.get("profile_radii", [])
        if radius_pairs:
            radius_profile_ids = [item[0] for item in radius_pairs]
            profile_radii = np.array([item[1] for item in radius_pairs], dtype=float)
            radius_positions = np.arange(len(radius_profile_ids))
            radius_median = float(np.median(profile_radii))
            ax_radii.scatter(radius_positions, profile_radii, s=28, c="#009E73",
                             zorder=3, label="Profilradius")
            ax_radii.axhline(radius_median, color="#D55E00", linewidth=2,
                             label=f"Median {radius_median:.3f} mm")
            ax_radii.set_title(
                f"Bogenradius je Profil\nMAD: {result['radius_mad_mm']:.3f} mm"
            )
            ax_radii.set_xlabel("Profilnummer")
            ax_radii.set_ylabel("Radius [mm]")
            ax_radii.set_xticks(radius_positions)
            ax_radii.set_xticklabels(radius_profile_ids)
            if len(radius_profile_ids) > 10:
                ax_radii.tick_params(axis="x", labelrotation=45)
            ax_radii.grid(True, alpha=0.3)
            ax_radii.legend(loc="best")
        else:
            ax_radii.set_axis_off()
            ax_radii.text(0.5, 0.5, "Kein sicherer Rohdaten-Bogenfit",
                          ha="center", va="center", transform=ax_radii.transAxes)
    else:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(plot_data[:, 0], plot_data[:, 1], s=4, c="#555555",
                   alpha=0.35, linewidths=0, label="Registrierte Profilpunkte")
        ax.set_title("Gemeinsamer Mehrprofil-Fit")

    _draw_fitted_line(ax, result["line1"], "Schenkellinie 1", "#0072B2")
    _draw_fitted_line(ax, result["line2"], "Schenkellinie 2", "#D55E00")
    for index, circle_fit in enumerate(result.get("plot_circles", [])):
        _draw_circle_arc(ax, circle_fit, index)
    ax.set_xlabel("X [mm]")
    ax.set_ylabel("Z [mm]")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    title = (f"Biegewinkel: {result['bend_deg']:.3f}°  |  "
             f"Öffnungswinkel: {result['opening_deg']:.3f}°")
    if result["inner_radius"] is not None and result["outer_radius"] is not None:
        title += (f"  |  R innen: {result['inner_radius']:.2f} mm"
                  f"  |  R außen: {result['outer_radius']:.2f} mm")
    elif result["inner_radius"] is not None:
        title += f"  |  Bogenradius: {result['inner_radius']:.2f} mm"
    fig.suptitle(title, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="Mehrprofil-Sammel-CSV")
    parser.add_argument("--center-x", type=float, default=None,
                        help="Biegezentrum X [mm]; Standard: CSV-Metadaten, niemals Sensor-Drehpunkt")
    parser.add_argument("--center-z", type=float, default=None,
                        help="Biegezentrum Z [mm]; Standard: CSV-Metadaten, niemals Sensor-Drehpunkt")
    parser.add_argument("--thickness", type=float, default=None,
                        help="Blechdicke [mm]; Standard: CSV-Metadaten")
    parser.add_argument("--ignore-center-metadata", action="store_true",
                        help="Biegezentrum-Metadaten ignorieren; Sensor-Drehpunkt wird nicht als Biegezentrum genutzt")
    parser.add_argument("--assume-global-coordinates", action="store_true",
                        help="Gemeinsame globale X/Z-Koordinaten bestätigen; sonst werden Profile einzeln ausgewertet")
    parser.add_argument("--tolerance", type=float, default=0.05,
                        help="RANSAC-Abstandstoleranz [mm]")
    parser.add_argument("--min-line-span", type=float, default=8.0,
                        help="Mindestlänge einer Schenkellinie [mm]")
    parser.add_argument("--iterations", type=int, default=800,
                        help="RANSAC-Iterationen pro Linienkandidat")
    parser.add_argument("--min-opening", type=float, default=30.0,
                        help="Kleinster unterstützter Öffnungswinkel [Grad]")
    parser.add_argument("--no-plot", action="store_true",
                        help="Keine Visualisierung als PNG speichern")
    parser.add_argument("--plot-path", type=Path, default=None,
                        help="Speicherpfad der Visualisierung; Standard: neben CSV als *_auswertung.png")
    args = parser.parse_args()

    try:
        data = read_scan(args.csv, args.center_x, args.center_z, args.thickness,
                         ignore_center_metadata=args.ignore_center_metadata,
                         assume_global_coordinates=args.assume_global_coordinates)
        result = analyze(data, args.tolerance, args.min_line_span,
                         args.iterations, args.min_opening)
    except (OSError, ValueError, NoBendDetected) as exc:
        print(f"KEINE VERLÄSSLICHE BIEGUNGSAUSWERTUNG: {exc}", file=sys.stderr)
        if "data" in locals():
            statuses = profile_diagnostics(data.profiles, args.tolerance,
                                           args.min_line_span, 100)
            print("Profilstatus:", file=sys.stderr)
            expected = data.expected_profiles or max(data.profiles)
            for pid in range(1, expected + 1):
                status = statuses.get(pid, "keine CSV-Treffer für dieses Profil")
                print(f"  Profil {pid}: {status}", file=sys.stderr)
        return 2

    if result["mode"] == "individual_median":
        print(f"Einzelprofil-Auswertung: Median aus {result['used_profiles']} Profil(en)")
        print(f"Repräsentative Linien aus Profil {result['selected_profile']} "
              f"({result['points']} eindeutige Punkte)")
        angles_text = ", ".join(
            f"{pid}: {angle:.3f}°" for pid, angle in result["profile_openings"]
        )
        print(f"Einzel-Öffnungswinkel [Profil: Winkel]: {angles_text}")
        print(f"Median absolute Abweichung der Öffnungswinkel: "
              f"{result['opening_mad_deg']:.3f}°")
    else:
        print(f"Gemeinsamer Fit: {result['points']} eindeutige Punkte aus "
              f"{result['used_profiles']} Profil(en)")
    if result["center"] is not None:
        print(f"Biege-/Kreismittelpunkt: ({result['center'][0]:.3f}, {result['center'][1]:.3f}) mm "
              f"[{result['center_source']}]")
    else:
        print("Biege-/Kreismittelpunkt: unbekannt; Winkel aus Schenkelgeraden bestimmt")
    print(f"Öffnungswinkel: {result['opening_deg']:.3f}°")
    print(f"Biegewinkel:    {result['bend_deg']:.3f}°")
    print(f"Schenkel 1:     {result['line1'].support} Punkte, {result['line1'].span_mm:.2f} mm")
    print(f"Schenkel 2:     {result['line2'].support} Punkte, {result['line2'].span_mm:.2f} mm")
    if result["inner_radius"] is not None and result["outer_radius"] is not None:
        print(f"Innenradius:    {result['inner_radius']:.3f} mm")
        print(f"Außenradius:    {result['outer_radius']:.3f} mm")
        print(f"Radiusdiagnose: {result['radius_note']}")
    elif result["inner_radius"] is not None:
        print(f"Bogenradius beobachtet: {result['inner_radius']:.3f} mm")
        print(f"Radiusdiagnose: {result['radius_note']}")
    else:
        print(f"Radius nicht sicher bestimmt: {result['radius_note']}")
    if result.get("radius_profile_count", 0):
        print(f"Radius aus {result['radius_profile_count']} verwertbaren Profil(en); "
              f"Medianabweichung: {result['radius_mad_mm']:.3f} mm")

    statuses = result["diagnostics"]
    usable_individual = sum("Einzelprofil verwertbar" in x for x in statuses.values())
    partial = sum("nur ein Schenkel" in x for x in statuses.values())
    print(f"Profilstatus: {usable_individual} einzeln auswertbar; {partial} Teilprofil(e) nur mit einem sichtbaren Schenkel.")
    for pid, status in sorted(statuses.items()):
        print(f"  Profil {pid}: {status}")
    expected = data.expected_profiles or max(data.profiles)
    missing = sorted(set(range(1, expected + 1)) - set(data.profiles))
    if missing:
        print("Profile ohne Treffer: " + ", ".join(map(str, missing)))
    if result["opening_deg"] < args.min_opening:
        print(f"WARNUNG: Öffnungswinkel liegt nahe der Untergrenze {args.min_opening:.1f}°.")
    if not args.no_plot:
        plot_path = args.plot_path or args.csv.with_name(f"{args.csv.stem}_auswertung.png")
        try:
            save_analysis_plot(result, plot_path)
        except (ImportError, OSError) as exc:
            print(f"Visualisierung nicht gespeichert: {exc}", file=sys.stderr)
        else:
            print(f"Visualisierung gespeichert: {plot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
