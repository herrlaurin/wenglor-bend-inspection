#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Kreisbahn-Raytracing für ein gebogenes Blech mit weCat3D-ähnlicher CSV.

Das Werkstück wird als Blech mit zwei geraden Schenkeln und einem tangentialen
Bogen modelliert. Der Sensor bewegt sich auf einer einstellbaren Kreisbahn um
einen festen Pivot; dieser ist unabhängig vom Biegezentrum und per Offset oder
absoluter Position einstellbar. Das Werkstück kann entlang Z verschoben werden.
Pro Profil werden 2048 Strahlen ausgesendet. Gespeichert wird je Strahl nur der
erste Treffer; verdeckte Konturabschnitte und Strahlen ohne Treffer fehlen in
der Sammel-CSV. Jede Trefferzeile enthält Profilnummer, Sensorwinkel und Pose.

Koordinaten: X nach rechts, Z nach oben; Winkel von +X gegen den Uhrzeigersinn.
Beispielwerte sind Platzhalter und können im Config-Block oder per CLI geändert
werden. Das Modell ist eine geometrische Näherung, kein Ersatz für die optische
Triangulation des realen weCat3D.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import math

import numpy as np


@dataclass(frozen=True)
class Config:
    # Fester Sensor-Drehpunkt (Pivot). None = nominales Biegezentrum plus Offset.
    # Die Kreisbahn ist damit ausdrücklich vom Biegezentrum getrennt.
    orbit_center_x_mm: float | None = None
    orbit_center_z_mm: float | None = None
    pivot_offset_x_mm: float = 0.0
    pivot_offset_z_mm: float = 0.0
    orbit_radius_mm: float = 200.0
    scan_start_angle_deg: float = -60.0
    scan_end_angle_deg: float = 60.0
    scan_angle_step_deg: float = 5.0
    scan_rotation_deg: float = -30.0
    one_sided_scan: bool = True
    # Werkstückverschiebung entlang Z. 0 ist die Mitte des 50-mm-Simulationswegs.
    bend_shift_z_mm: float = 0.0
    max_bend_travel_mm: float = 50.0
    # Beispiel-Sichtfeld passend zu etwa 62 mm Breite bei 150 mm Abstand.
    fov_deg: float = 2.0 * math.degrees(math.atan(31.0 / 150.0))
    ray_count: int = 2048

    # Werkstück: zwei Schenkel plus eine Biegung.
    bend_x_mm: float = 0.0
    bend_z_mm: float = 0.0
    bend_angle_deg: float = 150.0
    inner_radius_mm: float = 2.5
    thickness_mm: float = 2.0
    leg1_length_mm: float = 40.0
    leg2_length_mm: float = 40.0
    first_leg_angle_deg: float = 0.0

    # Für den aktuellen Auswertungstest ist das Sensorrauschen deaktiviert.
    noise_std_mm: float = 0.0
    intensity_mean: float = 820.0
    intensity_noise_std: float = 0.0
    seed: int | None = 1234
    save_png: bool = True
    save_metadata: bool = True


DEFAULT = Config()
OUTPUT_DIR = Path(__file__).resolve().parent / "sim_profile"


def resolve_bend_center(c: Config) -> np.ndarray:
    """Liefert das nominale geometrische Kreismittelpunkt des Blechbogens."""
    alpha = math.radians(c.first_leg_angle_deg)
    sign = 1.0 if c.bend_angle_deg > 0 else -1.0
    r_mid = c.inner_radius_mm + c.thickness_mm / 2.0
    tangent = np.array([c.bend_x_mm, c.bend_z_mm], dtype=float)
    return tangent + sign * r_mid * np.array([-math.sin(alpha), math.cos(alpha)])


def resolve_orbit_center(c: Config) -> np.ndarray:
    """Liefert den festen Sensor-Drehpunkt, getrennt vom Biegezentrum."""
    nominal_bend = resolve_bend_center(c)
    derived = nominal_bend + np.array([c.pivot_offset_x_mm, c.pivot_offset_z_mm])
    x = derived[0] if c.orbit_center_x_mm is None else c.orbit_center_x_mm
    z = derived[1] if c.orbit_center_z_mm is None else c.orbit_center_z_mm
    if not (math.isfinite(x) and math.isfinite(z)):
        raise ValueError("Koordinaten des Sensor-Drehpunkts müssen endlich sein.")
    return np.array([x, z], dtype=float)


def validate_config(c: Config) -> None:
    """Validiert Grenzen der verschiebbaren Werkstücklage und Kreisbahn."""
    if not math.isfinite(c.max_bend_travel_mm) or c.max_bend_travel_mm < 0:
        raise ValueError("Der maximale Blechverschiebeweg muss endlich und nicht negativ sein.")
    if not math.isfinite(c.bend_shift_z_mm):
        raise ValueError("Die Blechverschiebung muss endlich sein.")
    if abs(c.bend_shift_z_mm) > c.max_bend_travel_mm / 2.0 + 1e-9:
        raise ValueError(
            f"Blechverschiebung außerhalb des Gesamtwegs von {c.max_bend_travel_mm:g} mm; "
            "die Verschiebung wird relativ zur Wegmitte angegeben."
        )
    if not all(math.isfinite(v) for v in (c.pivot_offset_x_mm, c.pivot_offset_z_mm)):
        raise ValueError("Pivot-Offsets müssen endlich sein.")


def cross2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """2D-Kreuzprodukt; funktioniert auch mit Arrays."""
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def build_sheet_polygon(c: Config, sample_step_mm: float = 0.05) -> np.ndarray:
    """Erzeugt das geschlossene 2D-Polygon der Blechoberfläche."""
    if c.thickness_mm <= 0 or c.inner_radius_mm <= 0:
        raise ValueError("Blechdicke und Innenradius müssen größer als 0 sein.")
    if c.leg1_length_mm <= 0 or c.leg2_length_mm <= 0:
        raise ValueError("Beide Schenkellängen müssen größer als 0 sein.")
    if not 0 < abs(c.bend_angle_deg) < 175:
        raise ValueError("Der Betrag des Biegewinkels muss zwischen 0 und 175 Grad liegen.")

    theta = math.radians(c.bend_angle_deg)
    a0 = math.radians(c.first_leg_angle_deg)
    r_mid = c.inner_radius_mm + c.thickness_mm / 2.0
    sign = 1.0 if theta > 0 else -1.0

    # Lokaler Biegeanfang: erste Tangente zeigt nach +X.
    p0 = np.array([0.0, 0.0])
    n_line = max(2, int(math.ceil(c.leg1_length_mm / sample_step_mm)) + 1)
    first = np.column_stack((np.linspace(-c.leg1_length_mm, 0.0, n_line),
                             np.zeros(n_line)))

    # Kreisbogen mit Mittellinienradius = Innenradius + halbe Blechdicke.
    center = p0 + np.array([0.0, sign * r_mid])
    phi0 = -sign * math.pi / 2.0
    n_arc = max(3, int(math.ceil(r_mid * abs(theta) / sample_step_mm)) + 1)
    phi = np.linspace(phi0, phi0 + theta, n_arc)
    arc = center + r_mid * np.column_stack((np.cos(phi), np.sin(phi)))
    p1 = arc[-1]
    d1 = np.array([math.cos(theta), math.sin(theta)])
    n2 = max(2, int(math.ceil(c.leg2_length_mm / sample_step_mm)) + 1)
    second = p1 + np.linspace(0.0, c.leg2_length_mm, n2)[:, None] * d1

    # Mittellinie mit doppelten Anschlussstellen vermeiden.
    centerline = np.vstack((first, arc[1:], second[1:]))
    # Rotieren und in die vorgegebene Biegeposition verschieben.
    rot = np.array([[math.cos(a0), -math.sin(a0)],
                    [math.sin(a0),  math.cos(a0)]])
    origin = np.array([c.bend_x_mm, c.bend_z_mm + c.bend_shift_z_mm])
    centerline = centerline @ rot.T + origin

    # Lokale Normalen für gleichmäßige Blechdicke entlang des Mittellinienbogens.
    tangent = np.empty_like(centerline)
    tangent[0] = centerline[1] - centerline[0]
    tangent[-1] = centerline[-1] - centerline[-2]
    tangent[1:-1] = centerline[2:] - centerline[:-2]
    lengths = np.linalg.norm(tangent, axis=1)
    if np.any(lengths == 0):
        raise ValueError("Ungültige Mittellinie; bitte Geometrieparameter prüfen.")
    tangent /= lengths[:, None]
    normal = np.column_stack((-tangent[:, 1], tangent[:, 0]))

    half_t = c.thickness_mm / 2.0
    side_a = centerline + half_t * normal
    side_b = centerline - half_t * normal
    polygon = np.vstack((side_a, side_b[::-1]))
    return polygon


def first_hit(origin: np.ndarray, direction: np.ndarray,
              polygon: np.ndarray) -> tuple[float, np.ndarray, np.ndarray] | None:
    """Nächster positiver Strahl-Schnitt mit dem geschlossenen Polygon."""
    a = polygon
    b = np.roll(polygon, -1, axis=0)
    edge = b - a
    w = a - origin
    denom = cross2(np.broadcast_to(direction, edge.shape), edge)
    nonparallel = np.abs(denom) > 1e-12
    t = np.full(len(edge), np.inf)
    u = np.full(len(edge), np.inf)
    t[nonparallel] = cross2(w[nonparallel], edge[nonparallel]) / denom[nonparallel]
    u[nonparallel] = cross2(w[nonparallel],
                            np.broadcast_to(direction, edge.shape)[nonparallel]) / denom[nonparallel]
    valid = nonparallel & (t > 1e-9) & (u >= -1e-10) & (u <= 1.0 + 1e-10)
    if not np.any(valid):
        return None
    candidates = np.flatnonzero(valid)
    k = candidates[np.argmin(t[candidates])]
    point = origin + t[k] * direction
    edge_len = np.linalg.norm(edge[k])
    normal = np.array([-edge[k, 1], edge[k, 0]]) / max(edge_len, 1e-12)
    return float(t[k]), point, normal, int(k)


def first_hits_batch(origin: np.ndarray, directions: np.ndarray,
                     polygon: np.ndarray, chunk_size: int = 128):
    """Vektorisierte erste Treffer für viele Strahlen; chunks begrenzen Speicher."""
    a = polygon
    edge = np.roll(polygon, -1, axis=0) - polygon
    w = a - origin
    n_rays = len(directions)
    points = np.full((n_rays, 2), np.nan)
    normals = np.full((n_rays, 2), np.nan)
    distances = np.full(n_rays, np.inf)
    hit_mask = np.zeros(n_rays, dtype=bool)
    hit_edge_ids = np.full(n_rays, -1, dtype=int)
    for start in range(0, n_rays, chunk_size):
        stop = min(start + chunk_size, n_rays)
        d = directions[start:stop]
        denom = d[:, None, 0] * edge[None, :, 1] - d[:, None, 1] * edge[None, :, 0]
        nonparallel = np.abs(denom) > 1e-12
        numer_t = w[None, :, 0] * edge[None, :, 1] - w[None, :, 1] * edge[None, :, 0]
        numer_u = w[None, :, 0] * d[:, None, 1] - w[None, :, 1] * d[:, None, 0]
        t = np.divide(numer_t, denom, out=np.full_like(denom, np.inf), where=nonparallel)
        u = np.divide(numer_u, denom, out=np.full_like(denom, np.inf), where=nonparallel)
        valid = nonparallel & (t > 1e-9) & (u >= -1e-10) & (u <= 1.0 + 1e-10)
        t[~valid] = np.inf
        nearest = np.argmin(t, axis=1)
        best_t = t[np.arange(stop - start), nearest]
        good = np.isfinite(best_t)
        if not np.any(good):
            continue
        ray_ids = np.arange(start, stop)[good]
        edge_ids = nearest[good]
        distances[ray_ids] = best_t[good]
        points[ray_ids] = origin + best_t[good, None] * d[good]
        e = edge[edge_ids]
        lengths = np.linalg.norm(e, axis=1)
        normals[ray_ids] = np.column_stack((-e[:, 1], e[:, 0])) / np.maximum(lengths[:, None], 1e-12)
        hit_mask[ray_ids] = True
        hit_edge_ids[ray_ids] = edge_ids
    return distances, points, normals, hit_mask, hit_edge_ids


def simulate(c: Config, sensor_x_mm: float, sensor_z_mm: float,
             axis_angle_deg: float, rng: np.random.Generator | None = None,
             polygon: np.ndarray | None = None,
             allowed_face_edges: np.ndarray | None = None
             ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Ein Profil aus der vorgegebenen Sensorpose; je Strahl nur erster Treffer."""
    if polygon is None:
        polygon = build_sheet_polygon(c)
    if rng is None:
        rng = np.random.default_rng(c.seed)
    origin = np.array([sensor_x_mm, sensor_z_mm], dtype=float)
    axis = math.radians(axis_angle_deg)
    half_fov = math.radians(c.fov_deg) / 2.0
    if c.ray_count < 2:
        raise ValueError("Die Strahlenzahl muss mindestens 2 sein.")
    if not 0 < c.fov_deg < 180:
        raise ValueError("Das Sichtfeld muss zwischen 0 und 180 Grad liegen.")

    # 2048 Winkelstrahlen; Trefferberechnung vektorisiert, nur sichtbare Treffer bleiben.
    angles = np.linspace(axis - half_fov, axis + half_fov, c.ray_count)
    directions = np.column_stack((np.cos(angles), np.sin(angles)))
    _distance, points, surface_normals, hit_mask, hit_edge_ids = first_hits_batch(
        origin, directions, polygon
    )
    if allowed_face_edges is not None:
        valid_edges = hit_edge_ids >= 0
        keep_face = np.zeros_like(hit_mask)
        keep_face[valid_edges] = allowed_face_edges[hit_edge_ids[valid_edges]]
        hit_mask &= keep_face
    ray_ids = np.flatnonzero(hit_mask)
    if len(ray_ids) == 0:
        return np.empty((0, 3)), polygon, angles, origin

    visible_points = points[ray_ids]
    incidence = np.abs(np.sum(directions[ray_ids] * surface_normals[ray_ids], axis=1))
    intensities = c.intensity_mean * (0.25 + 0.75 * incidence)
    intensities += rng.normal(0.0, c.intensity_noise_std, len(ray_ids))
    if c.noise_std_mm:
        visible_points = visible_points + rng.normal(0.0, c.noise_std_mm, visible_points.shape)
    intensities = np.clip(np.round(intensities), 0, 1023)
    hits = np.column_stack((visible_points, intensities))

    # CSV-Profilpunkte in X-Reihenfolge wie beim üblichen Profil-Export.
    hits = hits[np.argsort(hits[:, 0])]
    return hits, polygon, angles, origin


def make_scan_angles(c: Config) -> np.ndarray:
    """Erzeugt Scanwinkel inklusive Endwinkel; letzter Schritt ggf. kürzer."""
    values = (c.scan_start_angle_deg, c.scan_end_angle_deg,
              c.scan_angle_step_deg, c.orbit_radius_mm)
    if not all(math.isfinite(v) for v in values):
        raise ValueError("Kreisbahnparameter müssen endlich sein.")
    if c.orbit_radius_mm <= 0:
        raise ValueError("Der Bahnradius muss größer als 0 sein.")
    step = abs(c.scan_angle_step_deg)
    if step <= 0:
        raise ValueError("Der Winkelschritt muss größer als 0 sein.")
    delta = c.scan_end_angle_deg - c.scan_start_angle_deg
    if abs(delta) < 1e-12:
        return np.array([c.scan_start_angle_deg], dtype=float)
    direction = 1.0 if delta > 0 else -1.0
    count = int(math.floor(abs(delta) / step + 1e-12))
    angles = c.scan_start_angle_deg + direction * step * np.arange(count + 1)
    if abs(angles[-1] - c.scan_end_angle_deg) > 1e-9:
        angles = np.append(angles, c.scan_end_angle_deg)
    else:
        angles[-1] = c.scan_end_angle_deg
    return angles


def _surface_edge_masks(polygon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Kantenmasken für die zwei Blechflächen; Stirnkanten sind ausgeschlossen."""
    n = len(polygon) // 2
    face_a = np.zeros(len(polygon), dtype=bool)
    face_b = np.zeros(len(polygon), dtype=bool)
    face_a[:n - 1] = True
    face_b[n:2 * n - 1] = True
    return face_a, face_b


def _choose_front_face(c: Config, polygon: np.ndarray, orbit_center: np.ndarray,
                       first_scan_angle: float) -> tuple[np.ndarray, str]:
    """Legt die von der ersten Scanpose sichtbare physische Blechfläche fest."""
    face_a, face_b = _surface_edge_masks(polygon)
    theta = math.radians(first_scan_angle)
    origin = orbit_center + c.orbit_radius_mm * np.array([math.cos(theta), math.sin(theta)])
    axis = math.radians(first_scan_angle + 180.0)
    half_fov = math.radians(c.fov_deg) / 2.0
    ray_angles = np.linspace(axis - half_fov, axis + half_fov, c.ray_count)
    directions = np.column_stack((np.cos(ray_angles), np.sin(ray_angles)))
    _, _, _, hit_mask, edge_ids = first_hits_batch(origin, directions, polygon)
    valid = hit_mask & (edge_ids >= 0)
    count_a = int(np.sum(valid & face_a[np.maximum(edge_ids, 0)]))
    count_b = int(np.sum(valid & face_b[np.maximum(edge_ids, 0)]))
    if count_a == 0 and count_b == 0:
        edge = np.roll(polygon, -1, axis=0) - polygon
        lengths = np.linalg.norm(edge, axis=1)
        normals = np.column_stack((-edge[:, 1], edge[:, 0])) / np.maximum(lengths[:, None], 1e-12)
        midpoints = 0.5 * (polygon + np.roll(polygon, -1, axis=0))
        to_sensor = origin - midpoints
        to_sensor /= np.maximum(np.linalg.norm(to_sensor, axis=1)[:, None], 1e-12)
        facing = np.maximum(np.sum(normals * to_sensor, axis=1), 0.0)
        count_a = float(np.sum(facing[face_a]))
        count_b = float(np.sum(facing[face_b]))
    if count_a >= count_b:
        return face_a, "Fläche A"
    return face_b, "Fläche B"


def simulate_scan(c: Config):
    """Simuliert die gedrehte Kreisbahn und sperrt die Rückseite nach der Startpose."""
    validate_config(c)
    polygon = build_sheet_polygon(c)
    rng = np.random.default_rng(c.seed)
    center = resolve_orbit_center(c)
    scan_angles = make_scan_angles(c) + c.scan_rotation_deg
    allowed_face_edges = None
    face_name = "beide Flächen"
    if c.one_sided_scan:
        allowed_face_edges, face_name = _choose_front_face(
            c, polygon, center, float(scan_angles[0])
        )
        print(f"Einseitiger Scan: nur {face_name} aus der ersten Scanpose; Rückseite wird verworfen.")
    profiles = []
    for scan_angle in scan_angles:
        theta = math.radians(float(scan_angle))
        origin = center + c.orbit_radius_mm * np.array([math.cos(theta), math.sin(theta)])
        axis_deg = float(scan_angle) + 180.0
        hits, _, ray_angles, _ = simulate(
            c, origin[0], origin[1], axis_deg, rng=rng, polygon=polygon,
            allowed_face_edges=allowed_face_edges,
        )
        profiles.append((float(scan_angle), origin, hits, ray_angles))
    return profiles, polygon, allowed_face_edges, face_name


def unique_csv_path(c: Config) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"scan_{abs(c.bend_angle_deg):g}g_R{c.inner_radius_mm:g}".replace(".", "p")
    base = OUTPUT_DIR / f"wecat3d_sim_raycast_{stamp}_{tag}"
    path = base.with_suffix(".csv")
    index = 1
    while path.exists():
        path = Path(f"{base}_{index}.csv")
        index += 1
    return path


def save_csv(path: Path, profiles, c: Config, face_name: str = "beide Flächen") -> None:
    """Speichert sichtbare Treffer; Metadatenzeilen sind optional."""
    center = resolve_orbit_center(c)
    with path.open("w", encoding="utf-8", newline="") as f:
        if c.save_metadata:
            total_points = sum(len(hits) for _, _, hits, _ in profiles)
            f.write(f"# wenglor weCat3D (simuliert) - Kreisbahnscan mit {len(profiles)} Profilen\n")
            f.write(f"# Points: {total_points}; Rays per profile: {c.ray_count}; Units: mm; Seed: {c.seed}\n")
            bend_center = resolve_bend_center(c) + np.array([0.0, c.bend_shift_z_mm])
            scan_start = c.scan_start_angle_deg + c.scan_rotation_deg
            scan_end = c.scan_end_angle_deg + c.scan_rotation_deg
            f.write(f"# Pivot center X={center[0]:g} mm, Z={center[1]:g} mm; "
                    f"Orbit center X={center[0]:g} mm, Z={center[1]:g} mm; "
                    f"Radius={c.orbit_radius_mm:g} mm; Start={scan_start:g} deg; "
                    f"End={scan_end:g} deg; Rotation={c.scan_rotation_deg:g} deg; "
                    f"Step={abs(c.scan_angle_step_deg):g} deg\n")
            f.write(f"# One-sided scan={c.one_sided_scan}; active surface={face_name}\n")
            f.write(f"# Pivot offset from nominal bend center X={c.pivot_offset_x_mm:g} mm, "
                    f"Z={c.pivot_offset_z_mm:g} mm\n")
            f.write(f"# Bend center X={bend_center[0]:g} mm, Z={bend_center[1]:g} mm; "
                    f"Bend shift Z={c.bend_shift_z_mm:g} mm; Travel={c.max_bend_travel_mm:g} mm total\n")
            f.write("# Coordinates: local sensor X/Z in mm; profiles concatenated but not spatially registered\n")
            f.write(f"# FOV={c.fov_deg:.6g} deg; Sensor points radially to fixed pivot\n")
            f.write(f"# Werkstueck: Biegung={c.bend_angle_deg:g} Grad, "
                    f"Innenradius={c.inner_radius_mm:g} mm, Dicke={c.thickness_mm:g} mm\n")
            f.write("X [mm],Z [mm],Intensity,Profile,Sensor angle [deg],Sensor X [mm],Sensor Z [mm]\n")
        else:
            # Profilnummer bleibt als Gruppierungsschlüssel erhalten; alle
            # Kommentar-Metadaten und optionalen Sensorpose-Spalten entfallen.
            f.write("X [mm],Z [mm],Intensity,Profile\n")
        for profile_id, (scan_angle, origin, hits, _ray_angles) in enumerate(profiles, start=1):
            # Weltkoordinaten in das lokale Sensorprofil transformieren.
            # Lokale X-Achse: quer zur mittleren Blickrichtung; lokale Z-Achse: Blickrichtung.
            axis = math.radians(scan_angle + 180.0)
            delta_x = hits[:, 0] - origin[0]
            delta_z = hits[:, 1] - origin[1]
            local_x = -math.sin(axis) * delta_x + math.cos(axis) * delta_z
            local_z = math.cos(axis) * delta_x + math.sin(axis) * delta_z
            for x, z, intensity in zip(local_x, local_z, hits[:, 2]):
                if c.save_metadata:
                    f.write(f"{x:.3f},{z:.3f},{int(intensity)},{profile_id},{scan_angle:.3f},"
                            f"{origin[0]:.3f},{origin[1]:.3f}\n")
                else:
                    f.write(f"{x:.3f},{z:.3f},{int(intensity)},{profile_id}\n")


def save_plot(path: Path, profiles, polygon: np.ndarray, c: Config,
              allowed_face_edges: np.ndarray | None = None) -> bool:
    """Speichert eine Übersicht über Werkstück, Kreisbahn und Profiltreffer."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib fehlt; PNG-Plot wird übersprungen.")
        return False
    fig, ax = plt.subplots(figsize=(10, 7))
    closed = np.vstack((polygon, polygon[0]))
    ax.plot(closed[:, 0], closed[:, 1], color="black", lw=1.5, label="Blechkontur")
    centers = np.array([origin for _, origin, _, _ in profiles])
    orbit_center = resolve_orbit_center(c)
    ax.plot(centers[:, 0], centers[:, 1], "--", color="tab:orange", lw=1.2,
            label="Sensor-Kreisbahn")
    ax.scatter(centers[:, 0], centers[:, 1], marker="x", s=28, c="tab:red",
               label="Sensorpositionen")
    cmap = plt.get_cmap("viridis", max(1, len(profiles)))
    for i, (scan_angle, origin, hits, ray_angles) in enumerate(profiles):
        if len(hits):
            ax.scatter(hits[:, 0], hits[:, 1], s=4, color=cmap(i), alpha=0.55,
                       label=f"Profil {i + 1} ({scan_angle:g}°)")
        # Pro Profil nur wenige Strahlen zeichnen.
        ray_step = max(1, len(ray_angles) // 12)
        for angle in ray_angles[::ray_step]:
            direction = np.array([math.cos(angle), math.sin(angle)])
            end = origin + c.orbit_radius_mm * 1.25 * direction
            hit = first_hit(origin, direction, polygon)
            if hit is not None and (
                    allowed_face_edges is None or allowed_face_edges[hit[3]]):
                end = hit[1]
            ax.plot([origin[0], end[0]], [origin[1], end[1]], color=cmap(i),
                    alpha=0.12, lw=0.5)
    ax.scatter([orbit_center[0]], [orbit_center[1]], marker="+", s=70,
               c="tab:orange", label="Fester Sensor-Drehpunkt")
    bend_center = resolve_bend_center(c) + np.array([0.0, c.bend_shift_z_mm])
    ax.scatter([bend_center[0]], [bend_center[1]], marker="o", s=45,
               facecolors="none", edgecolors="tab:purple", label="Biegezentrum")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("X [mm]")
    ax.set_ylabel("Z [mm]")
    ax.set_title(f"Kreisbahnscan: {len(profiles)} Profile, sichtbare Treffer (Plot global; CSV lokal)")
    ax.grid(True)
    ax.legend(loc="best", fontsize="small")
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=140)
    plt.close(fig)
    return True


def parse_args() -> Config:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--orbit-center-x", type=float, default=DEFAULT.orbit_center_x_mm,
                   help="Absoluter fester Sensor-Drehpunkt X [mm]")
    p.add_argument("--orbit-center-z", type=float, default=DEFAULT.orbit_center_z_mm,
                   help="Fester Sensor-Drehpunkt Z [mm]")
    p.add_argument("--pivot-offset-x", type=float, default=DEFAULT.pivot_offset_x_mm,
                   help="X-Abstand des festen Sensor-Drehpunkts zum nominalen Biegezentrum [mm]")
    p.add_argument("--pivot-offset-z", type=float, default=DEFAULT.pivot_offset_z_mm,
                   help="Z-Abstand des festen Sensor-Drehpunkts zum nominalen Biegezentrum [mm]")
    p.add_argument("--bend-shift-z", type=float, default=DEFAULT.bend_shift_z_mm,
                   help="Werkstückverschiebung Z relativ zur Wegmitte [mm], zulässig ± Gesamtweg/2")
    p.add_argument("--max-bend-travel", type=float, default=DEFAULT.max_bend_travel_mm,
                   help="Gesamter Werkstück-Verschiebeweg [mm], Standard 50")
    p.add_argument("--orbit-radius", type=float, default=DEFAULT.orbit_radius_mm,
                   help="Sensorbahnradius [mm]")
    p.add_argument("--scan-start-deg", type=float, default=DEFAULT.scan_start_angle_deg,
                   help="Startwinkel der Sensorbahn gegen +X [Grad]")
    p.add_argument("--scan-end-deg", type=float, default=DEFAULT.scan_end_angle_deg,
                   help="Endwinkel der Sensorbahn gegen +X [Grad]")
    p.add_argument("--scan-step-deg", type=float, default=DEFAULT.scan_angle_step_deg,
                   help="Winkelschritt; Endwinkel wird ggf. zusätzlich aufgenommen [Grad]")
    p.add_argument("--scan-rotation-deg", type=float, default=DEFAULT.scan_rotation_deg,
                   help="Drehung des gesamten Scanbogens; negativ = im Uhrzeigersinn [Grad]")
    p.add_argument("--both-sides", action="store_true",
                   help="Rückseitenfilter deaktivieren und beide Blechflächen aufnehmen")
    p.add_argument("--fov-deg", type=float, default=DEFAULT.fov_deg, help="Sensor-Sichtfeld [Grad]")
    p.add_argument("--rays", type=int, default=DEFAULT.ray_count, help="Strahlen je Profil (Standard 2048)")
    p.add_argument("--bend-x", type=float, default=DEFAULT.bend_x_mm, help="Biegeposition/Tangente X [mm]")
    p.add_argument("--bend-z", type=float, default=DEFAULT.bend_z_mm, help="Biegeposition/Tangente Z [mm]")
    p.add_argument("--bend-deg", type=float, default=DEFAULT.bend_angle_deg, help="Biegewinkel [Grad]")
    p.add_argument("--inner-radius", type=float, default=DEFAULT.inner_radius_mm, help="Innenradius [mm]")
    p.add_argument("--thickness", type=float, default=DEFAULT.thickness_mm, help="Blechdicke [mm]")
    p.add_argument("--leg1", type=float, default=DEFAULT.leg1_length_mm, help="Schenkellänge 1 [mm]")
    p.add_argument("--leg2", type=float, default=DEFAULT.leg2_length_mm, help="Schenkellänge 2 [mm]")
    p.add_argument("--first-leg-angle", type=float, default=DEFAULT.first_leg_angle_deg,
                   help="Richtung des ersten Schenkels [Grad]")
    p.add_argument("--noise", type=float, default=DEFAULT.noise_std_mm, help="Positionsrauschen [mm]")
    p.add_argument("--intensity-noise", type=float, default=DEFAULT.intensity_noise_std,
                   help="Intensitätsrauschen; Standard 0")
    p.add_argument("--seed", type=int, default=DEFAULT.seed, help="Seed; -1 für zufälliges Rauschen")
    p.add_argument("--no-png", action="store_true", help="Keinen Übersichtsplot speichern")
    p.add_argument("--no-metadata", action="store_true",
                   help="Keine Kommentar-Metadaten oder Sensorpose-Spalten in die CSV schreiben")
    a = p.parse_args()
    seed = None if a.seed == -1 else a.seed
    return Config(
        orbit_center_x_mm=a.orbit_center_x, orbit_center_z_mm=a.orbit_center_z,
        pivot_offset_x_mm=a.pivot_offset_x, pivot_offset_z_mm=a.pivot_offset_z,
        bend_shift_z_mm=a.bend_shift_z, max_bend_travel_mm=a.max_bend_travel,
        orbit_radius_mm=a.orbit_radius, scan_start_angle_deg=a.scan_start_deg,
        scan_end_angle_deg=a.scan_end_deg, scan_angle_step_deg=a.scan_step_deg,
        scan_rotation_deg=a.scan_rotation_deg, one_sided_scan=not a.both_sides,
        fov_deg=a.fov_deg, ray_count=a.rays,
        bend_x_mm=a.bend_x, bend_z_mm=a.bend_z, bend_angle_deg=a.bend_deg,
        inner_radius_mm=a.inner_radius, thickness_mm=a.thickness,
        leg1_length_mm=a.leg1, leg2_length_mm=a.leg2,
        first_leg_angle_deg=a.first_leg_angle, noise_std_mm=a.noise,
        intensity_noise_std=a.intensity_noise,
        seed=seed, save_png=not a.no_png, save_metadata=not a.no_metadata,
    )


def main() -> None:
    c = parse_args()
    profiles, polygon, allowed_face_edges, face_name = simulate_scan(c)
    out = unique_csv_path(c)
    save_csv(out, profiles, c, face_name)
    total_hits = sum(len(hits) for _, _, hits, _ in profiles)
    print(f"CSV gespeichert: {out}")
    print(f"Profile: {len(profiles)}; sichtbare Treffer insgesamt: {total_hits}")
    if c.save_png and save_plot(out, profiles, polygon, c, allowed_face_edges):
        print(f"Übersichtsplot gespeichert: {out.with_suffix('.png')}")


if __name__ == "__main__":
    main()
