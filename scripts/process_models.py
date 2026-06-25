#!/usr/bin/env python3
import argparse
import json
import math
import os
import re
import sys
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import trimesh
import coacd


Y_UP_TO_Z_UP = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


def sanitize_name(url: str, index: int) -> str:
    parsed = urlparse(url)
    base = Path(parsed.path).stem or f"model_{index:03d}"
    base = re.sub(r"[^A-Za-z0-9_.-]+", "_", base)
    return f"{index:03d}_{base}"


def read_filelist(path: Path) -> list[str]:
    urls = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        urls.append(line)
    if not urls:
        raise RuntimeError(f"No URLs found in {path}")
    return urls


def download(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    print(f"[download] {url}")
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "github-actions-coacd-pipeline"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        status = getattr(resp, "status", 200)
        if status != 200:
            raise RuntimeError(f"HTTP status {status} for {url}")
        data = resp.read()

    if not data:
        raise RuntimeError(f"Downloaded empty file from {url}")

    dst.write_bytes(data)
    print(f"[download] wrote {dst} ({len(data)} bytes)")


def as_trimesh(obj) -> trimesh.Trimesh:
    if isinstance(obj, trimesh.Trimesh):
        mesh = obj
    elif isinstance(obj, trimesh.Scene):
        dumped = obj.dump(concatenate=True)
        if isinstance(dumped, trimesh.Trimesh):
            mesh = dumped
        else:
            geometries = [g for g in dumped if isinstance(g, trimesh.Trimesh)]
            if not geometries:
                raise RuntimeError("Scene did not contain any mesh geometry")
            mesh = trimesh.util.concatenate(geometries)
    else:
        raise TypeError(f"Unsupported trimesh load result: {type(obj)}")

    if mesh.vertices is None or mesh.faces is None:
        raise RuntimeError("Mesh has no vertices or faces")
    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise RuntimeError("Mesh has zero vertices or faces")
    if mesh.faces.ndim != 2 or mesh.faces.shape[1] != 3:
        raise RuntimeError(f"Expected triangular faces, got shape {mesh.faces.shape}")

    return mesh


def load_mesh(path: Path) -> trimesh.Trimesh:
    print(f"[load] {path}")
    obj = trimesh.load(path, force="mesh", process=False)
    mesh = as_trimesh(obj)
    print(f"[load] vertices={len(mesh.vertices)} faces={len(mesh.faces)}")
    return mesh


def bounds_dict(mesh: trimesh.Trimesh) -> dict:
    b = np.asarray(mesh.bounds, dtype=float)
    extents = np.asarray(mesh.extents, dtype=float)
    return {
        "min": b[0].tolist(),
        "max": b[1].tolist(),
        "extents": extents.tolist(),
    }


def convert_y_up_to_z_up(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    converted = mesh.copy()
    converted.apply_transform(Y_UP_TO_Z_UP)
    return converted


def run_coacd(mesh: trimesh.Trimesh):
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int32)

    coacd_mesh = coacd.Mesh(vertices, faces)
    print("[coacd] running decomposition ...")
    parts = coacd.run_coacd(coacd_mesh)
    print(f"[coacd] num_parts={len(parts)}")
    if not parts:
        raise RuntimeError("CoACD returned zero parts")
    return parts


def part_to_trimesh(part) -> trimesh.Trimesh:
    # CoACD Python API commonly returns list[(vertices, faces)].
    # This wrapper also tolerates object-like returns.
    if isinstance(part, (tuple, list)) and len(part) >= 2:
        vertices, faces = part[0], part[1]
    elif hasattr(part, "vertices") and hasattr(part, "faces"):
        vertices, faces = part.vertices, part.faces
    else:
        raise TypeError(f"Unsupported CoACD part type: {type(part)}")

    m = trimesh.Trimesh(
        vertices=np.asarray(vertices, dtype=np.float64),
        faces=np.asarray(faces, dtype=np.int64),
        process=False,
    )
    if len(m.vertices) == 0 or len(m.faces) == 0:
        raise RuntimeError("CoACD produced an empty part")
    return m


def write_parts(parts, out_dir: Path) -> list[dict]:
    part_infos = []
    for i, part in enumerate(parts):
        mesh = part_to_trimesh(part)
        part_path = out_dir / f"part_{i:03d}.obj"
        mesh.export(part_path)

        info = {
            "index": i,
            "file": part_path.name,
            "num_vertices": int(len(mesh.vertices)),
            "num_faces": int(len(mesh.faces)),
            "bounds": bounds_dict(mesh),
        }
        part_infos.append(info)
        print(
            f"[write] {part_path} "
            f"vertices={info['num_vertices']} faces={info['num_faces']}"
        )
    return part_infos


def process_one(url: str, index: int, out_root: Path) -> dict:
    name = sanitize_name(url, index)
    out_dir = out_root / name
    out_dir.mkdir(parents=True, exist_ok=True)

    source_path = out_dir / "source.glb"
    converted_path = out_dir / "converted_zup.obj"
    manifest_path = out_dir / "manifest.json"

    download(url, source_path)

    mesh_yup = load_mesh(source_path)
    before_bounds = bounds_dict(mesh_yup)

    mesh_zup = convert_y_up_to_z_up(mesh_yup)
    after_bounds = bounds_dict(mesh_zup)

    mesh_zup.export(converted_path)
    print(f"[write] {converted_path}")

    parts = run_coacd(mesh_zup)
    part_infos = write_parts(parts, out_dir)

    manifest = {
        "url": url,
        "name": name,
        "source_file": source_path.name,
        "converted_file": converted_path.name,
        "num_vertices": int(len(mesh_zup.vertices)),
        "num_faces": int(len(mesh_zup.faces)),
        "num_parts": int(len(part_infos)),
        "axis_transform": {
            "description": "Y-up to Z-up, right-handed: (x, y, z) -> (x, -z, y)",
            "matrix_row_major": Y_UP_TO_Z_UP.tolist(),
        },
        "bounds_before_y_up": before_bounds,
        "bounds_after_z_up": after_bounds,
        "parts": part_infos,
    }

    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"[write] {manifest_path}")

    return manifest


def self_test() -> None:
    p = np.array([[1.0, 2.0, 3.0]], dtype=np.float64)
    ph = np.concatenate([p, np.ones((1, 1), dtype=np.float64)], axis=1)
    out = (Y_UP_TO_Z_UP @ ph.T).T[:, :3]
    expected = np.array([[1.0, -3.0, 2.0]], dtype=np.float64)

    if not np.allclose(out, expected):
        raise AssertionError(f"axis transform failed: got {out}, expected {expected}")

    # Extent mapping check with a simple non-symmetric box.
    box = trimesh.creation.box(extents=(1.0, 2.0, 3.0))
    before = np.asarray(box.extents)
    converted = convert_y_up_to_z_up(box)
    after = np.asarray(converted.extents)

    expected_extents = np.array([before[0], before[2], before[1]])
    if not np.allclose(after, expected_extents):
        raise AssertionError(
            f"extent mapping failed: got {after}, expected {expected_extents}"
        )

    print("[self-test] Y-up -> Z-up transform passed")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--filelist", type=Path, default=Path("filelist.txt"))
    parser.add_argument("--out", type=Path, default=Path("outputs"))
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return 0

    urls = read_filelist(args.filelist)
    args.out.mkdir(parents=True, exist_ok=True)

    all_manifests = []
    for i, url in enumerate(urls):
        print("=" * 80)
        print(f"[model {i + 1}/{len(urls)}] {url}")
        manifest = process_one(url, i, args.out)
        all_manifests.append(manifest)

    summary_path = args.out / "summary.json"
    summary = {
        "num_models": len(all_manifests),
        "models": all_manifests,
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    print("=" * 80)
    print(f"[done] wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
