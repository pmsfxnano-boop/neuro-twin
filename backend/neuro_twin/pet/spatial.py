"""Explicit PET spatial transforms into an MRI/native target space.

Registration estimation is deliberately out of scope. A trusted 4x4 affine must
be supplied by an external registration pipeline; this module applies it and
records quantitative-image resampling provenance.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

import numpy as np
from scipy.ndimage import affine_transform


PET_SPATIAL_VERSION = "0.1.0-b15"


@dataclass(frozen=True)
class PETSpatialTransform:
    version: str
    interpolation: str
    source_shape: tuple[int, ...]
    target_shape: tuple[int, ...]
    source_affine: tuple[tuple[float, ...], ...]
    target_affine: tuple[tuple[float, ...], ...]
    source_to_target_world: tuple[tuple[float, ...], ...]
    target_voxel_to_source_voxel: tuple[tuple[float, ...], ...]
    parameters_hash: str


def _hash_json(value) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def apply_affine_to_pet(
    pet_image: np.ndarray,
    source_affine: np.ndarray,
    target_shape: tuple[int, int, int],
    target_affine: np.ndarray,
    source_to_target_world: np.ndarray,
) -> tuple[np.ndarray, PETSpatialTransform]:
    """Resample a 3D or 4D quantitative PET image using a supplied world-space affine.

    ``source_to_target_world`` maps source-world coordinates to target-world
    coordinates. Continuous PET is interpolated linearly; no label-map nearest
    neighbour is used here.
    """
    pet = np.asarray(pet_image, dtype=float)
    if pet.ndim not in (3, 4):
        raise ValueError("PET image must be 3D or 4D")
    src_aff = np.asarray(source_affine, dtype=float)
    tgt_aff = np.asarray(target_affine, dtype=float)
    world = np.asarray(source_to_target_world, dtype=float)
    if src_aff.shape != (4, 4) or tgt_aff.shape != (4, 4) or world.shape != (4, 4):
        raise ValueError("affines must be 4x4")
    if not np.all(np.isfinite(np.vstack([src_aff, tgt_aff, world]))):
        raise ValueError("affines must be finite")
    if np.linalg.matrix_rank(src_aff) < 4 or np.linalg.matrix_rank(tgt_aff) < 4:
        raise ValueError("source/target affines must be invertible")

    # target voxel -> target world -> source world -> source voxel
    tgt_to_src_vox = np.linalg.inv(src_aff) @ np.linalg.inv(world) @ tgt_aff
    matrix = tgt_to_src_vox[:3, :3]
    offset = tgt_to_src_vox[:3, 3]
    if pet.ndim == 3:
        out = affine_transform(pet, matrix=matrix, offset=offset, output_shape=target_shape, order=1, mode="constant", cval=np.nan)
    else:
        frames = [
            affine_transform(pet[..., i], matrix=matrix, offset=offset, output_shape=target_shape, order=1, mode="constant", cval=np.nan)
            for i in range(pet.shape[3])
        ]
        out = np.stack(frames, axis=3)

    params = {
        "version": PET_SPATIAL_VERSION,
        "interpolation": "linear",
        "source_shape": list(pet.shape),
        "target_shape": list(target_shape),
        "source_affine": src_aff.tolist(),
        "target_affine": tgt_aff.tolist(),
        "source_to_target_world": world.tolist(),
        "target_voxel_to_source_voxel": tgt_to_src_vox.tolist(),
    }
    prov = PETSpatialTransform(
        version=PET_SPATIAL_VERSION,
        interpolation="linear",
        source_shape=tuple(pet.shape),
        target_shape=tuple(target_shape),
        source_affine=tuple(tuple(float(x) for x in row) for row in src_aff),
        target_affine=tuple(tuple(float(x) for x in row) for row in tgt_aff),
        source_to_target_world=tuple(tuple(float(x) for x in row) for row in world),
        target_voxel_to_source_voxel=tuple(tuple(float(x) for x in row) for row in tgt_to_src_vox),
        parameters_hash=_hash_json(params),
    )
    return out, prov
