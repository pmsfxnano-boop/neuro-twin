"""Minimal NIfTI-1 reader for reproducible feature extraction.

This module deliberately covers the common scalar NIfTI-1 image types needed by
this research pipeline without introducing a hard nibabel dependency. It is not
a replacement for a full NIfTI implementation: unsupported compound datatypes
and NIfTI-2 are rejected explicitly.
"""
from __future__ import annotations

from dataclasses import dataclass
import gzip
from pathlib import Path
import struct
from typing import BinaryIO

import numpy as np


class NiftiFormatError(ValueError):
    """Raised when a file is not a supported NIfTI-1 image."""


_DTYPE_MAP = {
    2: np.dtype("u1"),
    256: np.dtype("i1"),
    4: np.dtype("i2"),
    512: np.dtype("u2"),
    8: np.dtype("i4"),
    768: np.dtype("u4"),
    16: np.dtype("f4"),
    64: np.dtype("f8"),
    1024: np.dtype("i8"),
}


def _open(path: Path) -> BinaryIO:
    if path.name.endswith(".nii.gz"):
        return gzip.open(path, "rb")
    return path.open("rb")


def _read_exact(fh: BinaryIO, n: int) -> bytes:
    raw = fh.read(n)
    if len(raw) != n:
        raise NiftiFormatError(f"unexpected EOF: needed {n} bytes, got {len(raw)}")
    return raw


def _unpack(fmt_le: str, fmt_be: str, raw: bytes, little: bool):
    return struct.unpack(("<" if little else ">") + fmt_le, raw)[0]


@dataclass(frozen=True)
class NiftiHeader:
    endian: str
    sizeof_hdr: int
    dim: tuple[int, ...]
    pixdim: tuple[float, ...]
    datatype: int
    bitpix: int
    vox_offset: float
    scl_slope: float
    scl_inter: float
    qform_code: int
    sform_code: int
    quatern_b: float
    quatern_c: float
    quatern_d: float
    qoffset_x: float
    qoffset_y: float
    qoffset_z: float
    srow_x: tuple[float, float, float, float]
    srow_y: tuple[float, float, float, float]
    srow_z: tuple[float, float, float, float]
    magic: bytes

    @property
    def shape(self) -> tuple[int, ...]:
        ndim = self.dim[0]
        if ndim < 1 or ndim > 7:
            raise NiftiFormatError(f"invalid NIfTI dimensionality {ndim}")
        shape = tuple(int(x) for x in self.dim[1 : ndim + 1])
        if any(v <= 0 for v in shape):
            raise NiftiFormatError(f"invalid NIfTI dimensions {shape}")
        return shape

    @property
    def dtype(self) -> np.dtype:
        try:
            base = _DTYPE_MAP[self.datatype]
        except KeyError as exc:
            raise NiftiFormatError(f"unsupported scalar datatype {self.datatype}") from exc
        return base.newbyteorder("<" if self.endian == "little" else ">")

    @property
    def sform_affine(self) -> np.ndarray:
        if self.sform_code <= 0:
            raise NiftiFormatError("sform is not declared")
        affine = np.array([self.srow_x, self.srow_y, self.srow_z, (0.0, 0.0, 0.0, 1.0)], dtype=float)
        if not np.all(np.isfinite(affine)):
            raise NiftiFormatError("invalid sform affine")
        return affine

    @property
    def qform_affine(self) -> np.ndarray:
        if self.qform_code <= 0:
            raise NiftiFormatError("qform is not declared")
        zooms = np.array(self.pixdim[1:4], dtype=float)
        if len(zooms) != 3 or np.any(~np.isfinite(zooms)) or np.any(zooms <= 0):
            raise NiftiFormatError("invalid qform voxel sizes")
        b, c, d = float(self.quatern_b), float(self.quatern_c), float(self.quatern_d)
        if not np.all(np.isfinite([b, c, d])):
            raise NiftiFormatError("invalid qform quaternion")
        norm2 = b * b + c * c + d * d
        a2 = 1.0 - norm2
        if a2 < -1e-6:
            raise NiftiFormatError("invalid qform quaternion norm")
        a = float(np.sqrt(max(a2, 0.0)))
        r = np.array([
            [a*a + b*b - c*c - d*d, 2.0*(b*c - a*d), 2.0*(b*d + a*c)],
            [2.0*(b*c + a*d), a*a + c*c - b*b - d*d, 2.0*(c*d - a*b)],
            [2.0*(b*d - a*c), 2.0*(c*d + a*b), a*a + d*d - b*b - c*c],
        ], dtype=float)
        qfac = -1.0 if self.pixdim[0] < 0 else 1.0
        r[:, 0] *= zooms[0]
        r[:, 1] *= zooms[1]
        r[:, 2] *= zooms[2] * qfac
        affine = np.eye(4, dtype=float)
        affine[:3, :3] = r
        affine[:3, 3] = [self.qoffset_x, self.qoffset_y, self.qoffset_z]
        if not np.all(np.isfinite(affine)):
            raise NiftiFormatError("invalid qform affine")
        return affine

    @property
    def affine(self) -> np.ndarray:
        if self.sform_code > 0:
            return self.sform_affine
        if self.qform_code > 0:
            return self.qform_affine
        # Neither form is declared. This is a conservative voxel-scaling fallback;
        # callers must not interpret it as a registration anchor.
        zooms = np.array(self.pixdim[1:4], dtype=float)
        if len(zooms) != 3 or np.any(~np.isfinite(zooms)) or np.any(zooms <= 0):
            raise NiftiFormatError("invalid voxel sizes")
        return np.array(
            [[zooms[0], 0.0, 0.0, 0.0],
             [0.0, zooms[1], 0.0, 0.0],
             [0.0, 0.0, zooms[2], 0.0],
             [0.0, 0.0, 0.0, 1.0]],
            dtype=float,
        )

    @property
    def voxel_volume_mm3(self) -> float:
        det = float(abs(np.linalg.det(self.affine[:3, :3])))
        if not np.isfinite(det) or det <= 0:
            fallback = np.prod(np.abs(self.pixdim[1:4]))
            det = float(fallback)
        if not np.isfinite(det) or det <= 0:
            raise NiftiFormatError("invalid voxel volume")
        return det


@dataclass(frozen=True)
class NiftiImage:
    path: str
    header: NiftiHeader
    data: np.ndarray

    @property
    def affine(self) -> np.ndarray:
        return self.header.affine


def _parse_header(raw: bytes) -> NiftiHeader:
    if len(raw) != 348:
        raise NiftiFormatError("NIfTI-1 header must be 348 bytes")
    size_le = struct.unpack("<i", raw[0:4])[0]
    size_be = struct.unpack(">i", raw[0:4])[0]
    if size_le == 348:
        endian = "little"
        prefix = "<"
    elif size_be == 348:
        endian = "big"
        prefix = ">"
    else:
        raise NiftiFormatError("not a NIfTI-1 header (sizeof_hdr != 348)")

    def i16(off: int) -> int:
        return struct.unpack(prefix + "h", raw[off : off + 2])[0]

    def u8(off: int) -> int:
        return raw[off]

    def i32(off: int) -> int:
        return struct.unpack(prefix + "i", raw[off : off + 4])[0]

    def f32(off: int) -> float:
        return struct.unpack(prefix + "f", raw[off : off + 4])[0]

    dim = tuple(i16(40 + 2 * i) for i in range(8))
    pixdim = tuple(f32(76 + 4 * i) for i in range(8))
    srow_x = tuple(f32(280 + 4 * i) for i in range(4))
    srow_y = tuple(f32(296 + 4 * i) for i in range(4))
    srow_z = tuple(f32(312 + 4 * i) for i in range(4))
    return NiftiHeader(
        endian=endian,
        sizeof_hdr=i32(0),
        dim=dim,
        pixdim=pixdim,
        datatype=i16(70),
        bitpix=i16(72),
        vox_offset=f32(108),
        scl_slope=f32(112),
        scl_inter=f32(116),
        qform_code=i16(252),
        sform_code=i16(254),
        quatern_b=f32(256),
        quatern_c=f32(260),
        quatern_d=f32(264),
        qoffset_x=f32(268),
        qoffset_y=f32(272),
        qoffset_z=f32(276),
        srow_x=srow_x,
        srow_y=srow_y,
        srow_z=srow_z,
        magic=bytes(raw[344:348]),
    )


def read_nifti(path: str | Path, *, load_data: bool = True) -> NiftiImage | NiftiHeader:
    """Read a NIfTI-1 image or header only.

    ``load_data=False`` is intended for fast integrity/geometry inspection.  For
    feature extraction, ``load_data=True`` applies NIfTI scaling when ``scl_slope``
    is finite and non-zero.
    """
    path = Path(path)
    with _open(path) as fh:
        header = _parse_header(_read_exact(fh, 348))
        magic = header.magic
        if magic[:3] not in (b"n+1", b"ni1"):
            raise NiftiFormatError(f"unsupported NIfTI magic {magic!r}")
        if not load_data:
            return header
        offset = int(round(header.vox_offset))
        if offset < 348:
            offset = 352
        fh.seek(offset)
        count = int(np.prod(header.shape))
        raw = fh.read(count * header.dtype.itemsize)
        if len(raw) != count * header.dtype.itemsize:
            raise NiftiFormatError("NIfTI payload is shorter than declared dimensions")
        data = np.frombuffer(raw, dtype=header.dtype, count=count).reshape(header.shape, order="C")
        data = np.asarray(data, dtype=np.float64)
        slope = float(header.scl_slope)
        intercept = float(header.scl_inter)
        if np.isfinite(slope) and slope != 0.0:
            data = data * slope + intercept
        return NiftiImage(str(path), header, data)
