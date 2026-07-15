"""Fast GGUF reader - drop-in replacement for the ``gguf.GGUFReader`` library.

The upstream ``gguf`` library eagerly decodes every metadata field in
``GGUFReader.__init__``, including the ~776k tokenizer string entries
(``tokenizer.ggml.merges``/``.tokens``/``.scores``/``.token_type``), in pure Python
(~89 s per open on a 256k-vocab model). vLLM never consumes that tokenizer data
(it loads the tokenizer separately), so this reader **skips** ``tokenizer.ggml.*``
array/string decoding entirely via a size-only walker, while still parsing the
small scalar fields (notably ``general.alignment``) and the full tensor-info
section.

Result: ~89 s -> well under 1 s per open, byte-for-byte identical ``.tensors``.

This module is self-contained: it does not ``import gguf``. It exposes the
``GGUFReader``, ``ReaderTensor``, ``ReaderField``, ``GGMLQuantizationType``,
``GGUFValueType``, ``GGML_QUANT_SIZES``, ``quant_shape_to_byte_shape`` and
``quant_shape_from_byte_shape`` names that vLLM's GGUF loader consumes.
"""
import os
from enum import IntEnum
from typing import Any, NamedTuple

import numpy as np

GGUF_MAGIC = 0x46554747  # b"GGUF" little-endian
READER_SUPPORTED_VERSIONS = (3,)

# Default alignment (GGUF spec) - may be overridden by general.alignment field.
DEFAULT_ALIGNMENT = 32


class GGUFValueType(IntEnum):
    UINT8 = 0
    INT8 = 1
    UINT16 = 2
    INT16 = 3
    UINT32 = 4
    INT32 = 5
    FLOAT32 = 6
    BOOL = 7
    STRING = 8
    ARRAY = 9
    UINT64 = 10
    INT64 = 11
    FLOAT64 = 12


class GGMLQuantizationType(IntEnum):
    F32 = 0
    F16 = 1
    Q4_0 = 2
    Q4_1 = 3
    Q5_0 = 6
    Q5_1 = 7
    Q8_0 = 8
    Q8_1 = 9
    Q2_K = 10
    Q3_K = 11
    Q4_K = 12
    Q5_K = 13
    Q6_K = 14
    Q8_K = 15
    IQ2_XXS = 16
    IQ2_XS = 17
    IQ3_XXS = 18
    IQ1_S = 19
    IQ4_NL = 20
    IQ3_S = 21
    IQ2_S = 22
    IQ4_XS = 23
    I8 = 24
    I16 = 25
    I32 = 26
    I64 = 27
    F64 = 28
    IQ1_M = 29
    BF16 = 30


# (block_size, type_size_in_bytes) per GGML quant type.
GGML_QUANT_SIZES = {
    GGMLQuantizationType.F32: (1, 4),
    GGMLQuantizationType.F16: (1, 2),
    GGMLQuantizationType.Q4_0: (32, 18),
    GGMLQuantizationType.Q4_1: (32, 20),
    GGMLQuantizationType.Q5_0: (32, 22),
    GGMLQuantizationType.Q5_1: (32, 24),
    GGMLQuantizationType.Q8_0: (32, 34),
    GGMLQuantizationType.Q8_1: (32, 36),
    GGMLQuantizationType.Q2_K: (256, 84),
    GGMLQuantizationType.Q3_K: (256, 110),
    GGMLQuantizationType.Q4_K: (256, 144),
    GGMLQuantizationType.Q5_K: (256, 176),
    GGMLQuantizationType.Q6_K: (256, 210),
    GGMLQuantizationType.Q8_K: (256, 292),
    GGMLQuantizationType.IQ2_XXS: (256, 66),
    GGMLQuantizationType.IQ2_XS: (256, 74),
    GGMLQuantizationType.IQ3_XXS: (256, 98),
    GGMLQuantizationType.IQ1_S: (256, 50),
    GGMLQuantizationType.IQ4_NL: (18, 24),
    GGMLQuantizationType.IQ3_S: (256, 110),
    GGMLQuantizationType.IQ2_S: (256, 82),
    GGMLQuantizationType.IQ4_XS: (256, 136),
    GGMLQuantizationType.I8: (1, 1),
    GGMLQuantizationType.I16: (1, 2),
    GGMLQuantizationType.I32: (1, 4),
    GGMLQuantizationType.I64: (1, 8),
    GGMLQuantizationType.F64: (1, 8),
    GGMLQuantizationType.IQ1_M: (256, 56),
    GGMLQuantizationType.BF16: (1, 2),
}

# Map scalar GGUF value type -> numpy dtype.
_SCALAR_NP = {
    GGUFValueType.UINT8: np.uint8,
    GGUFValueType.INT8: np.int8,
    GGUFValueType.UINT16: np.uint16,
    GGUFValueType.INT16: np.int16,
    GGUFValueType.UINT32: np.uint32,
    GGUFValueType.INT32: np.int32,
    GGUFValueType.FLOAT32: np.float32,
    GGUFValueType.BOOL: np.bool_,
    GGUFValueType.UINT64: np.uint64,
    GGUFValueType.INT64: np.int64,
    GGUFValueType.FLOAT64: np.float64,
}

# Types that get a typed numpy view (matching the upstream library's _build_tensors).
_TYPED_VIEWS = {
    GGMLQuantizationType.F16: np.float16,
    GGMLQuantizationType.F32: np.float32,
    GGMLQuantizationType.F64: np.float64,
    GGMLQuantizationType.I8: np.int8,
    GGMLQuantizationType.I16: np.int16,
    GGMLQuantizationType.I32: np.int32,
    GGMLQuantizationType.I64: np.int64,
}


def quant_shape_to_byte_shape(shape, quant_type: GGMLQuantizationType):
    block_size, type_size = GGML_QUANT_SIZES[quant_type]
    if shape[-1] % block_size != 0:
        raise ValueError(
            f"Quantized tensor shape {shape} last dim not divisible by "
            f"{quant_type.name} block size {block_size}")
    return (*shape[:-1], shape[-1] // block_size * type_size)


def quant_shape_from_byte_shape(shape, quant_type: GGMLQuantizationType):
    block_size, type_size = GGML_QUANT_SIZES[quant_type]
    if shape[-1] % type_size != 0:
        raise ValueError(
            f"Quantized byte shape {shape} last dim not divisible by "
            f"{quant_type.name} type size {type_size}")
    return (*shape[:-1], shape[-1] // type_size * block_size)


class ReaderField(NamedTuple):
    offset: int
    name: str
    parts: list = []
    data: list = [-1]
    types: list = []


class ReaderTensor(NamedTuple):
    name: str
    tensor_type: GGMLQuantizationType
    shape: Any
    n_elements: int
    n_bytes: int
    data_offset: int
    data: Any
    field: Any


class GGUFReader:
    """Minimal, fast GGUF reader.

    Only ``.tensors`` and ``general.alignment`` are consumed by vLLM, so
    ``tokenizer.ggml.*`` metadata arrays are walked by size only (no string
    decoding) - this is what makes construction ~100x faster than the upstream
    library on large-vocab models.
    """

    def __init__(self, path: os.PathLike, mode: str = "r"):
        self.data = np.memmap(path, mode=mode)
        self.fields: dict = {}
        self.tensors: list = []
        # Default to in-order (host little-endian); refined by the magic check.
        self.byte_order = "I" if sys_little_endian() else "S"

        offs = 0
        magic = self._get(offs, np.uint32)
        if int(magic[0]) != GGUF_MAGIC:
            # Try opposite byte order (matches upstream's behavior).
            swapped = magic.view(magic.dtype.newbyteorder("S"))
            if int(swapped[0]) == GGUF_MAGIC:
                self.byte_order = "S"
                magic = swapped
            else:
                raise ValueError("GGUF magic invalid")
        offs += 4

        version = self._get(offs, np.uint32)
        if int(version[0]) not in READER_SUPPORTED_VERSIONS:
            raise ValueError(
                f"GGUF version {int(version[0])} not supported")
        offs += 4

        temp_counts = self._get(offs, np.uint64, 2)
        offs += 16
        tensor_count, kv_count = int(temp_counts[0]), int(temp_counts[1])

        offs = self._build_fields(offs, kv_count)
        offs, tensor_fields = self._build_tensor_info(offs, tensor_count)

        new_align = self.fields.get("general.alignment")
        if new_align is not None:
            alignment = int(new_align.parts[-1][0])
            if alignment == 0 or (alignment & (alignment - 1)) != 0:
                raise ValueError("Invalid general.alignment: must be power of 2")
            self.alignment = alignment
        else:
            self.alignment = DEFAULT_ALIGNMENT
        padding = offs % self.alignment
        if padding != 0:
            offs += self.alignment - padding
        self.data_offset = offs

        self._build_tensors(offs, tensor_fields)

    # -- public helpers ----------------------------------------------------

    def get_field(self, key: str):
        return self.fields.get(key, None)

    def __getitem__(self, idx: int):
        return self.tensors[idx]

    # -- low-level reads ---------------------------------------------------

    def _get(self, offset, dtype, count=1):
        itemsize = np.dtype(dtype).itemsize
        end = offset + itemsize * count
        arr = self.data[offset:end].view(dtype=dtype)[:count]
        if self.byte_order == "S":
            arr = arr.view(arr.dtype.newbyteorder("S"))
        return arr

    def _get_str(self, offset):
        slen = self._get(offset, np.uint64)
        return slen, self._get(offset + 8, np.uint8, int(slen[0]))

    # -- field walkers -----------------------------------------------------

    def _skip_field(self, offs: int, vtype: int) -> int:
        """Walk a field by size only (no string decode). For tokenizer arrays."""
        gt = GGUFValueType(vtype)
        if gt == GGUFValueType.STRING:
            slen = self._get(offs, np.uint64)
            return offs + 8 + int(slen[0])
        if gt == GGUFValueType.ARRAY:
            itype = self._get(offs, np.uint32)
            offs += 4
            alen = self._get(offs, np.uint64)
            offs += 8
            n = int(alen[0])
            elem_type = GGUFValueType(int(itype[0]))
            # String arrays are variable-length: walk lengths. Use a tight loop
            # over the raw memmap buffer (int.from_bytes) - ~50x faster than a
            # numpy slice per element across ~776k tokenizer entries.
            if elem_type == GGUFValueType.STRING:
                buf = memoryview(self.data)
                for _ in range(n):
                    slen = int.from_bytes(buf[offs:offs + 8], "little")
                    offs += 8 + slen
                return offs
            # Fixed-size scalar array: bulk skip.
            itemsize = np.dtype(_SCALAR_NP[elem_type]).itemsize
            return offs + n * itemsize
        # Scalar.
        return offs + np.dtype(_SCALAR_NP[gt]).itemsize

    def _get_field_parts(self, offs: int, vtype: int):
        """Full parse of one field value -> (size_bytes, parts, idxs, types)."""
        orig_offs = offs
        types = [GGUFValueType(vtype)]
        if GGUFValueType(vtype) == GGUFValueType.STRING:
            sparts = list(self._get_str(offs))
            size = sum(int(part.nbytes) for part in sparts)
            return size, sparts, [1], types
        if GGUFValueType(vtype) == GGUFValueType.ARRAY:
            itype = self._get(offs, np.uint32)
            offs += int(itype.nbytes)
            alen = self._get(offs, np.uint64)
            offs += int(alen.nbytes)
            aparts = [itype, alen]
            data_idxs = []
            for idx in range(int(alen[0])):
                curr_size, curr_parts, curr_idxs, curr_types = \
                    self._get_field_parts(offs, int(itype[0]))
                if idx == 0:
                    types += curr_types
                idxs_offs = len(aparts)
                aparts += curr_parts
                data_idxs += (idx + idxs_offs for idx in curr_idxs)
                offs += curr_size
            return offs - orig_offs, aparts, data_idxs, types
        nptype = _SCALAR_NP.get(GGUFValueType(vtype))
        if nptype is not None:
            val = self._get(offs, nptype)
            return int(val.nbytes), [val], [0], types
        raise ValueError(f"Unknown/unhandled field type {GGUFValueType(vtype)}")

    def _push_field(self, field: ReaderField) -> int:
        if field.name in self.fields:
            raise KeyError(
                f"Duplicate {field.name} already in list at offset "
                f"{field.offset}")
        self.fields[field.name] = field
        return sum(int(part.nbytes) for part in field.parts)

    def _build_fields(self, offs: int, count: int) -> int:
        for _ in range(count):
            orig_offs = offs
            kv_klen, kv_kdata = self._get_str(offs)
            offs += int(kv_klen.nbytes + kv_kdata.nbytes)
            raw_kv_type = self._get(offs, np.uint32)
            offs += int(raw_kv_type.nbytes)
            # Skip tokenizer arrays/strings by size - vLLM never reads them and
            # decoding ~776k strings dominates construction time.
            key = str(bytes(kv_kdata), encoding="utf-8")
            if key.startswith("tokenizer.ggml."):
                offs = self._skip_field(offs, int(raw_kv_type[0]))
                continue
            field_size, field_parts, field_idxs, field_types = \
                self._get_field_parts(offs, int(raw_kv_type[0]))
            parts = [kv_klen, kv_kdata, raw_kv_type] + field_parts
            idxs_offs = 3
            self._push_field(ReaderField(
                orig_offs, key, parts,
                [idx + idxs_offs for idx in field_idxs], field_types))
            offs += field_size
        return offs

    # -- tensor info / data ------------------------------------------------

    def _get_tensor_info_field(self, orig_offs: int) -> ReaderField:
        offs = orig_offs
        name_len, name_data = self._get_str(offs)
        offs += int(name_len.nbytes + name_data.nbytes)
        n_dims = self._get(offs, np.uint32)
        offs += int(n_dims.nbytes)
        dims = self._get(offs, np.uint64, int(n_dims[0]))
        offs += int(dims.nbytes)
        raw_dtype = self._get(offs, np.uint32)
        offs += int(raw_dtype.nbytes)
        offset_tensor = self._get(offs, np.uint64)
        offs += int(offset_tensor.nbytes)
        return ReaderField(
            orig_offs, str(bytes(name_data), encoding="utf-8"),
            [name_len, name_data, n_dims, dims, raw_dtype, offset_tensor],
            [1, 3, 4, 5])

    def _build_tensor_info(self, offs: int, count: int):
        tensor_fields = []
        for _ in range(count):
            field = self._get_tensor_info_field(offs)
            offs += sum(int(part.nbytes) for part in field.parts)
            tensor_fields.append(field)
        return offs, tensor_fields

    def _build_tensors(self, start_offs: int, fields: list) -> None:
        tensors = []
        for field in fields:
            _name_len, name_data, _n_dims, dims, raw_dtype, offset_tensor = \
                field.parts
            ggml_type = GGMLQuantizationType(int(raw_dtype[0]))
            n_elems = int(np.prod(dims))
            np_dims = tuple(reversed(dims.tolist()))
            block_size, type_size = GGML_QUANT_SIZES[ggml_type]
            n_bytes = n_elems * type_size // block_size
            data_offs = int(start_offs + int(offset_tensor[0]))
            if ggml_type in _TYPED_VIEWS:
                item_count = n_elems
                item_type = _TYPED_VIEWS[ggml_type]
            else:
                item_count = n_bytes
                item_type = np.uint8
                np_dims = quant_shape_to_byte_shape(np_dims, ggml_type)
            tensors.append(ReaderTensor(
                name=field.name, tensor_type=ggml_type, shape=dims,
                n_elements=n_elems, n_bytes=n_bytes, data_offset=data_offs,
                data=self._get(data_offs, item_type, item_count).reshape(np_dims),
                field=field))
        self.tensors = tensors


def sys_little_endian() -> bool:
    import sys
    return sys.byteorder == "little"
