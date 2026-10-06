"""Strip the vision encoder from an Ollama gemma3-family GGUF (setup tool).

Writes a copy of the GGUF without the `v.*` (CLIP vision tower) and `mm.*`
(projection) tensors, keeping every metadata key and all text tensors
byte-identical. The result is the same text model (e.g. MedGemma 1.5 4B),
just without the multimodal projector that Ollama would otherwise load,
warm up and keep resident (~1 GiB) even for text-only traffic.

One-time setup utility, not part of the runtime package. Needs the `gguf`
package: `pip install gguf`. See AGENTS.md "Models".

Usage:
    python scripts/strip_vision_gguf.py --src <blob.gguf> --out <text-only.gguf>
"""

import argparse
import struct

import numpy as np
from gguf import GGUFReader, GGUFValueType, GGMLQuantizationType, GGUFWriter

_DROP_PREFIXES = ("v.", "mm.")


def _read_exact(f, n):
    b = f.read(n)
    if len(b) != n:
        raise EOFError
    return b


def parse_kv(path):
    """Independent minimal GGUF v2/v3 metadata parser -> [(key, vtype, value)].

    Strings decode to str, scalars to int/float/bool, arrays to
    (item_type, list). Avoids gguf-py ReaderField.data, whose array contents
    are not decoded values in the installed version.
    """
    with open(path, "rb") as f:
        if _read_exact(f, 4) != b"GGUF":
            raise ValueError("not a GGUF file")
        _ver, _tc, kc = struct.unpack("<IQQ", _read_exact(f, 20))

        def rstr():
            (n,) = struct.unpack("<Q", _read_exact(f, 8))
            return _read_exact(f, n).decode("utf-8")

        def rval(t):
            fmt = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i",
                   6: "<f", 10: "<Q", 11: "<q", 12: "<d"}
            if t in fmt:
                (v,) = struct.unpack(fmt[t], _read_exact(f, struct.calcsize(fmt[t])))
                return v
            if t == 7:
                return _read_exact(f, 1) != b"\x00"
            if t == 8:
                return rstr()
            if t == 9:
                (it,) = struct.unpack("<I", _read_exact(f, 4))
                (n,) = struct.unpack("<Q", _read_exact(f, 8))
                return it, [rval(it) for _ in range(n)]
            raise ValueError(f"unsupported value type {t}")

        out = []
        for _ in range(kc):
            key = rstr()
            (t,) = struct.unpack("<I", _read_exact(f, 4))
            out.append((key, t, rval(t)))
        return out


def write_stripped(src, out):
    kvs = parse_kv(src)
    arch = next(v for k, t, v in kvs if k == "general.architecture")

    reader = GGUFReader(src)
    kept = [t for t in reader.tensors if not t.name.startswith(_DROP_PREFIXES)]
    dropped = len(reader.tensors) - len(kept)

    writer = GGUFWriter(out, arch, use_temp_file=True)
    nkv = 0
    for key, t, val in kvs:
        if key == "general.architecture":
            continue
        if t == 9:
            sub_type, items = val
            writer.add_key_value(key, items, GGUFValueType.ARRAY, sub_type=sub_type)
        else:
            writer.add_key_value(key, val, GGUFValueType(t))
        nkv += 1
    for t in kept:
        byte_shape = t.data.shape if t.data.dtype == np.uint8 else None
        writer.add_tensor(t.name, t.data, raw_shape=byte_shape,
                          raw_dtype=GGMLQuantizationType(t.tensor_type))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=False)
    writer.close()
    return nkv, len(kept), dropped


def verify(src, out):
    src_kvs = {k: (t, v) for k, t, v in parse_kv(src)}
    out_kvs = {k: (t, v) for k, t, v in parse_kv(out)}
    assert set(src_kvs) == set(out_kvs), (
        f"kv keys differ: {set(src_kvs) ^ set(out_kvs)}")
    for k, (t, v) in src_kvs.items():
        assert out_kvs[k] == (t, v), f"kv value changed for {k}"

    src_r = GGUFReader(src)
    out_r = GGUFReader(out)
    src_t = {t.name: t for t in src_r.tensors if not t.name.startswith(_DROP_PREFIXES)}
    out_t = {t.name: t for t in out_r.tensors}
    assert set(src_t) == set(out_t), (
        f"tensor names differ: {set(src_t) ^ set(out_t)}")
    assert not any(n.startswith(_DROP_PREFIXES) for n in out_t), "vision tensor leaked"
    checked = 0
    for name, st in src_t.items():
        ot = out_t[name]
        assert (list(st.shape) == list(ot.shape)
                and st.tensor_type == ot.tensor_type), name
        if bytes(st.data[:65536]) != bytes(ot.data[:65536]):
            raise AssertionError(f"tensor data mismatch: {name}")
        checked += 1
    return len(out_kvs), checked


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    nkv, kept, dropped = write_stripped(args.src, args.out)
    print(f"written: {args.out}")
    print(f"  kv entries copied : {nkv} (all)")
    print(f"  tensors kept      : {kept}")
    print(f"  tensors dropped   : {dropped} ({'/'.join(_DROP_PREFIXES)}*)")

    vkv, vtensors = verify(args.src, args.out)
    print(f"verified: {vkv} kv entries equal, {vtensors} tensors "
          f"identical (shape, dtype, first 64 KiB)")


if __name__ == "__main__":
    main()
