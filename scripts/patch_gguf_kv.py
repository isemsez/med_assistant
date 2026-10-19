"""Add missing scalar metadata keys to a GGUF (setup tool).

llama.cpp occasionally promotes an optional hyperparameter to required (e.g.
`gemma3.attention.layer_norm_rms_epsilon`, required since ~b11400); GGUFs
produced by Ollama's fork or older converters lack the key even though the
runtime default they were served with (1e-6) is the correct value. This
writes a copy of the GGUF with the key added, keeping every other metadata
entry and all tensors byte-identical. Idempotent: a present key with the
same value is a no-op; a conflicting value is an error.

One-time setup utility, not part of the runtime package. Needs the `gguf`
package (`pip install gguf`). See AGENTS.md "Models".

Usage:
    python scripts/patch_gguf_kv.py --src in.gguf --out out.gguf \
        --kv gemma3.attention.layer_norm_rms_epsilon:float:1e-6 [--kv ...]
    python scripts/patch_gguf_kv.py --src in.gguf --in-place --kv ...
"""

import argparse

import numpy as np
from gguf import GGMLQuantizationType, GGUFReader, GGUFValueType, GGUFWriter

from strip_vision_gguf import parse_kv

_TYPES = {"float": GGUFValueType.FLOAT32, "int": GGUFValueType.INT32,
          "uint": GGUFValueType.UINT32, "bool": GGUFValueType.BOOL,
          "string": GGUFValueType.STRING}


def _parse_kv_arg(spec):
    key, tname, value = spec.split(":", 2)
    if tname not in _TYPES:
        raise ValueError(f"--kv type must be one of {sorted(_TYPES)}: {spec}")
    t = _TYPES[tname]
    if t == GGUFValueType.FLOAT32:
        val = float(value)
    elif t in (GGUFValueType.INT32, GGUFValueType.UINT32):
        val = int(value, 0)
    elif t == GGUFValueType.BOOL:
        val = value.lower() in {"1", "true", "yes"}
    else:
        val = value
    return key, t, val


def write_patched(src, out, patches):
    kvs = parse_kv(src)
    arch = next(v for k, t, v in kvs if k == "general.architecture")

    for key, t, val in patches:
        existing = next((v for k, vt, v in kvs if k == key), None)
        if existing is not None:
            if existing == val:
                print(f"  {key} already {val!r} - nothing to do for this key")
            else:
                raise SystemExit(f"refusing to change existing {key}: "
                                 f"{existing!r} -> {val!r}")
            patches = [p for p in patches if p[0] != key]

    reader = GGUFReader(src)
    writer = GGUFWriter(out, arch, use_temp_file=True)
    for key, t, val in kvs:
        if key == "general.architecture":
            continue
        if t == 9:
            sub_type, items = val
            writer.add_key_value(key, items, GGUFValueType.ARRAY, sub_type=sub_type)
        else:
            writer.add_key_value(key, val, GGUFValueType(t))
    for key, t, val in patches:  # appended after the copied keys
        writer.add_key_value(key, val, t)
    for tn in reader.tensors:
        byte_shape = tn.data.shape if tn.data.dtype == np.uint8 else None
        writer.add_tensor(tn.name, tn.data, raw_shape=byte_shape,
                          raw_dtype=GGMLQuantizationType(tn.tensor_type))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=False)
    writer.close()
    return len(patches)


def verify(src, out, patches):
    src_kvs = {k: (t, v) for k, t, v in parse_kv(src)}
    out_kvs = {k: (t, v) for k, t, v in parse_kv(out)}
    assert set(out_kvs) == set(src_kvs) | {k for k, _, _ in patches}, (
        f"kv keys differ: {set(out_kvs) ^ (set(src_kvs) | {k for k, _, _ in patches})}")
    for k in src_kvs:
        assert out_kvs[k] == src_kvs[k], f"kv value changed for {k}"

    src_r, out_r = GGUFReader(src), GGUFReader(out)
    assert [t.name for t in src_r.tensors] == [t.name for t in out_r.tensors], \
        "tensor list changed"
    for a, b in zip(src_r.tensors, out_r.tensors):
        assert list(a.shape) == list(b.shape) and a.tensor_type == b.tensor_type, a.name
        if bytes(a.data[:65536]) != bytes(b.data[:65536]):
            raise AssertionError(f"tensor data mismatch: {a.name}")
    return len(out_r.tensors)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", required=True)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--out")
    g.add_argument("--in-place", action="store_true")
    p.add_argument("--kv", action="append", required=True,
                   metavar="KEY:TYPE:VALUE",
                   help="scalar key to add, e.g. "
                        "gemma3.attention.layer_norm_rms_epsilon:float:1e-6")
    args = p.parse_args()

    patches = [_parse_kv_arg(spec) for spec in args.kv]
    out = args.src if args.in_place else args.out
    tmp = out + ".patched.tmp" if args.in_place else out

    added = write_patched(args.src, tmp, patches)
    n = verify(args.src, tmp, patches)
    if args.in_place:
        import os
        os.replace(tmp, out)
    print(f"written: {out}")
    print(f"  keys added : {added}")
    print(f"  tensors    : {n} identical (shape, dtype, first 64 KiB)")


if __name__ == "__main__":
    main()
