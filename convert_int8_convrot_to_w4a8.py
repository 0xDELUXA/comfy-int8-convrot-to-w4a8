#!/usr/bin/env python
"""Convert a ComfyUI int8_tensorwise convrot checkpoint to asym_w4a8_int8, without the
original fp16 weights.

Both formats rotate the weight with the same normalized regular Hadamard (ConvRot) before
quantizing, so the int8 payload already stores the tensor the W4A8 quantizer wants as its
input. The conversion re-quantizes in the rotated basis and never rotates at all.

Requires torch and safetensors only.

Usage:
    python convert_int8_convrot_to_w4a8.py --src IN.safetensors --dst OUT.safetensors
    python convert_int8_convrot_to_w4a8.py --src IN.safetensors --verify-only
"""

from __future__ import annotations

import argparse
import json
import os
import re
import struct
import sys
import time

import torch
from safetensors import safe_open

SRC_FORMAT = "int8_tensorwise"
DST_FORMAT = "asym_w4a8_int8"

_ST_ITEMSIZE = {"BOOL": 1, "I8": 1, "U8": 1, "I16": 2, "U16": 2, "F16": 2, "BF16": 2,
                "F8_E4M3": 1, "F8_E5M2": 1, "I32": 4, "U32": 4, "F32": 4, "I64": 8,
                "U64": 8, "F64": 8}

# Lloyd-Max-optimal 16 levels for a group-normalized Gaussian; ConvRot makes the rotated
# groups Gaussian, so this table matches a per-tensor fit for real models.
_FIXED_LUT = (
    -0.980602, -0.794529, -0.638165, -0.500986, -0.377321, -0.263187, -0.155210, -0.050720,
    0.052541, 0.156985, 0.265284, 0.379533, 0.502636, 0.638953, 0.794876, 0.980671,
)
_GATE_KURTOSIS = -0.1
_KURTOSIS_SAMPLE = 1 << 19
_ALS_ITERS = 2
_CODEBOOK_SAMPLE_ELEMS = 1 << 22
_ROW_ELEM_BUDGET = 1 << 22

# ComfyUI's quantized Embedding only loads per-row formats (fp8, int8_tensorwise), so a
# W4A8-packed table fails to load. The checkpoint does not record module types; match names.
_EMBEDDING_NAME = re.compile(
    r"(?:^|\.)(?:embed_tokens|token_embedding|position_embedding|word_embeddings|shared|wte|wpe)$"
)


def _fit_codebook(normalized, levels=16, iterations=25, sample_size=300000):
    samples = normalized.flatten()
    if samples.numel() > sample_size:
        gen = torch.Generator(device=samples.device).manual_seed(0)
        idx = torch.randint(0, samples.numel(), (sample_size,), device=samples.device,
                            generator=gen)
        samples = samples[idx]
    samples = samples.float()
    codebook = torch.quantile(samples, torch.linspace(0, 1, levels, device=samples.device))
    for _ in range(iterations):
        assignments = (samples.unsqueeze(-1) - codebook).abs().argmin(-1)
        updated = codebook.clone()
        for index in range(levels):
            selected = assignments == index
            if selected.any():
                updated[index] = samples[selected].mean()
        codebook = updated
    return codebook.contiguous()


def _codebook_for(normalized):
    """Frozen LUT unless the group-normalized weight is heavy-tailed, then fit one."""
    x = normalized.detach().flatten()
    if x.numel() > _KURTOSIS_SAMPLE:
        gen = torch.Generator(device=x.device).manual_seed(0)
        x = x[torch.randint(0, x.numel(), (_KURTOSIS_SAMPLE,), device=x.device, generator=gen)]
    x = x.float()
    excess_kurtosis = ((x - x.mean()) / (x.std() + 1e-9)).pow(4).mean() - 3.0
    if excess_kurtosis.item() <= _GATE_KURTOSIS:
        return torch.tensor(_FIXED_LUT, device=normalized.device, dtype=torch.float32)
    return _fit_codebook(normalized)


def _assign_codes(normalized, codebook):
    """Nearest codebook index; the codebook is sorted, so one searchsorted plus a compare."""
    last = codebook.numel() - 1
    pos = torch.searchsorted(codebook, normalized.contiguous())
    lo = (pos - 1).clamp(0, last)
    hi = pos.clamp(0, last)
    dlo = (normalized - codebook[lo]).abs_()
    dhi = (normalized - codebook[hi]).abs_()
    return torch.where(dhi < dlo, hi, lo).to(torch.int32)


def _assign_grid(grouped, levels, s_channel):
    """Nearest decoded INT8 level per weight, on the per-group level grid."""
    n, groups, gsize = grouped.shape
    last = levels.shape[-1] - 1
    lv = levels.reshape(n * groups, last + 1).contiguous()
    tg = (grouped / s_channel.view(-1, 1, 1)).reshape(n * groups, gsize).contiguous()
    pos = torch.searchsorted(lv, tg)
    lo = (pos - 1).clamp(0, last)
    hi = pos.clamp(0, last)
    dlo = tg.sub(torch.gather(lv, 1, lo)).abs_()
    dhi = tg.sub(torch.gather(lv, 1, hi)).abs_()
    return torch.where(dhi < dlo, hi, lo).to(torch.int32).reshape(n, groups, gsize)


def _decide_codebook(rotated, group_size):
    """One codebook per tensor, from evenly strided rows so chunking cannot change it."""
    n, k = rotated.shape
    max_rows = max(1, _CODEBOOK_SAMPLE_ELEMS // max(k, 1))
    if n > max_rows:
        idx = torch.linspace(0, n - 1, max_rows, device=rotated.device).long()
        sample = rotated.index_select(0, idx)
    else:
        sample = rotated
    grouped = sample.contiguous().float().view(sample.shape[0], k // group_size, group_size)
    normalized = grouped / grouped.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    return _codebook_for(normalized)


def _quantize_chunk(rotated, group_size, codebook):
    n, k = rotated.shape
    groups = k // group_size
    grouped = rotated.float().view(n, groups, group_size)

    group_scale = grouped.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    quantized = _assign_codes(grouped / group_scale, codebook)
    for _ in range(_ALS_ITERS):
        levels = codebook[quantized]
        group_scale = (
            (grouped * levels).sum(-1, keepdim=True)
            / (levels * levels).sum(-1, keepdim=True).clamp(min=1e-8)
        ).clamp(min=1e-8)
        quantized = _assign_codes(grouped / group_scale, codebook)

    shifted = codebook[quantized] * group_scale
    s_channel = (shifted.abs().amax(dim=(1, 2)) / 127.0).clamp(min=1e-8)
    s_rel = (group_scale.squeeze(-1) / s_channel.unsqueeze(1)).float().contiguous()
    s_rel = s_rel.to(torch.float8_e4m3fn).contiguous()

    grid = (codebook.view(1, 1, 16) * s_rel.float().unsqueeze(-1)).round_().clamp_(-127, 127)
    unsigned = _assign_grid(grouped, grid, s_channel).view(n, k)
    packed = (
        ((unsigned[:, 0::2] & 0xF) | ((unsigned[:, 1::2] & 0xF) << 4)).to(torch.int8).contiguous()
    )
    return packed, s_rel, s_channel.float().contiguous()


def quantize_rotated(rotated, group_size):
    """Pack an already-rotated weight into W4A8 storage, in row chunks to cap peak memory."""
    n, k = rotated.shape
    codebook = _decide_codebook(rotated, group_size)
    block = max(1, _ROW_ELEM_BUDGET // max(k, 1))
    if n <= block:
        packed, s_rel, s_channel = _quantize_chunk(rotated.contiguous(), group_size, codebook)
        return packed, s_rel, s_channel, codebook

    parts = [[], [], []]
    for r0 in range(0, n, block):
        for dst, t in zip(parts, _quantize_chunk(rotated[r0:r0 + block].contiguous(),
                                                 group_size, codebook)):
            dst.append(t)
    return (torch.cat(parts[0], 0), torch.cat(parts[1], 0), torch.cat(parts[2], 0), codebook)


def dequantize_rotated(packed, s_rel, s_channel, codebook, group_size):
    """Decode W4A8 storage back to its rotated-basis float weight."""
    n, k_half = packed.shape
    k = k_half * 2
    groups = k // group_size
    raw = packed.to(torch.int32) & 0xFF
    codes = torch.empty(n, k, dtype=torch.int32, device=packed.device)
    codes[:, 0::2] = raw & 0xF
    codes[:, 1::2] = (raw >> 4) & 0xF
    values = codebook.float()[codes].view(n, groups, group_size) * s_rel.float().unsqueeze(-1)
    int8_weight = values.view(n, k).round().clamp_(-127, 127)
    return (int8_weight.view(n, groups, group_size) * s_channel.float().view(n, 1, 1)).view(n, k)


def validate_shape(n, k, group_size, convrot_groupsize):
    if (
        k % 16 != 0
        or k % group_size != 0
        or k % convrot_groupsize != 0
        or group_size < 4
        or (16 % group_size != 0 and group_size % 16 != 0)
    ):
        raise ValueError(
            f"K={k} must be divisible by 16, group_size={group_size} and "
            f"convrot_groupsize={convrot_groupsize}"
        )


def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    metadata = header.pop("__metadata__", {}) or {}
    return header, metadata


def dst_layer_conf(group_size, convrot_groupsize):
    return {"format": DST_FORMAT, "group_size": group_size, "convrot": True,
            "convrot_groupsize": convrot_groupsize}


def source_layer_configs(header, metadata, path):
    """Return {layer_name: conf} plus which carrier the source uses.

    Two carriers exist in the wild: a `_quantization_metadata` JSON blob in the safetensors
    header, and one `<layer>.comfy_quant` uint8 tensor per layer.
    """
    if "_quantization_metadata" in metadata:
        blob = json.loads(metadata["_quantization_metadata"])
        return dict(blob.get("layers", {})), "metadata"

    names = [k[: -len(".comfy_quant")] for k in header if k.endswith(".comfy_quant")]
    if not names:
        raise SystemExit(f"{path}: no quantization metadata found (not a quantized checkpoint?)")
    confs = {}
    with safe_open(path, framework="pt") as f:
        for name in names:
            confs[name] = json.loads(bytes(f.get_tensor(name + ".comfy_quant").numpy().tolist()))
    return confs, "comfy_quant"


def to_bytes(t):
    return t.contiguous().cpu().view(torch.uint8).numpy().tobytes()


def plan(src, group_size, keep_int8):
    header, metadata = read_header(src)
    confs, carrier = source_layer_configs(header, metadata, src)

    convert, skip = [], []
    for name, conf in sorted(confs.items()):
        wkey, skey = name + ".weight", name + ".weight_scale"
        reason = None
        if conf.get("format") != SRC_FORMAT:
            reason = f"format={conf.get('format')}"
        elif not conf.get("convrot", False):
            reason = "not convrot"
        elif wkey not in header or header[wkey]["dtype"] != "I8" or len(header[wkey]["shape"]) != 2:
            reason = "weight is not 2D int8"
        elif skey not in header:
            reason = "missing weight_scale"
        elif _EMBEDDING_NAME.search(name):
            reason = "embedding table"
        elif keep_int8 is not None and keep_int8.search(name):
            reason = "matched --keep-int8"
        else:
            n, k = header[wkey]["shape"]
            try:
                validate_shape(n, k, group_size, int(conf.get("convrot_groupsize", 256)))
            except ValueError as exc:
                reason = str(exc)
        (skip.append((name, reason)) if reason else convert.append(name))

    return header, metadata, confs, carrier, convert, skip


def build_items(header, confs, carrier, convert, group_size):
    """Ordered output plan. Each item is ('copy', name) or ('layer', name, pieces, g)."""
    items = []
    converting = set(convert)
    emitted = set()

    for name in convert:
        n, k = header[name + ".weight"]["shape"]
        g = int(confs[name].get("convrot_groupsize", 256))
        pieces = [
            (name + ".weight", "I8", [n, k // 2]),
            (name + ".weight_s_rel", "F8_E4M3", [n, k // group_size]),
            (name + ".weight_s_channel", "F32", [n]),
            (name + ".weight_codebook", "F32", [16]),
        ]
        if carrier == "comfy_quant":
            blob = json.dumps(dst_layer_conf(group_size, g)).encode("utf-8")
            pieces.append((name + ".comfy_quant", "U8", [len(blob)]))
        items.append(("layer", name, pieces, g))
        emitted.update(p[0] for p in pieces)

    for key in header:
        base = key.rsplit(".", 1)[0]
        if base in converting and key in emitted:
            continue
        if base in converting and key.split(".")[-1] in ("weight", "weight_scale", "comfy_quant"):
            continue
        items.append(("copy", key))
    return items


def numel(shape):
    total = 1
    for s in shape:
        total *= s
    return total


def convert_layer(f, name, conf, group_size, device):
    q = f.get_tensor(name + ".weight").to(device)
    s = f.get_tensor(name + ".weight_scale").to(device)
    g = int(conf.get("convrot_groupsize", 256))
    # The int8 payload is the rotated weight; fp32 keeps the re-quantization exact.
    rotated = q.float() * s.float().view(-1, 1)
    del q, s
    packed, s_rel, s_channel, codebook = quantize_rotated(rotated, group_size)
    return rotated, packed, s_rel, s_channel, codebook, g


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst")
    ap.add_argument("--group-size", type=int, default=16)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--keep-int8", default=None,
                    help="regex of layer names to leave as int8 (produces a mixed checkpoint)")
    ap.add_argument("--verify-every", type=int, default=40,
                    help="report rotated-basis relative L2 error every N converted layers")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--verify-only", action="store_true",
                    help="quantize a few layers and report error, write nothing")
    args = ap.parse_args()

    if not args.dry_run and not args.verify_only and not args.dst:
        ap.error("--dst is required unless --dry-run or --verify-only")

    keep = re.compile(args.keep_int8) if args.keep_int8 else None
    header, metadata, confs, carrier, convert, skip = plan(args.src, args.group_size, keep)

    print(f"source        : {args.src}")
    print(f"carrier       : {carrier}")
    print(f"quant layers  : {len(confs)}  ->  convert {len(convert)}, keep {len(skip)}")
    for name, reason in skip[:10]:
        print(f"  keep as-is: {name}  ({reason})")
    if len(skip) > 10:
        print(f"  ... and {len(skip) - 10} more")

    items = build_items(header, confs, carrier, convert, args.group_size)

    out_header, offset = {}, 0
    for item in items:
        pieces = item[2] if item[0] == "layer" else [
            (item[1], header[item[1]]["dtype"], header[item[1]]["shape"])
        ]
        for key, dtype, shape in pieces:
            size = numel(shape) * _ST_ITEMSIZE[dtype]
            out_header[key] = {"dtype": dtype, "shape": list(shape),
                               "data_offsets": [offset, offset + size]}
            offset += size

    src_size = os.path.getsize(args.src)
    print(f"output tensors: {len(out_header)}")
    print(f"size          : {src_size / 2**30:.2f} GiB  ->  ~{offset / 2**30:.2f} GiB")
    if args.dry_run:
        return 0

    out_meta = dict(metadata)
    if carrier == "metadata":
        blob = json.loads(metadata["_quantization_metadata"])
        for name in convert:
            blob["layers"][name] = dst_layer_conf(
                args.group_size, int(confs[name].get("convrot_groupsize", 256))
            )
        out_meta["_quantization_metadata"] = json.dumps(blob)

    header_bytes = json.dumps({**out_header, "__metadata__": out_meta}).encode("utf-8")
    header_bytes += b" " * ((8 - len(header_bytes) % 8) % 8)

    errors, done, started = [], 0, time.time()
    dst = None if args.verify_only else open(args.dst, "wb")
    try:
        if dst is not None:
            dst.write(struct.pack("<Q", len(header_bytes)))
            dst.write(header_bytes)
        with open(args.src, "rb") as raw, safe_open(args.src, framework="pt") as f:
            raw.seek(0)
            src_data_start = 8 + struct.unpack("<Q", raw.read(8))[0]
            for item in items:
                if item[0] == "copy":
                    if dst is None:
                        continue
                    begin, end = header[item[1]]["data_offsets"]
                    raw.seek(src_data_start + begin)
                    left = end - begin
                    while left:
                        chunk = raw.read(min(left, 1 << 24))
                        if not chunk:
                            raise SystemExit(f"short read on {item[1]}")
                        dst.write(chunk)
                        left -= len(chunk)
                    continue

                _, name, pieces, _g = item
                rotated, packed, s_rel, s_channel, codebook, g = convert_layer(
                    f, name, confs[name], args.group_size, args.device
                )
                if args.verify_every and done % args.verify_every == 0:
                    deq = dequantize_rotated(packed, s_rel, s_channel, codebook, args.group_size)
                    rel = ((deq - rotated).norm() / rotated.norm()).item()
                    errors.append((name, rel))
                    print(f"  [{done + 1}/{len(convert)}] {name}: relL2={rel:.5f}")
                del rotated
                if dst is not None:
                    out = {"weight": packed, "weight_s_rel": s_rel,
                           "weight_s_channel": s_channel.float(),
                           "weight_codebook": codebook.float()}
                    for key, _dt, _sh in pieces:
                        suffix = key[len(name) + 1:]
                        if suffix == "comfy_quant":
                            dst.write(json.dumps(dst_layer_conf(args.group_size, g)).encode("utf-8"))
                        else:
                            dst.write(to_bytes(out[suffix]))
                done += 1
                if args.verify_only and done >= 3:
                    break
                if done % 25 == 0:
                    rate = done / (time.time() - started)
                    print(f"  {done}/{len(convert)} layers  ({rate:.1f}/s, "
                          f"eta {(len(convert) - done) / rate / 60:.1f} min)")
    finally:
        if dst is not None:
            dst.close()

    if errors:
        worst = max(e for _, e in errors)
        print(f"\nrotated-basis relL2 vs the int8 source: "
              f"mean {sum(e for _, e in errors) / len(errors):.5f}, worst {worst:.5f} "
              f"({len(errors)} layers sampled)")
    if not args.verify_only:
        print(f"\nwrote {args.dst}  ({os.path.getsize(args.dst) / 2**30:.2f} GiB, "
              f"{(time.time() - started) / 60:.1f} min)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
