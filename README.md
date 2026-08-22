Converts a ComfyUI `int8_tensorwise` ConvRot safetensors checkpoint to `asym_w4a8_int8`, without needing the original fp16 weights. Shrinks the checkpoint by about 40% on average (30-43% measured across 10 real models).

Needs `torch` and `safetensors`.

### Why it works

Both formats rotate the weight with the same normalized regular Hadamard (ConvRot) before quantizing, and that matrix is symmetric and orthogonal (`H @ H == I`). So the int8 payload on
disk already is the rotated weight the W4A8 quantizer wants as its input. The conversion dequantizes the int8 rows, re-quantizes in the rotated basis, and never rotates at all.

### What it costs

The int8 hop is not free, but it is second order: the two quantization errors add in quadrature, and the int8 row-wise error sits far below the 4-bit error.

| weight | int8 only | fp16 -> w4a8 | int8 -> w4a8 |
| --- | --- | --- | --- |
| gaussian 4096x4096 | 0.00869 | 0.07311 | 0.07368 |

Relative L2. Quadrature predicts `sqrt(0.07308^2 + 0.00866^2) = 0.07359` against `0.07365` measured, so the penalty is about +0.8% on ConvRot-rotated (Gaussian) weights.

### Usage

```
python convert_int8_convrot_to_w4a8.py --src IN.safetensors --dst OUT.safetensors
```

- `--dry-run` prints the conversion plan and output size, writes nothing.
- `--verify-only` quantizes a few layers and reports the rotated-basis relative L2 error.
- `--keep-int8 REGEX` leaves matching layers as int8, producing a mixed checkpoint.
- `--device cpu` works if there is no GPU, just slower.
- `--group-size` defaults to 16, which is what ComfyUI's loader and the published
  `asym_w4a8_int8` checkpoints use.

Both metadata carriers are handled: a `_quantization_metadata` blob in the safetensors header, and per-layer `<layer>.comfy_quant` tensors. Everything that is not a converted layer is copied through byte for byte.

### Output

Per converted layer, matching the published `asym_w4a8_int8` checkpoints:

| tensor | dtype | shape |
| --- | --- | --- |
| `<layer>.weight` | int8 | `[N, K/2]` packed int4 codes |
| `<layer>.weight_s_rel` | float8_e4m3 | `[N, K/group_size]` |
| `<layer>.weight_s_channel` | float32 | `[N]` |
| `<layer>.weight_codebook` | float32 | `[16]` Lloyd-Max levels |

### Limitations

- Layers marked `convrot: false` are left as int8. Converting those would need the rotation this script deliberately avoids. ComfyUI skips ConvRot when `in_features % 256 != 0`, so
  mixed checkpoints exist.
- Embedding tables (`embed_tokens`, `token_embedding`, `position_embedding`, `word_embeddings`, `shared`, `wte`, `wpe`) are left as int8. ComfyUI's quantized `Embedding` only loads fp8 and `int8_tensorwise`, since those dequantize per looked-up row, so a W4A8 table fails with a shape mismatch.
- Symmetric codebook path only. ComfyUI's loader never reads a `correction` tensor for this format, so the asymmetric variant would not load anyway.