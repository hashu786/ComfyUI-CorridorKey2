# ComfyUI-CorridorKey2

ComfyUI nodes for [CorridorKey 2.0](https://github.com/nikopueringer/CorridorKey2.0), Corridor Digital's neural green/blue screen keyer. They're laid out like [SeanBRVFX/ComfyUI-CorridorKey](https://github.com/SeanBRVFX/ComfyUI-CorridorKey) for 1.0: an IMAGE plus a MASK hint in, and `fg / matte / processed / QC` out (plus an RGBA `cutout`). They run the 2.0 model and its temporal pipeline unchanged.

## Nodes

### CorridorKey 2.0

Keys an IMAGE batch as one clip. Frames are processed in order through overlapping temporal Hann windows, at native resolution.

| Input | Notes |
|---|---|
| `image` | Green/blue screen frames (the whole batch = one clip). |
| `mask` (optional) | Hint, white = subject. **None**: the model chooses the subject. **1 mask**: first-frame hint, carried through the shot. **One mask per frame**: full hint video (SAM / BiRefNet / roto / garbage matte). Other counts are an error, not silently reused. |
| `gamma_space` | `sRGB` for normal video/PNG. `Linear` for scene-linear plates (keeps values > 1.0). |
| `hint_quality` | Trust in the hint's edges. 0.95 clean matte, 0.75 rough automated mask, ≤ 0.5 loose, dilated or scribbled mask. Too high with a puffy mask leaves a soft halo the shape of the mask. |
| `carry_hint` / `carry_hint_quality` | Feed the model's own matte back across overlapping windows (upstream default on). Turn off to follow a full hint video literally. |
| `despill_strength` | 0–1, default 0.5 (upstream default). Applies to `processed`, `QC` and `cutout`; `fg` is never despilled. |
| `window` | Stage-1 Hann window `chunk/stride`. `auto` picks the largest that fits free VRAM (e.g. 80/40 at 1080p on a 32 GB card). |
| `low_vram` | Exact staged inference: lower peak VRAM, slower, uses temp disk. |
| `unload_other_models` | Unload other ComfyUI models first so auto selection sees the whole GPU. |

| Output | Contents |
|---|---|
| `fg` | Straight foreground, no despill. |
| `matte` | Alpha, MASK. |
| `processed` | Despilled foreground premultiplied by the matte (subject over black). Premultiplied in linear light. |
| `QC` | Despilled composite over a checkerboard, always sRGB. |
| `cutout` | RGBA: despilled straight colour + alpha (upstream's cutout pass). Save Image writes a transparent PNG. |

`fg`, `processed` and `cutout` are in the same colour space as the input: sRGB in sRGB mode, unclamped scene-linear in Linear mode (for EXR / Nuke).

Only connected outputs are computed and held in RAM, so leave unused outputs unconnected on long clips.

### CorridorKey 2.0 (Render to Disk)

Calls upstream `run_inference` directly on a video file or frame folder and writes EXR sequences (unclamped linear, half or float) or videos (MP4, ProRes 4444 cutout). Frames stream from disk, so this is the one to use for long 4K plates: a 200-frame 4K clip is about 20 GB as a ComfyUI IMAGE batch before the model touches it. It also supports upstream's `input_is_linear` for linear EXR plates, and hint paths. Video output uses ffmpeg on PATH, or the one that ships with ComfyUI's imageio-ffmpeg.

## Install

1. Clone into `ComfyUI/custom_nodes/`:
   ```
   cd ComfyUI/custom_nodes
   git clone https://github.com/hashu786/ComfyUI-CorridorKey2
   ```
   or download the ZIP from GitHub and extract it there as `ComfyUI-CorridorKey2`.
   Dependencies are normally already in ComfyUI. If not, install `requirements.txt` with ComfyUI's Python
   (`python_embeded\python.exe -m pip install -r ...` on Windows portable).
2. Restart ComfyUI. The nodes are under **CorridorKey**. Try `example_workflows/corridorkey2_basic.json`
   (built-in nodes only).

Requires an NVIDIA GPU (RTX 30-series or newer; the model runs in bf16). A 1080p clip fits in about 6 GB of VRAM;
more VRAM buys longer temporal windows.

### Models (downloaded automatically)

The first time a CorridorKey node runs it downloads about **5 GB** into `ComfyUI/models/corridorkey2/`, then
carries on. That first run takes a few minutes; progress is shown in the ComfyUI console. Later runs start
immediately. The files come straight from their publishers:

```
ComfyUI/models/corridorkey2/
  weights/corridorkey_v2.safetensors       CorridorKey 2.0 (Corridor Digital, Hugging Face)
  weights/foundation_stats.safetensors
  vfms/cradio/C-RADIOv4-SO400M/            NVIDIA C-RADIOv4-SO400M (Hugging Face)
  vfms/moge/model.pt                       Microsoft MoGe-2 ViT-B (Hugging Face)
  vfms/rvm/pretrain_rvm_small16_256_204031069.npz   Google DeepMind RVM (Google Cloud Storage)
```

- **Shared or custom location:** add `corridorkey2: path/to/folder` to `extra_model_paths.yaml`; the models
  are looked for (and downloaded) there.
- **Download ahead of time / offline machines:** run `python scripts/download_weights.py [--package_root DEST]`
  with ComfyUI's Python (`python_embeded\python.exe` on Windows portable), then copy the folder over.
  Set the environment variable `CORRIDORKEY2_NO_DOWNLOAD=1` to turn the automatic download off.

## Differences from upstream

The upstream `corridorkey/` package and `third_party/MoGe/` are vendored **unmodified** (commit `485f5e9`, 2026-10-15). `corridor_key2/` adapts them to ComfyUI:

- **Tensor I/O.** Frames come from IMAGE tensors through readers that are drop-in replacements for upstream's frame-folder readers. Preprocessing is bit-identical to upstream's frame-folder path. The RVM prepass, Hann stage-1 streaming, carry hints, low-VRAM staging and stage 2 are upstream's own functions.
- **No version pins that fight ComfyUI.** Upstream pins `numpy<2` and `transformers<5`; nothing here needs them:
  - C-RADIO is built directly and its safetensors loaded with `load_state_dict`, rather than relying on `from_pretrained` behaviour in transformers 5. Upstream's loader is the fallback.
  - `utils3d` (a git-only MoGe dependency) is stubbed when missing. MoGe imports it, but the encoder CorridorKey uses never calls it.
- **Process hygiene.** Upstream sets `HF_HUB_OFFLINE` for the whole process; it's restored after loading so other nodes can still download. The model is cached between runs and offloaded to CPU after each run. Low-VRAM chunking is reset between runs.

## Credits

- [CorridorKey 2.0](https://github.com/nikopueringer/CorridorKey2.0): model by Zachary Harris with Niko Pueringer and Corridor Digital.
- Node layout and progress reporting follow [SeanBRVFX/ComfyUI-CorridorKey](https://github.com/SeanBRVFX/ComfyUI-CorridorKey), the CorridorKey 1.0 node.
- Foundation models: NVIDIA C-RADIO, Microsoft MoGe, Google DeepMind RVM.

## License

CorridorKey is © Corridor Digital, licensed under the CorridorKey Licence (CC BY-NC-SA 4.0 with additional terms); see `LICENSE`. Commercial use of the *output* is allowed. Derivatives must stay free and keep the CorridorKey name. Organisations over US $20M annual revenue need a CorridorKey seat per user ([corridorkey.com](https://corridorkey.com)). This node pack is released under the same licence, as the CorridorKey Licence requires. It does not redistribute any model weights: they are downloaded from their publishers and stay under their own licences; see `THIRD_PARTY_NOTICES.md`.
