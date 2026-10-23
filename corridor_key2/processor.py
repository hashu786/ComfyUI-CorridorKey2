"""In-memory CorridorKey 2.0 inference on ComfyUI IMAGE / MASK tensors.

This mirrors upstream `corridorkey.runtime.run_inference` step for step and
calls the same upstream functions (RVM prepass, Hann stage-1 streaming,
low-VRAM staging, stage 2). The only difference is where frames come from and
go to: upstream reads video files or frame folders and writes videos or EXRs;
here frames are read from tensors and written back into tensors.

Upstream reaches its readers through module-level helpers called with a path
string. Tensor clips are registered under a `ck2tensor://` key and those helpers
are wrapped to recognise the key, so every other code path stays upstream's.
"""

from __future__ import annotations

import gc
import traceback
import uuid
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F

from ..corridorkey import color_utils as cu
from ..corridorkey import runtime as ck_runtime
from ..corridorkey import video_io as ck_io
from . import engine

TENSOR_PREFIX = "ck2tensor://"


# ---------------------------------------------------------------------------
# Registered tensor clips
# ---------------------------------------------------------------------------


@dataclass
class TensorClip:
    images: torch.Tensor  # (T, H, W, 3) float32, CPU
    linear_input: bool
    masks: torch.Tensor | None = None  # (T, H, W) float32, CPU, full hint video
    on_frame_read: Callable[[], None] | None = None


_CLIPS: dict[str, TensorClip] = {}


def _clip(key) -> TensorClip | None:
    if isinstance(key, str) and key.startswith(TENSOR_PREFIX):
        return _CLIPS.get(key)
    return None


def _frame_views(clip: TensorClip, t: int, meta: dict, *, need_model: bool):
    """Same as upstream preprocess_frame_dir_path, but from a tensor frame."""
    raw = clip.images[t].numpy().astype(np.float32, copy=False)
    # upstream video_io._read_frame_dir_views
    if clip.linear_input:
        linear = raw
        display = np.clip(cu.linear_to_srgb(np.maximum(linear, 0.0)), 0.0, 1.0)
    else:
        display = np.clip(raw, 0.0, 1.0)
        linear = cu.srgb_to_linear(display)
    # upstream video_io.preprocess_frame_dir_path
    h_out, w_out = meta["shape_hw"]
    h_model, w_model = meta["model_native_hw"]
    display = ck_io.resize_hwc_lanczos(display, (h_model, w_model))
    linear = ck_io.resize_hwc_lanczos(linear, (h_model, w_model))
    display, _ = ck_io.center_pad_to_shape(np.clip(display, 0.0, 1.0), h_out, w_out)
    display_t = torch.from_numpy(
        np.ascontiguousarray(display.transpose(2, 0, 1))).to(dtype=torch.bfloat16)
    if not need_model:
        return None, display_t, display_t
    linear, _ = ck_io.center_pad_to_shape(linear, h_out, w_out)
    model = cu.linear_to_asinh(linear)
    model_t = torch.from_numpy(
        np.ascontiguousarray(model.transpose(2, 0, 1))).to(dtype=torch.bfloat16)
    return model_t, display_t, display_t


def _hint_frame(mask_hw: torch.Tensor, meta: dict) -> torch.Tensor:
    """Same as upstream preprocess_hint_frame, for a float mask in [0, 1]."""
    h_out, w_out = meta["shape_hw"]
    hint = mask_hw.numpy().astype(np.float32, copy=False)
    hint, _ = ck_io.center_pad_to_shape(np.clip(hint, 0.0, 1.0), h_out, w_out)
    return torch.from_numpy(np.ascontiguousarray(hint[None])).to(dtype=torch.bfloat16)


class TensorWindowReader:
    """Drop-in for upstream SequentialFrameDirWindowReader."""

    def __init__(self, source: str, meta: dict):
        self.clip = _clip(source)
        if self.clip is None:
            raise RuntimeError(f"unknown tensor clip {source!r}")
        self.meta = meta
        self.next_t = 0
        self.cache: dict[int, tuple] = {}

    def close(self):
        pass

    def _read_until(self, t_end: int):
        while self.next_t < t_end:
            if self.clip.on_frame_read is not None:
                self.clip.on_frame_read()
            self.cache[self.next_t] = _frame_views(self.clip, self.next_t, self.meta,
                                                   need_model=True)
            self.next_t += 1

    def window(self, t0: int, t1: int):
        self._read_until(t1)
        model = torch.stack([self.cache[t][0] for t in range(t0, t1)])
        foundation = torch.stack([self.cache[t][1] for t in range(t0, t1)])
        review = torch.stack([self.cache[t][2] for t in range(t0, t1)])
        for old_t in [k for k in self.cache if k < t0]:
            del self.cache[old_t]
        return model, foundation, review


class TensorHintReader:
    """Drop-in for upstream SequentialHintVideoWindowReader."""

    def __init__(self, clip: TensorClip, meta: dict):
        self.clip = clip
        self.meta = meta

    def close(self):
        pass

    def window(self, t0: int, t1: int):
        return torch.stack([_hint_frame(self.clip.masks[t], self.meta) for t in range(t0, t1)])


def _iter_tensor_chunks(clip: TensorClip, meta: dict, *, chunk_size: int, need_model: bool):
    T = int(meta["num_frames"])
    t = 0
    while t < T:
        t0 = t
        model_list = [] if need_model else None
        foundation_list = []
        while t < T and len(foundation_list) < chunk_size:
            if clip.on_frame_read is not None:
                clip.on_frame_read()
            model, foundation, _ = _frame_views(clip, t, meta, need_model=need_model)
            if need_model:
                model_list.append(model)
            foundation_list.append(foundation)
            t += 1
        yield t0, (torch.stack(model_list) if need_model else None), torch.stack(foundation_list)


def _install_hooks() -> None:
    if getattr(ck_runtime, "_ck2_hooks", False):
        return
    upstream_iter = ck_runtime.iter_frame_dir_chunks
    upstream_hint_reader = ck_runtime.make_hint_reader

    def iter_frame_dir_chunks(frame_dir, meta, *, chunk_size, need_model):
        clip = _clip(frame_dir)
        if clip is not None:
            return _iter_tensor_chunks(clip, meta, chunk_size=chunk_size, need_model=need_model)
        return upstream_iter(frame_dir, meta, chunk_size=chunk_size, need_model=need_model)

    def make_hint_reader(path, meta):
        clip = _clip(path)
        if clip is not None:
            return TensorHintReader(clip, meta) if clip.masks is not None else None
        return upstream_hint_reader(path, meta)

    ck_runtime.iter_frame_dir_chunks = iter_frame_dir_chunks
    ck_runtime.make_hint_reader = make_hint_reader
    ck_runtime._ck2_hooks = True


_install_hooks()


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

WINDOW_CHOICES = ["auto", "80/40", "48/24", "24/12", "8/4", "4/2", "2/1", "1/1"]


@dataclass
class KeySettings:
    gamma_space: str = "sRGB"
    hint_quality: float = 0.95
    carry_hint: bool = True
    carry_hint_quality: float = 0.95
    despill_strength: float = 0.5
    window: str = "auto"
    low_vram: bool = False
    unload_other_models: bool = True


def build_config(window: str, low_vram: bool, **kwargs) -> ck_runtime.InferenceConfig:
    """Mirror infer.py: an explicit window or low_vram turns automatic selection off."""
    if window not in WINDOW_CHOICES:
        raise ValueError(f"window must be one of {WINDOW_CHOICES}, got {window!r}")
    if window == "auto":
        chunk, stride = 80, 40
        auto = not low_vram
    else:
        chunk, stride = (int(v) for v in window.split("/"))
        auto = False
    return ck_runtime.InferenceConfig(
        hann_chunk=chunk, hann_stride=stride, auto_vram=auto, low_vram=bool(low_vram),
        temp_dir=engine.comfy_temp_dir(), **kwargs)


# ---------------------------------------------------------------------------
# Rendering stage-2 logits to ComfyUI tensors
# ---------------------------------------------------------------------------


OUTPUT_NAMES = ("fg", "matte", "processed", "QC", "cutout")


def _render(meta: dict, logits: torch.Tensor, checker_lin: torch.Tensor, *,
            linear_out: bool, despill: float, needed: set[str]):
    """Returns a dict of NCHW float tensors for the `needed` outputs.

    Colour handling follows upstream's renderers. In sRGB mode the outputs
    match upstream's video path: `fg` = video `fg`, `cutout` = video `cutout`
    (despilled straight sRGB + alpha), `QC` = video `checker`. In Linear mode
    they match the EXR path: unclamped scene-linear, despill in linear light.
    `processed` is the cutout colour premultiplied by alpha (done in linear
    light, then encoded to sRGB in sRGB mode).
    """
    out: dict[str, torch.Tensor] = {}
    alpha = ck_runtime._crop_native_torch(logits[:, 0:1].float(), meta).clamp(0.0, 1.0)
    if "matte" in needed:
        out["matte"] = alpha
    if not needed & {"fg", "processed", "QC", "cutout"}:
        return out
    fg_model = ck_runtime._crop_native_torch(logits[:, 1:4].float(), meta)
    pf_linear = ck_runtime._linear_adjust_inverse_torch(cu.asinh_to_linear(fg_model), meta)
    pf_srgb = cu.linear_to_srgb(pf_linear.clamp(min=0.0)).clamp(0.0, 1.0)
    if "fg" in needed:
        out["fg"] = pf_linear if linear_out else pf_srgb

    pf_despill_srgb = None
    if needed & {"QC"} or (not linear_out and needed & {"processed", "cutout"}):
        pf_despill_srgb = ck_runtime._despill_average_srgb_torch(pf_srgb, strength=despill)
    if needed & {"processed", "cutout"}:
        if linear_out:
            color = ck_runtime._despill_average_srgb_torch(pf_linear, strength=despill)
            if "processed" in needed:
                out["processed"] = color * alpha
        else:
            color = pf_despill_srgb.clamp(0.0, 1.0)  # upstream clamps its sRGB cutout
            if "processed" in needed:
                out["processed"] = cu.linear_to_srgb(
                    (cu.srgb_to_linear(color) * alpha).clamp(min=0.0)).clamp(0.0, 1.0)
        if "cutout" in needed:
            out["cutout"] = torch.cat([color, alpha], dim=1)
    if "QC" in needed:
        comp_lin = cu.srgb_to_linear(pf_despill_srgb) * alpha + checker_lin * (1.0 - alpha)
        out["QC"] = cu.linear_to_srgb(comp_lin.clamp(min=0.0)).clamp(0.0, 1.0)
    return out


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def _prepare_masks(mask, T: int, h: int, w: int) -> torch.Tensor | None:
    if mask is None:
        return None
    m = mask.detach().float().cpu()
    if m.ndim == 2:
        m = m[None]
    if m.ndim == 4 and m.shape[-1] == 1:
        m = m[..., 0]
    if m.ndim != 3:
        raise ValueError(f"mask must be (B, H, W), got shape {tuple(mask.shape)}")
    if m.shape[0] not in (1, T):
        raise ValueError(
            f"mask batch has {m.shape[0]} frames but the image batch has {T}. "
            "Provide 1 mask (first-frame hint) or one mask per frame (full hint).")
    if tuple(m.shape[-2:]) != (h, w):
        engine.log(f"resizing hint mask {tuple(m.shape[-2:])} -> {(h, w)}")
        m = F.interpolate(m[:, None], size=(h, w), mode="bilinear", align_corners=False)[:, 0]
    return m.clamp(0.0, 1.0).contiguous()


def key_tensors(image: torch.Tensor, mask: torch.Tensor | None, settings: KeySettings,
                progress: Callable[[str, int, int], None] | None = None,
                needed: set[str] | None = None):
    """Run CorridorKey 2.0 on an IMAGE batch.

    Returns (fg, matte, processed, QC, cutout). Only outputs in `needed` are
    computed and kept (all when None); the rest are 1-pixel placeholders, so
    unconnected outputs cost no RAM.
    """
    needed = set(OUTPUT_NAMES) if needed is None else set(needed) & set(OUTPUT_NAMES)
    if image.ndim == 3:
        image = image[None]
    images = image.detach().float().cpu()[..., :3].contiguous()
    T, h, w, _ = images.shape
    masks = _prepare_masks(mask, T, h, w)
    linear_in = settings.gamma_space == "Linear"

    try:
        import comfy.model_management as mm

        check_interrupt = mm.throw_exception_if_processing_interrupted
    except Exception:  # pragma: no cover - outside ComfyUI
        def check_interrupt():
            return None

    def report(message: str, done: int):
        if progress is not None:
            progress(message, done, T)

    if settings.unload_other_models:
        engine.free_comfy_vram()
    report("loading model", 0)
    model = engine.get_model(status=lambda message: report(message, 0))

    cfg = build_config(
        settings.window, settings.low_vram,
        carry_hint=bool(settings.carry_hint),
        hint_quality=float(settings.hint_quality),
        carry_hint_quality=float(settings.carry_hint_quality),
        despill_strength=float(settings.despill_strength),
        frame_dir_linear=linear_in,
    )

    # Same meta as upstream probe_frame_dir_meta (native resolution, quantum 128).
    q = cfg.shape_quantum
    h_pad, w_pad = ck_io.ceil_to_multiple(h, q), ck_io.ceil_to_multiple(w, q)
    pad_l, pad_t = (w_pad - w) // 2, (h_pad - h) // 2
    meta = {
        "native_fps": 24.0,
        "native_hw": (h, w),
        "output_hw": (h, w),
        "model_native_hw": (h, w),
        "model_scale": 1.0,
        "source_hw": (h, w),
        "shape_hw": (h_pad, w_pad),
        "start_frame": 0,
        "num_frames": T,
        "pad": (pad_l, w_pad - w - pad_l, pad_t, h_pad - h - pad_t),
        "native_resolution": True,
        "exr_linear": linear_in,
        "linear_brightness": 1.0,
        "linear_contrast": 1.0,
        "linear_contrast_pivot": 0.18,
        "despill_strength": float(settings.despill_strength),
    }
    H, W = meta["shape_hw"]

    full_hint = masks is not None and (masks.shape[0] == T)
    first_hint = masks is not None and not full_hint
    initial_hint = _hint_frame(masks[0], meta) if first_hint else None
    mode = "full hint" if full_hint else ("first-frame hint" if first_hint else "no hint")

    if cfg.auto_vram:
        ck_runtime.apply_auto_vram(cfg, (H, W), T)
    engine.log(f"{T} frame(s) @ {w}x{h} ({mode}, {settings.gamma_space}); "
               f"window {cfg.hann_chunk}/{cfg.hann_stride}, low_vram={cfg.low_vram}")

    key = f"{TENSOR_PREFIX}{uuid.uuid4().hex}"
    _CLIPS[key] = TensorClip(images=images, linear_input=linear_in,
                             masks=masks if full_hint else None,
                             on_frame_read=check_interrupt)
    tag = key[len(TENSOR_PREFIX):][:12]
    rvm_path = cfg.temp_dir / f"{tag}_rvm_feat.f16"
    baton_path = cfg.temp_dir / f"{tag}_baton.f16"
    foundation_prefix = cfg.temp_dir / f"{tag}_foundation"
    temp_files = [rvm_path, baton_path,
                  cfg.temp_dir / f"{tag}_foundation_cradio.f32",
                  cfg.temp_dir / f"{tag}_foundation_moge.f32"]

    channels = {"fg": 3, "matte": 0, "processed": 3, "QC": 3, "cutout": 4}
    outs: dict[str, torch.Tensor] = {}
    for name in OUTPUT_NAMES:
        c = channels[name]
        if name in needed:
            shape = (T, h, w, c) if c else (T, h, w)
        else:
            shape = (1, 1, 1, c) if c else (1, 1, 1)
        outs[name] = (torch.empty if name in needed else torch.zeros)(shape, dtype=torch.float32)
    est_gb = sum(T * h * w * max(channels[n], 1) * 4 for n in needed) / 2**30
    engine.log(f"outputs: {', '.join(n for n in OUTPUT_NAMES if n in needed) or 'none'} "
               f"(~{est_gb:.1f} GB RAM)")
    written = [0]

    try:
        with torch.no_grad():
            checker = torch.from_numpy(ck_io.make_checker(h, w)).unsqueeze(0)
            checker_lin = cu.srgb_to_linear(checker).cuda()

            ck_runtime._phase1_setup(model, stage1_on_gpu=not cfg.low_vram)
            engine.set_low_vram(model, cfg)

            if model.foundations.rvm is not None:
                report("RVM prepass", 0)
                ck_runtime.precompute_rvm_to_memmap(
                    model, key, meta, rvm_path,
                    chunk=max(1, min(int(cfg.hann_chunk), 8)), reader_kind="frame_dir")
                rvm_feat_path = rvm_path
            else:
                rvm_feat_path = None

            foundation_paths = None
            if cfg.low_vram:
                report("foundation prepass", 0)
                foundation_paths = ck_runtime.precompute_foundations_to_memmap(
                    model, key, meta, foundation_prefix,
                    chunk=max(1, int(cfg.low_vram_foundation_chunk)), reader_kind="frame_dir")
                model.foundations.cpu()
                model.stage2.cpu()
                model.stage1.cuda()
                torch.cuda.empty_cache()
            else:
                model.stage2.cuda()
                model.stage2.eval()
                torch.cuda.empty_cache()

            stage2_batch = max(1, int(cfg.stage2_batch))

            def write_prediction_batches(items):
                check_interrupt()
                device = next(model.stage2.parameters()).device
                for offset in range(0, len(items), stage2_batch):
                    sub = items[offset:offset + stage2_batch]
                    rgb_t = torch.stack([item[2] for item in sub], dim=0)
                    rgb_t = rgb_t.unsqueeze(0).to(device, non_blocking=True)
                    baton_np = np.stack([item[1] for item in sub], axis=0)
                    baton_t = torch.from_numpy(baton_np).unsqueeze(0)
                    baton_t = baton_t.to(device, non_blocking=True).float()
                    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                        logits = model.stage2(rgb_t, baton_t)
                        if isinstance(logits, tuple):
                            logits = logits[0]
                    rendered = _render(
                        meta, logits[0], checker_lin, linear_out=linear_in,
                        despill=float(settings.despill_strength), needed=needed)
                    idx = torch.tensor([item[0] for item in sub], dtype=torch.long)
                    for name, tensor in rendered.items():
                        if name == "matte":
                            outs[name][idx] = tensor[:, 0].float().cpu()
                        else:
                            outs[name][idx] = tensor.permute(0, 2, 3, 1).float().cpu()
                    written[0] += len(sub)
                    del logits, baton_t, rgb_t, baton_np, rendered
                report(f"keyed {written[0]}/{T}", written[0])

            ck_runtime.hann_stage1_stream(
                model, key, meta, reader_cls=TensorWindowReader,
                reader_kind="frame_dir", chunk=cfg.hann_chunk, stride=cfg.hann_stride,
                baton_path=baton_path, initial_hint=initial_hint,
                initial_hint_quality=cfg.hint_quality,
                full_hint_video_path=key if full_hint else None,
                full_hint_quality=cfg.hint_quality,
                rvm_path=rvm_feat_path, foundation_paths=foundation_paths,
                carry_hint_from_aux=cfg.carry_hint,
                carry_hint_quality=cfg.carry_hint_quality,
                on_baton_frames=None if cfg.low_vram else write_prediction_batches)

            if cfg.low_vram:
                ck_runtime._phase2_setup(model)
                baton = np.memmap(baton_path, dtype=np.float16, mode="r",
                                  shape=(T, model.stage1.d_16, H // 16, W // 16))
                reader = TensorWindowReader(key, meta)
                try:
                    for t0 in range(0, T, stage2_batch):
                        t1 = min(T, t0 + stage2_batch)
                        model_chunk, _, _ = reader.window(t0, t1)
                        items = [(t, np.asarray(baton[t]).copy(), model_chunk[t - t0])
                                 for t in range(t0, t1)]
                        write_prediction_batches(items)
                        del model_chunk, items
                finally:
                    del baton
            torch.cuda.synchronize()
    except BaseException as exc:
        # Drop the failed frames' locals (open memmaps, GPU tensors) so the temp
        # files can be deleted on Windows and VRAM is freed. The traceback text
        # is unaffected.
        traceback.clear_frames(exc.__traceback__)
        raise
    finally:
        _CLIPS.pop(key, None)
        gc.collect()
        for p in temp_files:
            engine.safe_remove(p)
        engine.offload(model)

    if written[0] != T:
        raise RuntimeError(f"CorridorKey produced {written[0]} of {T} frames")
    report("done", T)
    return tuple(outs[name] for name in OUTPUT_NAMES)


# ---------------------------------------------------------------------------
# Render-to-disk (upstream run_inference, unchanged)
# ---------------------------------------------------------------------------


def find_ffmpeg() -> str:
    import shutil

    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:
        raise FileNotFoundError(
            "ffmpeg not found on PATH and imageio-ffmpeg is unavailable; "
            "use output_format=exr or install ffmpeg.") from exc


def render_to_disk(input_path: str, output_dir: str, *, outputs: set[str],
                   output_format: str, exr_dtype: str, input_is_linear: bool,
                   frame_dir_fps: float, hint_first_frame: str | None,
                   hint_video: str | None, num_frames: int, cutout_linear: bool,
                   settings: KeySettings) -> None:
    if settings.unload_other_models:
        engine.free_comfy_vram()
    model = engine.get_model()
    cfg = build_config(
        settings.window, settings.low_vram,
        carry_hint=bool(settings.carry_hint),
        hint_quality=float(settings.hint_quality),
        carry_hint_quality=float(settings.carry_hint_quality),
        despill_strength=float(settings.despill_strength),
        frame_dir_linear=bool(input_is_linear),
        frame_dir_fps=float(frame_dir_fps),
        num_frames=int(num_frames),
        outputs=set(outputs),
        output_format=output_format,
        exr_dtype=exr_dtype,
        cutout_linear=bool(cutout_linear),
    )
    writer_cfg = ck_io.WriterConfig(
        ffmpeg_bin=find_ffmpeg() if output_format == "video" else "ffmpeg")
    try:
        engine.set_low_vram(model, cfg)
        ck_runtime.run_inference(
            model, input_path, output_dir, cfg, writer_cfg,
            initial_hint_path=hint_first_frame or None,
            hint_video_path=hint_video or None)
    except BaseException as exc:
        traceback.clear_frames(exc.__traceback__)
        raise
    finally:
        gc.collect()
        engine.offload(model)
