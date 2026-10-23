from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

from .corridor_key2.processor import (
    OUTPUT_NAMES, WINDOW_CHOICES, KeySettings, key_tensors, render_to_disk,
)


def _connected_outputs(prompt, unique_id) -> set[str] | None:
    """Names of this node's outputs that some other node consumes.

    Returns None (compute everything) if the prompt graph isn't available.
    """
    if not isinstance(prompt, dict) or unique_id is None:
        return None
    used: set[int] = set()
    for node in prompt.values():
        inputs = node.get("inputs", {}) if isinstance(node, dict) else {}
        for value in inputs.values():
            if (isinstance(value, (list, tuple)) and len(value) == 2
                    and str(value[0]) == str(unique_id) and isinstance(value[1], int)):
                used.add(value[1])
    return {OUTPUT_NAMES[i] for i in used if 0 <= i < len(OUTPUT_NAMES)}


def _build_progress_reporter(unique_id: str | None) -> Callable[[str, int, int], None]:
    progress_bar = None
    prompt_server = None
    try:
        from comfy.utils import ProgressBar

        progress_bar = ProgressBar(1, node_id=unique_id)
    except Exception:
        progress_bar = None
    if unique_id:
        try:
            from server import PromptServer

            prompt_server = PromptServer.instance
        except Exception:
            prompt_server = None

    def report(message: str, completed: int, total: int) -> None:
        bounded_total = max(int(total), 1)
        bounded_completed = max(0, min(int(completed), bounded_total))
        if progress_bar is not None:
            progress_bar.update_absolute(bounded_completed, total=bounded_total)
        if prompt_server is not None and unique_id:
            try:
                prompt_server.send_progress_text(message, unique_id)
            except Exception:
                pass
        print(f"[CorridorKey2] {message}")

    return report


def _shared_inputs() -> dict[str, tuple]:
    return {
        "hint_quality": ("FLOAT", {
            "default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01,
            "tooltip": "How much to trust the hint mask. 0.95 = clean matte or roto, "
                       "0.75 = rough automated mask (SAM, BiRefNet), 0.5 or less = loose "
                       "scribble or garbage matte. Ignored when no hint is connected.",
        }),
        "carry_hint": ("BOOLEAN", {
            "default": True,
            "tooltip": "Feed the model's own matte from overlapping Hann windows back in as "
                       "a hint (upstream default). Turn off to use a full hint video literally.",
        }),
        "carry_hint_quality": ("FLOAT", {
            "default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01,
            "tooltip": "Trust value for the carried hint.",
        }),
        "despill_strength": ("FLOAT", {
            "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
            "tooltip": "Green despill for processed/QC/cutout (cutout/checker on disk). "
                       "0 = none, 1 = full. Upstream default 0.5.",
        }),
        "window": (WINDOW_CHOICES, {
            "default": "auto",
            "tooltip": "Hann window chunk/stride for stage 1. Bigger = better temporal "
                       "consistency, more VRAM. 'auto' picks the largest that fits free VRAM.",
        }),
        "low_vram": ("BOOLEAN", {
            "default": False,
            "tooltip": "Exact staged inference with lower peak VRAM (slower, uses temp disk). "
                       "Setting it turns off auto selection; with window=auto it uses 80/40.",
        }),
        "unload_other_models": ("BOOLEAN", {
            "default": True,
            "tooltip": "Unload other ComfyUI models first so CorridorKey gets the whole GPU.",
        }),
    }


class CorridorKey2:
    DESCRIPTION = (
        "CorridorKey 2.0 neural green/blue screen keyer. Works with no hint, a single "
        "first-frame hint mask, or one hint mask per frame. Runs at native resolution with "
        "temporal Hann windows across the whole batch."
    )
    OUTPUT_TOOLTIPS = (
        "Straight (un-premultiplied) foreground, no despill. Same colour space as the "
        "input: sRGB, or unclamped scene-linear in Linear mode.",
        "Alpha matte.",
        "Despilled foreground premultiplied by the matte (subject over black), same colour "
        "space as the input. Premultiplied in linear light.",
        "QC composite over a checkerboard with despill, sRGB.",
        "RGBA cutout: despilled straight colour + alpha (upstream's cutout pass). Save with "
        "Save Image for a transparent PNG. Same colour space as the input.",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Green/blue screen frames. The whole batch is "
                                               "keyed as one clip, in order."}),
                "gamma_space": (["sRGB", "Linear"], {
                    "tooltip": "sRGB for normal video/PNG frames. Linear for scene-linear "
                               "plates (e.g. linear EXRs), keeping values above 1.0.",
                }),
                **_shared_inputs(),
            },
            "optional": {
                "mask": ("MASK", {
                    "tooltip": "Optional hint (white = subject). 1 mask = first-frame hint; "
                               "one per frame = full hint video. Leave empty to let the model "
                               "choose the subject.",
                }),
            },
            "hidden": {"unique_id": "UNIQUE_ID", "prompt": "PROMPT"},
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = OUTPUT_NAMES
    FUNCTION = "run"
    CATEGORY = "CorridorKey"

    def run(self, image, gamma_space, hint_quality, carry_hint, carry_hint_quality,
            despill_strength, window, low_vram, unload_other_models, mask=None,
            unique_id=None, prompt=None):
        settings = KeySettings(
            gamma_space=str(gamma_space),
            hint_quality=float(hint_quality),
            carry_hint=bool(carry_hint),
            carry_hint_quality=float(carry_hint_quality),
            despill_strength=float(despill_strength),
            window=str(window),
            low_vram=bool(low_vram),
            unload_other_models=bool(unload_other_models),
        )
        return key_tensors(image, mask, settings, progress=_build_progress_reporter(unique_id),
                           needed=_connected_outputs(prompt, unique_id))


def _default_output_dir() -> str:
    try:
        import folder_paths

        return str(Path(folder_paths.get_output_directory()) / "CorridorKey2")
    except Exception:
        return "CorridorKey2"


class CorridorKey2RenderToDisk:
    DESCRIPTION = (
        "Runs upstream CorridorKey 2.0 inference straight from a video file or frame folder "
        "to EXR sequences or videos on disk. Frames stream from disk, so long 4K plates "
        "don't have to fit in RAM as a ComfyUI batch."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_path": ("STRING", {
                    "default": "", "tooltip": "Video file, or a folder of EXR/PNG/JPG frames "
                                              "(read in sorted filename order)."}),
                "output_dir": ("STRING", {
                    "default": "", "tooltip": "Empty = ComfyUI/output/CorridorKey2."}),
                "output_format": (["exr", "video"], {
                    "tooltip": "exr: unclamped scene-linear OpenEXR sequences. video: "
                               "display-referred previews (MP4, ProRes 4444 for cutout)."}),
                "out_alpha": ("BOOLEAN", {"default": True}),
                "out_fg": ("BOOLEAN", {"default": True}),
                "out_cutout": ("BOOLEAN", {"default": False,
                                           "tooltip": "Straight despilled RGB + alpha."}),
                "out_checker": ("BOOLEAN", {"default": False,
                                            "tooltip": "QC composite over a checkerboard."}),
                "input_is_linear": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Frame folders only: treat frames as scene-linear (linear EXR "
                               "plates). Without it EXRs are clipped to [0, 1]."}),
                "frame_dir_fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 240.0,
                                            "step": 0.001}),
                "exr_dtype": (["half", "float"], {"default": "half"}),
                "cutout_linear": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Video output only: write the cutout as linear instead of sRGB."}),
                "num_frames": ("INT", {"default": -1, "min": -1, "max": 1000000,
                                       "tooltip": "-1 = whole clip."}),
                **_shared_inputs(),
            },
            "optional": {
                "hint_first_frame": ("STRING", {
                    "default": "", "tooltip": "Optional first-frame matte image path."}),
                "hint_video": ("STRING", {
                    "default": "", "tooltip": "Optional per-frame hint video or frame folder."}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output_dir",)
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = "CorridorKey"

    def run(self, input_path, output_dir, output_format, out_alpha, out_fg, out_cutout,
            out_checker, input_is_linear, frame_dir_fps, exr_dtype, cutout_linear,
            num_frames, hint_quality, carry_hint, carry_hint_quality, despill_strength,
            window, low_vram, unload_other_models, hint_first_frame="", hint_video=""):
        input_path = str(input_path).strip().strip('"')
        if not input_path or not os.path.exists(input_path):
            raise FileNotFoundError(f"input_path not found: {input_path!r}")
        out_dir = str(output_dir).strip().strip('"') or _default_output_dir()
        outputs = {name for name, on in (("alpha", out_alpha), ("fg", out_fg),
                                         ("cutout", out_cutout), ("checker", out_checker))
                   if on}
        if not outputs:
            raise ValueError("Enable at least one output.")
        settings = KeySettings(
            hint_quality=float(hint_quality), carry_hint=bool(carry_hint),
            carry_hint_quality=float(carry_hint_quality),
            despill_strength=float(despill_strength), window=str(window),
            low_vram=bool(low_vram), unload_other_models=bool(unload_other_models))
        render_to_disk(
            input_path, out_dir, outputs=outputs, output_format=str(output_format),
            exr_dtype=str(exr_dtype), input_is_linear=bool(input_is_linear),
            frame_dir_fps=float(frame_dir_fps),
            hint_first_frame=str(hint_first_frame).strip().strip('"') or None,
            hint_video=str(hint_video).strip().strip('"') or None,
            num_frames=int(num_frames), cutout_linear=bool(cutout_linear),
            settings=settings)
        return {"ui": {"text": [out_dir]}, "result": (out_dir,)}
