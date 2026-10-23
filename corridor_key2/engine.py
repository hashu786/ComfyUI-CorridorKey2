"""Model loading, caching and environment compatibility for CorridorKey 2.0.

The upstream `corridorkey` package is vendored unmodified. This module only
adapts how it is loaded inside a long-running ComfyUI process:

* `utils3d` is stubbed when missing. MoGe imports it at module level, but the
  only MoGe path CorridorKey uses (the DINOv2 encoder) never calls it.
* C-RADIO is instantiated directly and its safetensors are loaded with
  `load_state_dict`, instead of going through `PreTrainedModel.from_pretrained`.
  That keeps it working on transformers 5.x, which ComfyUI installs commonly
  ship and which upstream pins below. If the direct path fails, the upstream
  loader is used as a fallback.
* Upstream sets HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE for the whole process
  while loading. They are restored afterwards so other nodes can still
  download their models.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import threading
import types
from pathlib import Path

import torch

from ..corridorkey import foundations as ck_foundations
from ..corridorkey import runtime as ck_runtime

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PACKAGE_ROOT / "models"

RVM_FILE = "pretrain_rvm_small16_256_204031069.npz"

_LOG_PREFIX = "[CorridorKey2]"


def log(message: str) -> None:
    print(f"{_LOG_PREFIX} {message}")


# ---------------------------------------------------------------------------
# utils3d stub
# ---------------------------------------------------------------------------


class _MissingUtils3d(types.ModuleType):
    """Placeholder that imports cleanly but raises if any function is called."""

    def __getattr__(self, name: str):
        if name.startswith("__"):
            raise AttributeError(name)
        return _MissingUtils3d(f"{self.__name__}.{name}")

    def __call__(self, *args, **kwargs):
        raise RuntimeError(
            f"{self.__name__} was called but utils3d is not installed. CorridorKey 2.0 "
            "only uses MoGe's encoder, which does not need it. Install utils3d if "
            "another tool needs MoGe's geometry functions."
        )


def _ensure_utils3d() -> None:
    if "utils3d" in sys.modules:
        return
    try:
        import utils3d  # noqa: F401
    except Exception:
        sys.modules["utils3d"] = _MissingUtils3d("utils3d")
        log("utils3d not installed; using an import stub (not needed for keying).")


# ---------------------------------------------------------------------------
# C-RADIO loader that does not depend on transformers' from_pretrained
# ---------------------------------------------------------------------------


def _import_cradio_module(repo_path: Path):
    # Same package trick as upstream foundations._load_local_cradio_so400m, so
    # the snapshot's relative imports resolve.
    hf_model_path = repo_path / "hf_model.py"
    if not hf_model_path.is_file():
        raise FileNotFoundError(f"C-RADIO snapshot is missing {hf_model_path}")
    package_name = "_corridorkey_cradio_snapshot"
    package = sys.modules.get(package_name)
    if package is None:
        package = types.ModuleType(package_name)
        package.__file__ = str(repo_path / "__init__.py")
        package.__path__ = [str(repo_path)]
        sys.modules[package_name] = package
    module_name = f"{package_name}.hf_model"
    module = sys.modules.get(module_name)
    if module is None:
        spec = importlib.util.spec_from_file_location(module_name, hf_model_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not import C-RADIO module at {hf_model_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    return module


def _read_safetensors_dir(repo_path: Path) -> dict:
    from safetensors.torch import load_file

    index_path = repo_path / "model.safetensors.index.json"
    if index_path.is_file():
        with open(index_path, encoding="utf-8") as f:
            files = sorted(set(json.load(f)["weight_map"].values()))
    else:
        files = ["model.safetensors"]
    state: dict = {}
    for name in files:
        state.update(load_file(str(repo_path / name), device="cpu"))
    return state


_CONFIG_META_KEYS = {
    "architectures", "auto_map", "model_type", "transformers_version",
    "torch_dtype", "dtype", "_name_or_path",
}


def _load_cradio_direct(repo_path: Path):
    module = _import_cradio_module(repo_path)
    try:
        config = module.RADIOConfig.from_pretrained(str(repo_path), local_files_only=True)
    except Exception as exc:
        log(f"RADIOConfig.from_pretrained failed ({exc!r}); building config from config.json")
        with open(repo_path / "config.json", encoding="utf-8") as f:
            raw = json.load(f)
        config = module.RADIOConfig(
            **{k: v for k, v in raw.items() if k not in _CONFIG_META_KEYS})

    model = module.RADIOModel(config)
    state = _read_safetensors_dir(repo_path)
    missing, unexpected = model.load_state_dict(state, strict=False)
    param_names = {name for name, _ in model.named_parameters()}
    missing_params = [k for k in missing if k in param_names]
    if missing_params:
        raise RuntimeError(
            f"C-RADIO checkpoint is missing {len(missing_params)} parameters "
            f"(e.g. {missing_params[:4]})")
    if unexpected:
        log(f"C-RADIO: ignored {len(unexpected)} unexpected checkpoint tensors")
    dtypes = sorted({str(p.dtype).replace("torch.", "") for p in model.parameters()})
    log(f"C-RADIO loaded directly ({len(state)} tensors, dtype {'/'.join(dtypes)})")
    return model


def _patch_cradio_loader() -> None:
    if getattr(ck_foundations, "_ck2_patched", False):
        return
    upstream_loader = ck_foundations._load_local_cradio_so400m

    def robust_loader(repo_path):
        try:
            return _load_cradio_direct(Path(repo_path))
        except Exception as exc:
            log(f"Direct C-RADIO load failed ({exc!r}); trying upstream from_pretrained")
            return upstream_loader(repo_path)

    ck_foundations._load_local_cradio_so400m = robust_loader
    ck_foundations._ck2_patched = True


# ---------------------------------------------------------------------------
# Paths and cached model
# ---------------------------------------------------------------------------


def _prefer_safetensors(folder: Path, stem: str) -> Path:
    st = folder / f"{stem}.safetensors"
    pt = folder / f"{stem}.pt"
    if not st.exists() and pt.exists():
        return pt
    return st


# Models live in ComfyUI/models/corridorkey2/ by default. That survives node
# reinstalls/updates, and a `corridorkey2:` entry in extra_model_paths.yaml can
# point it at a shared location instead. The old in-node models/ folder is still
# found if it holds a complete set.
MODEL_FOLDER = "corridorkey2"


def _register_model_folder() -> None:
    try:
        import folder_paths

        folder_paths.add_model_folder_path(
            MODEL_FOLDER, os.path.join(folder_paths.models_dir, MODEL_FOLDER))
    except Exception:  # pragma: no cover - outside ComfyUI
        pass


_register_model_folder()


def model_roots() -> list[Path]:
    roots: list[Path] = []
    try:
        import folder_paths

        roots += [Path(p) for p in folder_paths.get_folder_paths(MODEL_FOLDER)]
    except Exception:  # pragma: no cover - outside ComfyUI
        pass
    roots.append(MODELS_DIR)  # legacy location inside the node folder
    seen, unique = set(), []
    for r in roots:
        key = os.path.normcase(os.path.abspath(r))
        if key not in seen:
            seen.add(key)
            unique.append(r)
    return unique


def resolve_paths(root: Path | None = None) -> ck_runtime.RuntimePaths:
    root = Path(root) if root is not None else model_roots()[0]
    weights_dir = root / "weights"
    vfms_dir = root / "vfms"
    return ck_runtime.RuntimePaths(
        # package_root is where upstream finds third_party/MoGe.
        package_root=PACKAGE_ROOT,
        weights=_prefer_safetensors(weights_dir, "corridorkey_v2"),
        foundation_stats=_prefer_safetensors(weights_dir, "foundation_stats"),
        rvm_ckpt=vfms_dir / "rvm" / RVM_FILE,
        moge_checkpoint=vfms_dir / "moge" / "model.pt",
        cradio_dir=vfms_dir / "cradio" / "C-RADIOv4-SO400M",
    )


_REQUIRED_FILES = 6  # files checked by _missing()


def _missing(paths: ck_runtime.RuntimePaths) -> list[Path]:
    required = (
        paths.weights, paths.foundation_stats, paths.rvm_ckpt, paths.moge_checkpoint,
        Path(paths.cradio_dir) / "model.safetensors", Path(paths.cradio_dir) / "hf_model.py",
    )
    return [Path(p) for p in required if not Path(p).exists()]


def find_complete_paths() -> ck_runtime.RuntimePaths | None:
    for root in model_roots():
        paths = resolve_paths(root)
        if not _missing(paths):
            return paths
    return None


def download_target() -> Path:
    """Where to download: a root that already has some files, else the first root."""
    roots = model_roots()
    for root in roots:
        paths = resolve_paths(root)
        if len(_missing(paths)) < _REQUIRED_FILES:  # partly downloaded: resume there
            return root
    return roots[0]


def manual_download_message(root: Path) -> str:
    return (
        f"Download them manually into {root} (about 5 GB) by running\n"
        "  python -s ComfyUI/custom_nodes/ComfyUI-CorridorKey2/scripts/download_weights.py "
        f'--package_root "{root}"\n'
        "with ComfyUI's Python (python_embeded\\python.exe on Windows portable), "
        "then queue the workflow again."
    )


def _load_download_script():
    script = PACKAGE_ROOT / "scripts" / "download_weights.py"
    spec = importlib.util.spec_from_file_location("_corridorkey2_download_weights", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ensure_models(status=None) -> ck_runtime.RuntimePaths:
    """Return complete model paths, downloading them on first use if needed.

    Set CORRIDORKEY2_NO_DOWNLOAD=1 to disable the automatic download.
    """
    paths = find_complete_paths()
    if paths is not None:
        return paths
    root = download_target()
    missing = _missing(resolve_paths(root))
    if os.environ.get("CORRIDORKEY2_NO_DOWNLOAD", "").strip() not in ("", "0"):
        lines = "\n".join(f"  - {p}" for p in missing)
        raise FileNotFoundError(
            "CorridorKey 2.0 model files are missing (auto-download is disabled by "
            f"CORRIDORKEY2_NO_DOWNLOAD):\n{lines}\n\n" + manual_download_message(root))
    msg = (f"First run: downloading CorridorKey 2.0 models (about 5 GB) to {root}. "
           "This takes a few minutes; progress is shown in the console.")
    log(msg)
    if status is not None:
        status("downloading models (first run, ~5 GB) - see console")
    try:
        dl = _load_download_script()
        dl.download_all(root)
    except Exception as exc:
        raise RuntimeError(
            f"Automatic download of the CorridorKey 2.0 models failed: {exc!r}\n\n"
            + manual_download_message(root)) from exc
    paths = resolve_paths(root)
    still = _missing(paths)
    if still:
        raise FileNotFoundError(
            "Download finished but files are still missing: "
            + ", ".join(str(p) for p in still) + "\n\n" + manual_download_message(root))
    log(f"Models ready in {root}")
    return paths


_lock = threading.Lock()
_model = None
_model_key = None


def get_model(status=None):
    """Load (once) and return the CorridorKey 2.0 model. Parts may be on CPU."""
    global _model, _model_key
    if not torch.cuda.is_available():
        raise RuntimeError("CorridorKey 2.0 requires an NVIDIA CUDA GPU.")
    with _lock:
        paths = ensure_models(status)
        key = (str(paths.weights), Path(paths.weights).stat().st_mtime)
        if _model is not None and _model_key == key:
            return _model
        _model = None
        if status is not None:
            status("loading model")
        _ensure_utils3d()
        _patch_cradio_loader()
        env_keys = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
        saved_env = {k: os.environ.get(k) for k in env_keys}
        try:
            model = ck_runtime.load_model(paths, device="cuda")
        finally:
            for k, v in saved_env.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        model.cpu()
        torch.cuda.empty_cache()
        _model, _model_key = model, key
        return _model


def offload(model) -> None:
    model.cpu()
    torch.cuda.empty_cache()


def free_comfy_vram() -> None:
    """Ask ComfyUI to unload its models so auto window selection sees free VRAM."""
    try:
        import comfy.model_management as mm

        mm.unload_all_models()
        mm.soft_empty_cache()
    except Exception as exc:  # pragma: no cover - outside ComfyUI
        log(f"Could not unload other ComfyUI models: {exc!r}")
    torch.cuda.empty_cache()


def comfy_temp_dir() -> Path:
    try:
        import folder_paths

        base = Path(folder_paths.get_temp_directory())
    except Exception:  # pragma: no cover - outside ComfyUI
        import tempfile

        base = Path(tempfile.gettempdir())
    path = base / "corridorkey2"
    path.mkdir(parents=True, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# torch >= 2.10 compatibility
# ---------------------------------------------------------------------------


class _RmsNormKeepsDtype(types.ModuleType):
    """torch.nn.functional, except rms_norm returns its input's dtype.

    Upstream was built on torch 2.9, where autocast left F.rms_norm in the
    activation dtype (bf16). From torch 2.10 autocast runs it in fp32, so q/k
    reach attention as fp32 while SDPA returns bf16, and upstream's strict
    index_copy_ / index_put_ fail ("expected the same dtype"). Casting the
    result back restores the torch 2.9 dataflow. It is installed only into the
    vendored blocks.py, the sole caller of rms_norm, so nothing else in
    ComfyUI is affected.
    """

    def __init__(self):
        super().__init__("corridorkey2_functional")
        import torch.nn.functional as F

        self._F = F

    def __getattr__(self, name):
        return getattr(self._F, name)

    def rms_norm(self, input, *args, **kwargs):
        out = self._F.rms_norm(input, *args, **kwargs)
        return out if out.dtype == input.dtype else out.to(input.dtype)


def apply_torch_compat() -> None:
    from ..corridorkey import blocks as ck_blocks

    if not isinstance(ck_blocks.F, _RmsNormKeepsDtype):
        ck_blocks.F = _RmsNormKeepsDtype()


def safe_remove(path) -> None:
    """Upstream safe_remove, but a file Windows still has locked is skipped.

    After an error or interrupt, a numpy memmap can stay open until garbage
    collection. Upstream's os.remove then raises PermissionError and hides the
    original error. A skipped file sits in ComfyUI/temp, which ComfyUI clears
    on startup.
    """
    if path is None:
        return
    try:
        os.remove(path)
    except FileNotFoundError:
        pass
    except PermissionError:
        log(f"temp file still in use, left for ComfyUI to clear on restart: {path}")


apply_torch_compat()
ck_runtime.safe_remove = safe_remove  # used by upstream run_inference's cleanup


class _ZeroLowVram:
    low_vram_enc_frame_chunk = 0
    low_vram_mlp_chunk_tokens = 0
    low_vram_swin_window_batch = 0
    low_vram_full_attn_query_chunk_tokens = 0


def set_low_vram(model, cfg: ck_runtime.InferenceConfig) -> None:
    """Apply or clear upstream's low-VRAM chunking.

    Upstream configures this once per process. ComfyUI reuses the model across
    runs, so it is reset when a later run doesn't use low-VRAM mode.
    """
    ck_runtime.configure_low_vram_inference(model, cfg if cfg.low_vram else _ZeroLowVram())
