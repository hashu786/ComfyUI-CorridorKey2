"""
@title: ComfyUI-CorridorKey2
@nickname: CorridorKey 2.0
@description: ComfyUI nodes for CorridorKey 2.0 neural green/blue screen keying.
"""

from .nodes import CorridorKey2, CorridorKey2RenderToDisk

NODE_CLASS_MAPPINGS = {
    "CorridorKey2": CorridorKey2,
    "CorridorKey2RenderToDisk": CorridorKey2RenderToDisk,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CorridorKey2": "CorridorKey 2.0",
    "CorridorKey2RenderToDisk": "CorridorKey 2.0 (Render to Disk)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
