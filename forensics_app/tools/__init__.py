"""Register course functionality here so it appears in the sidebar."""

from .grayscale import GrayscaleTool
from .image_info import ImageInfoTool
from .registry import ToolRegistry
from .blur import BlurTool
from .channel_split import ChannelSplitTool
from .channel_swap import ChannelSwapTool
from .histogram import HistogramTool
from .contrast_stretching import ContrastStretchingTool
from .mask import MaskTool


def build_tool_registry() -> ToolRegistry:
    return ToolRegistry(
        [
            ImageInfoTool(),
            GrayscaleTool(),
            BlurTool(),
            ChannelSplitTool(),
            ChannelSwapTool(),
            HistogramTool(),
            ContrastStretchingTool(),
            MaskTool(),
        ]
    )


__all__ = ["ToolRegistry", "build_tool_registry"]
