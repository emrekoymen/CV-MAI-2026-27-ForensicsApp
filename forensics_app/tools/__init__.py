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
from .sharpening import SharpeningTool
from .edges import EdgeTool
from .convolution import ConvolutionTool
from .filters import FiltersTool
from .canny import CannyTool
from .template_matching import CrossCorrelationTool, TemplateMatchingTool
from .hog import HogTool
from .person_detection import PersonDetectionTool


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
            EdgeTool(),
            SharpeningTool(),
            ConvolutionTool(),
            FiltersTool(),
            CannyTool(),
            CrossCorrelationTool(),
            TemplateMatchingTool(),
            HogTool(),
            PersonDetectionTool(),
        ]
    )


__all__ = ["ToolRegistry", "build_tool_registry"]
