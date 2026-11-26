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
from .corner_detection import CornerDectectionTool
from .descriptors import DescriptorsTool
from .descriptor_matching import DescriptorMatchingTool
from .closest_match import ClosestMatchTool
from .k_means import KMeansTool
from .word_histogram import WordHistogramTool
from .classification import ClassificationTool
from .face_detection import FaceDetectionTool, WebcamFaceDetectionTool
from .eigenfaces import EigenfaceBuilderTool
from .face_recognition import FaceRecognitionTool
from .yolo_detection import YoloDetectionTool
from .blood_cell_detection import BloodCellDetectionTool
from .cnn_classification import CnnClassificationTool, ModelPassportTool
from .feature_projection import FeatureProjectionTool
from .clip_classification import ClipZeroShotTool


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
            CornerDectectionTool(),
            DescriptorsTool(),
            DescriptorMatchingTool(),
            ClosestMatchTool(),
            KMeansTool(),
            WordHistogramTool(),
            ClassificationTool(),
            FaceDetectionTool(),
            WebcamFaceDetectionTool(),
            EigenfaceBuilderTool(),
            FaceRecognitionTool(),
            YoloDetectionTool(),
            BloodCellDetectionTool(),
            ModelPassportTool(),
            CnnClassificationTool(),
            FeatureProjectionTool(),
            ClipZeroShotTool(),
        ]
    )


__all__ = ["ToolRegistry", "build_tool_registry"]
