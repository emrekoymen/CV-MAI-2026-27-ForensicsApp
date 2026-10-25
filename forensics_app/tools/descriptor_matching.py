import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage import feature, measure, transform

from .base import ForensicsTool, ToolResult
from .mask import open_image

GAP = 10                   # pixels between the two images
BACKGROUND = (40, 40, 40)
MAX_RATIO = 0.8            # Lowe's ratio test value recommended for SIFT (D.G. Lowe, "Distinctive Image
                           # Features from Scale-Invariant Keypoints", International Journal of Computer Vision, 2004)

def descriptor_orb(image,n_keypoints=200):
    orb = feature.ORB(n_keypoints=n_keypoints)
    orb.detect_and_extract(image)
    return orb.keypoints, orb.descriptors

def descriptor_sift(image,n_octaves=8,n_scales=3,sigma=1.6):
    sift = feature.SIFT(n_octaves=n_octaves,n_scales=n_scales,sigma_min=sigma)
    sift.detect_and_extract(image)
    return sift.keypoints, sift.descriptors

def descriptor_match(descriptors1,descriptors2,cross_check=True,max_ratio=1.0):
    matches = feature.match_descriptors(descriptors1,descriptors2,cross_check=cross_check,
        max_ratio=max_ratio)
    return matches

def ransac_inliers(keypoints1, keypoints2, matches):
    """Return one bool per match: True if it fits the projective transform found by RANSAC.

    All False when there are fewer than 4 matches or no transform could be found.
    """
    inliers = np.zeros(len(matches), dtype=bool)
    if len(matches) >= 4:
        # skimage transforms work in (x, y) = (col, row), so the keypoints are flipped
        src = keypoints1[matches[:, 0]][:, ::-1]
        dst = keypoints2[matches[:, 1]][:, ::-1]
        _, found = measure.ransac((src, dst), transform.ProjectiveTransform,
            min_samples=4, residual_threshold=2, max_trials=1000, rng=0)
        if found is not None:
            inliers = found
    return inliers

def draw_matches(im1, im2, keypoints1, keypoints2, matches, labels=None, inliers=None):
    """Place both images side by side, circle every keypoint and join matched pairs."""

    # Scale both images to the taller one's height, so a small image isn't dwarfed by a large one.
    # Keypoints were found on the original sizes, so they are scaled by the same factor.
    height = max(im1.height, im2.height)
    scale1, scale2 = height / im1.height, height / im2.height
    im1 = im1.resize((round(im1.width * scale1), height))
    im2 = im2.resize((round(im2.width * scale2), height))

    thickness = max(1, height // 200)
    radius = 4 * thickness

    offset = im1.width + GAP
    font_size = max(10, max(im1.width, im2.width) // 20)
    strip = int(font_size * 1.6) if labels else 0
    canvas = Image.new("RGB", (offset + im2.width, strip + max(im1.height, im2.height)), BACKGROUND)
    canvas.paste(im1, (0, strip))
    canvas.paste(im2, (offset, strip))
    draw = ImageDraw.Draw(canvas)



    if labels:
        font = ImageFont.load_default(size=font_size)
        draw.text((im1.width / 2, strip / 2), labels[0], font=font, anchor="mm")
        draw.text((offset + im2.width / 2, strip / 2), labels[1], font=font, anchor="mm")

    # Keypoints are  in each image. We shift them to where that image sits on the canvas
    to_left = lambda col, row: (col * scale1, row * scale1 + strip)
    to_right = lambda col, row: (col * scale2 + offset, row * scale2 + strip)

    matched1 = keypoints1[matches[:, 0]]
    matched2 = keypoints2[matches[:, 1]]

    unmatched1 = np.ones(len(keypoints1), dtype=bool)
    unmatched1[matches[:, 0]] = False
    unmatched2 = np.ones(len(keypoints2), dtype=bool)
    unmatched2[matches[:, 1]] = False

    for row, col in keypoints1[unmatched1]:
        draw.circle(to_left(col, row), radius=radius, outline="black", width=2 * thickness)
    for row, col in keypoints2[unmatched2]:
        draw.circle(to_right(col, row), radius=radius, outline="black", width=2 * thickness)

    if inliers is None:
        # A different color per match, so crossing lines can be told apart
        colors = [tuple(int(v) for v in c) for c in np.random.default_rng(0).integers(64, 256, size=(len(matches), 3))]
    else:
        colors = [(0, 200, 0) if inlier else (255, 0, 0) for inlier in inliers]
    for (r1, c1), (r2, c2), color in zip(matched1, matched2, colors):
        draw.circle(to_left(c1, r1), radius=radius, outline=color, width=2 * thickness)
        draw.circle(to_right(c2, r2), radius=radius, outline=color, width=2 * thickness)
        draw.line([to_left(c1, r1), to_right(c2, r2)], fill=color, width=thickness)
    return canvas

class DescriptorMatchingTool(ForensicsTool):
    tool_id = "descriptor_matching"
    title = "Descriptor Matching"
    category = "Detection and matching"
    description = "Match the SIFT descriptors of two images and keep the matches consistent with RANSAC."

    def run(self, parent, document):
        image2 = open_image(parent, 'Add second image')
        if image2 is None:
            return None
        img = np.array(document.current.convert("L"), dtype=float) / 255
        img2 = np.array(image2.convert("L"), dtype=float) / 255

        keypoints1, descriptors1 = descriptor_sift(img)
        keypoints2, descriptors2 = descriptor_sift(img2)
        matches = descriptor_match(descriptors1, descriptors2, cross_check=True, max_ratio=MAX_RATIO)
        inliers = ransac_inliers(keypoints1, keypoints2, matches)

        im1 = document.current.convert("L").convert("RGB")
        im2 = image2.convert("L").convert("RGB")
        result = draw_matches(im1, im2, keypoints1, keypoints2, matches, inliers=inliers)
        n_inliers = int(inliers.sum())

        return ToolResult(
            image=result,
            message=f"Found {len(matches)} matches ({n_inliers} RANSAC inliers) using SIFT",
            details={"Operation": "Descriptor matching", "Extractor": "SIFT", "Max ratio": MAX_RATIO,
                        "Keypoints (image 1)": len(keypoints1), "Keypoints (image 2)": len(keypoints2),
                        "Matches": len(matches), "RANSAC inliers": n_inliers},
                )
