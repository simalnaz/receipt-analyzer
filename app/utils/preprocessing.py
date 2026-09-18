import numpy as np
import cv2
from PIL import Image

DEFAULT_PREPROCESSING_CONFIG = {
    'threshold_block_size': 11,
    'threshold_c': 2,
    'denoise_strength': 10,
    'scale_factor': 2.0,
}


def load_image_as_array(image_path):
    image = Image.open(image_path).convert('RGB')
    return np.array(image)


def to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def denoise_image(image, strength=10):
    return cv2.fastNlMeansDenoising(image, h=strength)


def apply_threshold(image, block_size=11, c=2):
    return cv2.adaptiveThreshold(
        image,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY,
        blockSize=block_size,
        C=c,
    )


def deskew_image(image):
    coords = np.column_stack(np.where(image > 0))
    if len(coords) == 0:
        return image
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = 90 + angle
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        image,
        rotation_matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def scale_image(image, factor=2.0):
    height, width = image.shape[:2]
    return cv2.resize(
        image,
        (int(width * factor), int(height * factor)),
        interpolation=cv2.INTER_CUBIC,
    )


def prepare_image_for_ocr(image_path, config=None):
    if config is None:
        config = DEFAULT_PREPROCESSING_CONFIG
    image = load_image_as_array(image_path)
    gray = to_grayscale(image)
    denoised = denoise_image(gray, strength=config['denoise_strength'])
    thresholded = apply_threshold(
        denoised,
        block_size=config['threshold_block_size'],
        c=config['threshold_c'],
    )
    deskewed = deskew_image(thresholded)
    scaled = scale_image(deskewed, factor=config['scale_factor'])
    return scaled
