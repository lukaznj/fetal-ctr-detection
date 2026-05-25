import cv2
import numpy as np

def preprocess_image(img_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Preprocessing is intentionally classical:
    - conversion to grayscale
    - intensity normalization
    - median filter for speckle noise
    - CLAHE for local contrast
    - bilateral filter for edge preservation
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

    median = cv2.medianBlur(gray, 5)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    contrast = clahe.apply(median)

    bilateral = cv2.bilateralFilter(contrast, d=7, sigmaColor=45, sigmaSpace=45)

    # Mild sharpening helps keep ellipse edges more visible.
    blur = cv2.GaussianBlur(bilateral, (0, 0), 2.0)
    sharpened = cv2.addWeighted(bilateral, 1.5, blur, -0.5, 0)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    final = cv2.morphologyEx(sharpened, cv2.MORPH_CLOSE, kernel, iterations=1)

    return gray, final
