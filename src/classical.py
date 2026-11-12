"""Classical baseline: morphological black-hat on the limb-flattened disk."""
import cv2


def blackhat_response(flat, size=41, sigma=1.5):
    """closing(image) - image: positive on structures darker than surroundings and narrower than `size`."""
    sm = cv2.GaussianBlur(flat, (0, 0), sigma)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    return cv2.morphologyEx(sm, cv2.MORPH_BLACKHAT, k)
