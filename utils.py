# utils.py
import numpy as np
from PIL import Image
import xospy


def get_webcam_frame(height: int=512) -> np.ndarray:
    cam_w, _ = xospy.video.webcam.get_resolution()
    cam_bytes = xospy.video.webcam.get_frame()
    bytes_per_pixel = 3
    total_pixels = len(cam_bytes) // bytes_per_pixel
    cam_h = total_pixels // cam_w

    if cam_w * cam_h * bytes_per_pixel != len(cam_bytes):
        raise Exception("Webcam resolution doesn't match buffer size. Skipping.")

    cam_array = np.frombuffer(cam_bytes, dtype=np.uint8).reshape((cam_h, cam_w, 3))

    scale = cam_h / height
    new_w = int(cam_w / scale)
    new_h = height
    cam_array = np.array(Image.fromarray(cam_array).resize((new_w, new_h), Image.LANCZOS))

    cam_array = cam_array[:, ::-1]  # horizontal flip
    cam_array = np.mean(cam_array, axis=2).astype(np.uint8)
    cam_array = np.expand_dims(cam_array, axis=2)

    return cam_array


def draw_cross(
    frame: np.ndarray,
    x: float,
    y: float,
    size: int = 10,
    color=(255, 0, 0, 255),
    thickness: int = 3,
):
    height, width, _ = frame.shape
    x, y = int(x), int(y)

    half_thick = thickness // 2

    # Horizontal line
    for dx in range(-size, size + 1):
        xi = x + dx
        if 0 <= xi < width:
            for dy in range(-half_thick, half_thick + 1):
                yi = y + dy
                if 0 <= yi < height:
                    frame[yi, xi] = color

    # Vertical line
    for dy in range(-size, size + 1):
        yi = y + dy
        if 0 <= yi < height:
            for dx in range(-half_thick, half_thick + 1):
                xi = x + dx
                if 0 <= xi < width:
                    frame[yi, xi] = color

