import numpy as np
import cv2
from segmentation import best_channel, flatten_illumination, clean_page


def _shaded(gray):
    """Multiply by a horizontal 0.55→1.0 illumination gradient."""
    grad = np.linspace(0.55, 1.0, gray.shape[1], dtype=np.float32)
    return (gray.astype(np.float32) * grad[None, :]).astype(np.uint8)


def test_flatten_removes_gradient():
    base = np.full((400, 600), 220, np.uint8)
    cv2.line(base, (50, 200), (550, 200), 30, 3)      # some "ink"
    shaded = _shaded(base)
    flat = flatten_illumination(shaded)
    bg = flat[:150]                                    # ink-free region
    assert float(bg.std()) < 6.0
    assert float(_shaded(base)[:150].std()) > 20.0     # gradient was real


def test_flatten_preserves_ink():
    base = np.full((400, 600), 220, np.uint8)
    cv2.line(base, (50, 200), (550, 200), 30, 3)
    flat = flatten_illumination(_shaded(base))
    assert flat[198:203, 300].min() < 140              # stroke still dark


def test_best_channel_picks_highest_contrast():
    b = np.full((100, 100), 200, np.uint8)             # flat blue channel
    g = np.full((100, 100), 200, np.uint8)
    r = np.full((100, 100), 230, np.uint8)
    cv2.line(r, (0, 50), (99, 50), 20, 3)              # ink visible in red
    bgr = cv2.merge([b, g, r])
    chosen = best_channel(bgr)
    assert np.array_equal(chosen, r)


def test_clean_page_returns_grayscale_same_size():
    bgr = cv2.merge([np.full((300, 500), v, np.uint8) for v in (180, 200, 230)])
    out = clean_page(bgr)
    assert out.ndim == 2 and out.shape == (300, 500)
