import cv2
from cv2.typing import MatLike


def load_image(path: str) -> MatLike:
    """
    Memuat dan memvalidasi gambar dari path yang diberikan.

    Args:
        path: Path ke file gambar.

    Returns:
        Gambar yang telah dimuat.

    Raises:
        ValueError: Jika gambar tidak dapat dibaca.
    """
    img = cv2.imread(path)
    if img.size == 0:
        raise ValueError("Gagal membaca file gambar")
    return img
