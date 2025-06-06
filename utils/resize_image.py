from typing import cast
import cv2
from cv2.typing import MatLike


def resize_image(
    image: MatLike,
    width: int | None = None,
    height: int | None = None,
    inter: int = cv2.INTER_AREA,
) -> MatLike:
    """
    Mengubah ukuran gambar secara manual dengan mempertahankan aspect ratio.

    Args:
        image: Gambar input yang akan diubah ukurannya.
        width: Lebar target (opsional).
        height: Tinggi target (opsional).
        inter: Metode interpolasi untuk resize.

    Returns:
        Gambar yang telah diubah ukurannya.
    """
    h = cast(int, image.shape[0])
    w = cast(int, image.shape[1])

    # Tangani kasus resize
    match (width, height):
        case (None, None):
            # Tidak ada dimensi yang diberikan, return gambar asli
            return image
        case (None, new_height):
            # Hitung berdasarkan tinggi
            r = new_height / float(h)
            dim = (int(w * r), new_height)
        case (new_width, None):
            # Hitung berdasarkan lebar
            r = new_width / float(w)
            dim = (new_width, int(h * r))
        case (new_width, new_height):
            # Kedua lebar dan tinggi diberikan
            dim = (new_width, new_height)

    # Ubah ukuran gambar
    return cv2.resize(image, dim, interpolation=inter)
