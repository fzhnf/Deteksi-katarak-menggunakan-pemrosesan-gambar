import tkinter.filedialog


def select_image_file() -> str:
    """
    Membuka dialog untuk memilih file gambar.

    Returns:
        Path ke file gambar yang dipilih, atau string kosong jika dibatalkan.
    """
    return tkinter.filedialog.askopenfilename(
        title="Pilih Gambar Mata",
        filetypes=[("File Gambar", "*.jpg *.jpeg *.png *.bmp *.tiff")],
    )
