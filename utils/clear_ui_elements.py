from tkinter import Widget


def clear_ui_elements(elements: list[Widget]) -> None:
    """
    Menghapus semua elemen UI dari daftar dan membersihkan daftar.

    Args:
        elements: Daftar elemen UI yang akan dihapus.
    """
    for element in elements:
        element.destroy()
    elements.clear()
