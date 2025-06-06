from tkinter import Label, Tk, Widget


def create_error_label(root: Tk, error_msg: str, result_labels: list[Widget]) -> None:
    """
    Membuat dan menampilkan label error di UI.

    Args:
        root: Widget root Tkinter.
        error_msg: Pesan error yang akan ditampilkan.
        result_labels: Daftar untuk menyimpan referensi label.
    """
    error_label = Label(root, text=f"Error: {error_msg}", fg="red", font=("Arial", 10))
    error_label.pack()
    result_labels.append(error_label)
