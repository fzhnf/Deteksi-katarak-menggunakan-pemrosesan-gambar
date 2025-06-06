from tkinter import Button, Frame, Label, Tk, Widget
from typing import final

import cv2
from cv2.typing import MatLike
from matplotlib import pyplot as plt

import utils


@final
class SimpleCataractDetector:
    """
    Kelas untuk aplikasi Deteksi Katarak sederhana.

    Kelas ini menggunakan pendekatan berbasis mean intensity
    untuk mendeteksi katarak pada gambar mata.
    """

    def __init__(self):
        """
        Inisialisasi aplikasi dan mengatur jendela utama.
        """
        self.root = Tk()
        self.root.title("Sistem Deteksi Katarak Sederhana")
        self.root.geometry("300x150")
        self.result_labels: list[Widget] = []
        self.setup_ui()

    def setup_ui(self):
        """
        Mengatur elemen-elemen antarmuka pengguna (UI) awal.
        """
        btn = Button(
            self.root,
            text="Pilih Gambar Mata",
            command=self.select_image,
            padx=20,
            pady=10,
            bg="#4a7abc",
            fg="white",
            font=("Arial", 12, "bold"),
        )
        btn.pack(expand=True)

    def clear_previous_results(self):
        """
        Menghapus semua hasil dari analisis sebelumnya dari UI.
        """
        # Use the utility function to clear UI elements
        utils.clear_ui_elements(self.result_labels)

    def select_image(self):
        """
        Membuka dialog untuk memilih file gambar, kemudian memproses
        dan menampilkan hasilnya.
        """
        path = utils.select_image_file()

        if not path:
            return

        try:
            self.clear_previous_results()
            img = utils.load_image(path)  # Use utility function to load image

            # Ubah ukuran gambar untuk konsistensi
            img = utils.resize_image(img, width=500)  # Use utility resize function
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)

            # Hitung mean dan standard deviation
            mean_arr = cv2.meanStdDev(gray)[0]
            std_arr = cv2.meanStdDev(gray)[1]
            mean_val = float(mean_arr[0, 0])  # pyright: ignore[reportAny]
            std_val = float(std_arr[0, 0])  # pyright: ignore[reportAny]

            # Diagnosis berdasarkan mean intensity
            diagnosis = self.diagnose_cataract(mean_val)

            self.display_results(mean_val, std_val, diagnosis)
            self.show_histogram(gray, mean_val)

        except Exception as e:
            # Use utility function to create error label
            utils.create_error_label(self.root, str(e), self.result_labels)

    def diagnose_cataract(self, mean_val: float) -> tuple[str, str]:
        """
        Mendiagnosis katarak berdasarkan nilai mean intensity.

        Args:
            mean_val: Nilai mean intensity dari gambar grayscale.

        Returns:
            Tuple berisi status dan deskripsi diagnosis.
        """
        if mean_val < 50:
            return "Tidak Ada Katarak", "Mata sehat"
        elif mean_val <= 100:
            return "Katarak Ringan", "Mata memiliki katarak ringan"
        else:
            return "Katarak Parah", "Mata memiliki katarak parah"

    def display_results(
        self, mean_val: float, std_val: float, diagnosis: tuple[str, str]
    ):
        """
        Menampilkan hasil analisis di UI.

        Args:
            mean_val: Nilai mean intensity.
            std_val: Nilai standard deviation.
            diagnosis: Tuple berisi status dan deskripsi diagnosis.
        """
        results_frame = Frame(self.root)
        results_frame.pack(pady=10)

        status, description = diagnosis

        # Tentukan warna teks berdasarkan diagnosis
        color = (
            "green"
            if "Tidak Ada" in status
            else "orange"
            if "Ringan" in status
            else "red"
        )

        status_label = Label(
            results_frame,
            text=status,
            font=("Arial", 12, "bold"),
            fg=color,
        )
        status_label.pack()
        self.result_labels.append(status_label)

        desc_label = Label(
            results_frame,
            text=description,
            font=("Arial", 10),
        )
        desc_label.pack()
        self.result_labels.append(desc_label)

        mean_label = Label(
            results_frame,
            text=f"Mean: {mean_val:.2f}",
            font=("Arial", 10, "bold"),
        )
        mean_label.pack()
        self.result_labels.append(mean_label)

        std_label = Label(
            results_frame,
            text=f"Standard Deviation: {std_val:.2f}",
            font=("Arial", 10, "bold"),
        )
        std_label.pack()
        self.result_labels.append(std_label)

        self.result_labels.append(results_frame)

    def show_histogram(self, gray: MatLike, mean_val: float):
        """
        Menampilkan histogram dari distribusi intensitas grayscale.

        Args:
            gray: Gambar grayscale untuk histogram.
            mean_val: Nilai mean untuk ditampilkan sebagai garis vertikal.
        """
        try:
            _ = plt.hist(gray.ravel(), 256, range=(0, 256))  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]
            _ = plt.axvline(mean_val, color="k", linestyle="dashed", linewidth=1)  # pyright: ignore[reportUnknownMemberType]
            _ = plt.title("Distribusi Intensitas Grayscale")  # pyright: ignore[reportUnknownMemberType]
            _ = plt.xlabel("Nilai Intensitas")  # pyright: ignore[reportUnknownMemberType]
            _ = plt.ylabel("Jumlah Piksel")  # pyright: ignore[reportUnknownMemberType]
            plt.show()  # pyright: ignore[reportUnknownMemberType]

        except Exception as e:
            # Use utility function for error label
            utils.create_error_label(
                self.root, f"Gagal menampilkan histogram: {str(e)}", self.result_labels
            )

    def run(self):
        """
        Menjalankan loop utama aplikasi Tkinter.
        """
        self.root.mainloop()


def main():
    """
    Fungsi utama untuk membuat instance dan menjalankan aplikasi.
    """
    app = SimpleCataractDetector()
    app.run()


if __name__ == "__main__":
    main()
