import tkinter.filedialog
from tkinter import Button, Frame, Label, Tk
from typing import cast, final

import cv2
from cv2.typing import MatLike
from matplotlib import pyplot as plt


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
        self.result_labels: list[Label | Frame] = []
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
        for label in self.result_labels:
            label.destroy()
        self.result_labels.clear()

    def manual_resize(
        self,
        image: MatLike,
        width: int | None = None,
        height: int | None = None,
        inter: int = cv2.INTER_AREA,
    ) -> MatLike:
        """
        Mengubah ukuran gambar secara manual dengan mempertahankan aspect ratio.
        Mengganti fungsi imutils.resize().

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

    def select_image(self):
        """
        Membuka dialog untuk memilih file gambar, kemudian memproses
        dan menampilkan hasilnya.
        """
        path = tkinter.filedialog.askopenfilename(
            title="Pilih Gambar Mata",
            filetypes=[("File Gambar", "*.jpg *.jpeg *.png *.bmp *.tiff")],
        )

        if not path:
            return

        try:
            self.clear_previous_results()
            img = cv2.imread(path)

            # Periksa apakah gambar berhasil dibaca atau tidak
            if img.size == 0:
                raise ValueError("Gagal membaca file gambar")

            # Ubah ukuran gambar untuk konsistensi
            img = self.manual_resize(img, width=500)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)

            # Hitung mean dan standard deviation dengan proper typing
            mean_arr = cv2.meanStdDev(gray)[0]
            std_arr = cv2.meanStdDev(gray)[1]
            mean_val = float(mean_arr[0, 0])  # pyright: ignore[reportAny]
            std_val = float(std_arr[0, 0])  # pyright: ignore[reportAny]

            # Diagnosis berdasarkan mean intensity
            diagnosis = self.diagnose_cataract(mean_val)

            self.display_results(mean_val, std_val, diagnosis)
            self.show_histogram(gray, mean_val)

        except Exception as e:
            # Tampilkan pesan error di UI jika terjadi masalah
            error_label = Label(
                self.root, text=f"Error: {str(e)}", fg="red", font=("Arial", 10)
            )
            error_label.pack()
            self.result_labels.append(error_label)

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
            # Jika gagal menampilkan histogram, tampilkan error di UI
            error_label = Label(
                self.root,
                text=f"Gagal menampilkan histogram: {str(e)}",
                fg="orange",
                font=("Arial", 8),
            )
            error_label.pack()
            self.result_labels.append(error_label)

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
