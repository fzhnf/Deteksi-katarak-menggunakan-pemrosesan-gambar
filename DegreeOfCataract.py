from tkinter import Button, Frame, Label, Tk, Widget
from typing import cast, final

import cv2
import numpy as np
from cv2.typing import MatLike
from PIL import Image, ImageTk

from utils import (
    clear_ui_elements,
    create_error_label,
    load_image,
    resize_image,
    select_image_file,
)


@final
class CataractDetector:
    """
    Kelas utama untuk aplikasi Deteksi Katarak.

    Kelas ini menangani antarmuka pengguna (UI), pemrosesan gambar,
    dan menampilkan hasil deteksi katarak.
    """

    def __init__(self):
        """
        Inisialisasi aplikasi dan mengatur jendela utama.
        """
        self.root = Tk()
        self.root.title("Sistem Deteksi Katarak")
        self.image_references: list[ImageTk.PhotoImage] = []
        self.result_elements: list[Widget] = []
        self.setup_ui()

    def setup_ui(self):
        """
        Mengatur elemen-elemen antarmuka pengguna (UI) awal.
        """
        btn = Button(
            self.root,
            text="Pilih Gambar Mata",
            command=self.select_image,
            font=("Arial", 12),
        )
        btn.pack(side="bottom", fill="both", expand=True, padx=10, pady=10)
        self.result_elements.append(btn)

    def clear_previous_results(self):
        """
        Menghapus semua hasil dan gambar dari analisis sebelumnya dari UI.
        """
        clear_ui_elements(self.result_elements)
        self.image_references.clear()

    def select_image(self):
        """
        Membuka dialog untuk memilih file gambar, kemudian memproses
        dan menampilkan hasilnya.
        """
        path = select_image_file()
        if not path:
            return

        try:
            self.clear_previous_results()
            img = load_image(path)

            # Ubah ukuran gambar untuk konsistensi
            img = resize_image(img, width=500)

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Terapkan filter untuk menghaluskan gambar
            kernel = np.ones((5, 5), np.float32) / 25
            imgfiltered = cv2.filter2D(gray, -1, kernel)

            # Buat kernel untuk operasi morfologi
            kernelOp = np.ones((10, 10), np.uint8)

            # Terapkan thresholding untuk segmentasi
            _, thresh_image = cv2.threshold(imgfiltered, 50, 255, cv2.THRESH_BINARY_INV)

            # Operasi morfologi (opening) untuk membersihkan noise
            morpho = cv2.morphologyEx(thresh_image, cv2.MORPH_OPEN, kernelOp)

            # Deteksi lingkaran (iris) menggunakan Hough Transform
            circles = cv2.HoughCircles(
                morpho,
                cv2.HOUGH_GRADIENT,
                1,
                20,
                param1=50,
                param2=30,
                minRadius=0,
                maxRadius=0,
            )

            # Periksa jika tidak ada lingkaran yang terdeteksi
            if circles.size == 0:
                raise ValueError("Tidak ada lingkaran (iris) yang terdeteksi")

            circles = np.round(circles[0, :]).astype("int")
            if len(circles) == 0:
                raise ValueError("Tidak ditemukan lingkaran (iris) yang valid")

            # Ambil koordinat dan radius dari lingkaran pertama yang terdeteksi
            x = cast(int, circles[0][0])
            y = cast(int, circles[0][1])
            r = cast(int, circles[0][2])

            # Buat mask untuk mengisolasi area iris
            img_morpho_copy = morpho.copy()
            rows = cast(int, img_morpho_copy.shape[0])
            cols = cast(int, img_morpho_copy.shape[1])

            # Buat mask sirkular menggunakan operasi vektor untuk efisiensi
            y_grid, x_grid = np.ogrid[:rows, :cols]
            mask = (x_grid - x) ** 2 + (y_grid - y) ** 2 <= r**2
            img_morpho_copy[~mask] = 0

            # Inversi gambar untuk mendeteksi katarak (area putih)
            imgg_inv = cv2.bitwise_not(img_morpho_copy)

            # Temukan kontur untuk menghitung area pupil
            contours_pupil, _ = cv2.findContours(
                img_morpho_copy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )

            pupil_area = 0
            if contours_pupil:
                largest_contour = max(contours_pupil, key=cv2.contourArea)
                pupil_area = cv2.contourArea(largest_contour)

            if pupil_area == 0:
                raise ValueError("Tidak dapat mendeteksi area pupil")

            # Temukan kontur untuk deteksi katarak
            contours_cat, _ = cv2.findContours(
                imgg_inv, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE
            )

            # Siapkan gambar untuk visualisasi hasil deteksi katarak
            cimg_cat = img_rgb.copy()
            cat_area = 0

            # Hitung total area katarak
            for cnt in contours_cat:
                area = cv2.contourArea(cnt)
                if area < pupil_area and area > 50:  # Filter noise kecil
                    _ = cv2.drawContours(cimg_cat, [cnt], -1, (0, 255, 0), 2)
                    cat_area += area

            # Hitung persentase katarak
            if cat_area > 0:
                cataract_percentage = (cat_area / (pupil_area + cat_area)) * 100
            else:
                cataract_percentage = 0.0

            self.display_results(pupil_area, cat_area, cataract_percentage)
            self.display_images(
                img_rgb, imgfiltered, thresh_image, img_morpho_copy, imgg_inv, cimg_cat
            )

        except Exception as e:
            create_error_label(self.root, str(e), self.result_elements)

    def display_results(
        self, pupil_area: float, cat_area: float, cataract_percentage: float
    ):
        """
        Menampilkan hasil analisis (area pupil, area katarak, persentase) di UI.

        Args:
            pupil_area: Area pupil yang terdeteksi dalam piksel.
            cat_area: Area katarak yang terdeteksi dalam piksel.
            cataract_percentage: Persentase katarak yang dihitung.
        """
        results_frame = Frame(self.root)
        results_frame.pack(pady=10)
        self.result_elements.append(results_frame)

        pupil_label = Label(
            results_frame,
            text=f"Area Pupil: {pupil_area:.0f} piksel",
            font=("Arial", 10, "bold"),
        )
        pupil_label.pack()
        self.result_elements.append(pupil_label)

        cat_label = Label(
            results_frame,
            text=f"Area Katarak: {cat_area:.0f} piksel",
            font=("Arial", 10, "bold"),
        )
        cat_label.pack()
        self.result_elements.append(cat_label)

        # Tentukan warna teks berdasarkan tingkat keparahan katarak
        color = (
            "green"
            if cataract_percentage < 10
            else "orange"
            if cataract_percentage < 30
            else "red"
        )
        percentage_label = Label(
            results_frame,
            text=f"Persentase Katarak: {cataract_percentage:.2f}%",
            font=("Arial", 12, "bold"),
            fg=color,
        )
        percentage_label.pack()
        self.result_elements.append(percentage_label)

    def display_images(
        self,
        img_rgb: MatLike,
        imgfiltered: MatLike,
        thresh_image: MatLike,
        img_morpho_copy: MatLike,
        imgg_inv: MatLike,
        cimg_cat: MatLike,
    ):
        """
        Menampilkan gambar-gambar dari setiap tahap pemrosesan di UI.

        Args:
            img_rgb: Gambar asli (RGB).
            imgfiltered: Gambar setelah difilter.
            thresh_image: Gambar setelah thresholding.
            img_morpho_copy: Mask pupil yang dihasilkan.
            imgg_inv: Gambar mask yang diinversi.
            cimg_cat: Gambar akhir dengan deteksi katarak.
        """
        # Konversi gambar grayscale ke 3-channel (RGB) untuk ditampilkan
        imgfiltered_rgb = cv2.cvtColor(imgfiltered, cv2.COLOR_GRAY2RGB)
        thresh_image_rgb = cv2.cvtColor(thresh_image, cv2.COLOR_GRAY2RGB)
        img_morpho_copy_rgb = cv2.cvtColor(img_morpho_copy, cv2.COLOR_GRAY2RGB)
        imgg_inv_rgb = cv2.cvtColor(imgg_inv, cv2.COLOR_GRAY2RGB)

        images = [
            img_rgb,
            imgfiltered_rgb,
            thresh_image_rgb,
            img_morpho_copy_rgb,
            imgg_inv_rgb,
            cimg_cat,
        ]

        # Konversi array numpy ke objek gambar PIL
        pil_images = [Image.fromarray(img) for img in images]

        # Ubah ukuran gambar untuk thumbnail
        width, height = 120, 120
        resized_images: list[ImageTk.PhotoImage] = []
        for img in pil_images:
            resized_img = img.resize((width, height), Image.Resampling.LANCZOS)
            resized_images.append(ImageTk.PhotoImage(resized_img))

        # Simpan referensi gambar agar tidak dihapus oleh garbage collector Python
        self.image_references.extend(resized_images)

        # Buat frame untuk menampilkan gambar-gambar proses
        images_frame = Frame(self.root)
        images_frame.pack(pady=10)
        self.result_elements.append(images_frame)

        labels = [
            "Asli",
            "Filter",
            "Threshold",
            "Mask Pupil",
            "Inversi",
            "Deteksi Katarak",
        ]

        # Tampilkan setiap gambar beserta labelnya
        for tk_image, label_text in zip(resized_images, labels):
            panel_frame = Frame(images_frame)
            panel_frame.pack(side="left", padx=5, pady=5)
            self.result_elements.append(panel_frame)

            panel = Label(panel_frame, image=tk_image)
            panel.pack()
            self.result_elements.append(panel)

            text_label = Label(panel_frame, text=label_text, font=("Arial", 8))
            text_label.pack()
            self.result_elements.append(text_label)

    def run(self):
        """
        Menjalankan loop utama aplikasi Tkinter.
        """
        self.root.mainloop()


def main():
    """
    Fungsi utama untuk membuat instance dan menjalankan aplikasi.
    """
    app = CataractDetector()
    app.run()


if __name__ == "__main__":
    main()
