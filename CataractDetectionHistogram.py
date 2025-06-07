import cv2
import imutils
import numpy as np
from matplotlib import pyplot as plt
from tkinter import *
import tkinter.filedialog
from PIL import Image, ImageTk

def select_image():
    """
    Fungsi untuk memilih gambar, memprosesnya untuk deteksi katarak,
    dan menampilkan hasilnya pada citra yang telah diperbaiki.
    """
    path = tkinter.filedialog.askopenfilename()
    if len(path) > 0:
        # 1. Muat dan ubah ukuran gambar
        img = cv2.imread(path)
        img = imutils.resize(img, width=500)

        # 2. Pra-pemrosesan Gambar
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Peningkatan Citra dengan Median Filter untuk menghilangkan noise
        gray_filtered = cv2.medianBlur(gray, 5)
        
        # Lanjutkan dengan Gaussian Blur untuk menghaluskan gambar
        gray_blurred = cv2.GaussianBlur(gray_filtered, (9, 9), 2)

        # 3. Deteksi Pupil (Lingkaran)
        detected_circles = cv2.HoughCircles(gray_blurred,
                                           cv2.HOUGH_GRADIENT, 1, 20, param1=50,
                                           param2=30, minRadius=20, maxRadius=120)

        # 4. Analisis dan Visualisasi Hasil
        if detected_circles is not None:
            detected_circles = np.uint16(np.around(detected_circles))
            
            pt = detected_circles[0][0]
            a, b, r = pt[0], pt[1], pt[2]

            # --- PENYESUAIAN UTAMA ---
            # Buat kanvas display dari gambar yang sudah difilter (bersih dari noise).
            # Ubah dari grayscale kembali ke BGR agar bisa digambar dengan warna.
            improved_display = cv2.cvtColor(gray_filtered, cv2.COLOR_GRAY2BGR)

            analysis_radius = int(r * 0.6)
            analysis_mask = np.zeros(gray.shape, dtype="uint8")
            cv2.circle(analysis_mask, (a, b), analysis_radius, 255, -1)

            mean, stddev = cv2.meanStdDev(gray, mask=analysis_mask)
            
            # Gambar lingkaran deteksi dan analisis PADA CITRA YANG DIPERBAIKI
            cv2.circle(improved_display, (a, b), r, (0, 0, 255), 2)  # Lingkaran deteksi (merah)
            cv2.circle(improved_display, (a, b), analysis_radius, (0, 255, 0), 2) # Lingkaran analisis (hijau)

            # Tentukan status katarak
            mean_val = mean[0][0]
            print(f"Mean di dalam INTI Pupil: {mean_val:.2f}")

            if mean_val < 70:
                result_text = "Status: Sehat (Tidak Ada Katarak)"
            elif mean_val < 140:
                result_text = "Status: Terdeteksi Katarak Ringan"
            else:
                result_text = "Status: Terdeteksi Katarak Parah"
            
            print(result_text)

            # Tampilkan hasil pada CITRA YANG DIPERBAIKI
            cv2.putText(improved_display, result_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Hasil Deteksi (Citra Perbaikan)", improved_display)

            # Tampilkan histogram
            hist_roi = cv2.calcHist([gray], [0], analysis_mask, [256], [0, 256])
            plt.figure("Histogram Inti Pupil")
            plt.title("Distribusi Piksel di Area Inti Pupil")
            plt.xlabel("Intensitas Piksel")
            plt.ylabel("Jumlah Piksel")
            plt.plot(hist_roi)
            plt.axvline(mean_val, color='r', linestyle='dashed', linewidth=1)
            plt.xlim([0, 256])
            plt.show()

        else:
            print("Peringatan: Tidak dapat mendeteksi pupil pada gambar.")
            cv2.imshow("Gambar Asli (Pupil Tidak Terdeteksi)", img)

# --- Pengaturan GUI Tkinter ---
root = Tk()
root.title("Detektor Katarak Sederhana V4")
label = Label(root, text="Pilih gambar mata untuk dideteksi")
label.pack(pady=10, padx=20)
btn = Button(root, text="Pilih Gambar", command=select_image)
btn.pack(side="bottom", fill="both", expand="yes", padx="20", pady="20")
root.mainloop()