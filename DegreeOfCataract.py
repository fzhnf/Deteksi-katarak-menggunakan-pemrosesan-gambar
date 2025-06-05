import cv2
from cv2.typing import MatLike
import numpy as np
from tkinter import Tk, Button, Label, Frame
from PIL import ImageTk, Image
import tkinter.filedialog
from typing import final


@final
class CataractDetector:
    def __init__(self):
        self.root = Tk()
        self.root.title("Cataract Detection System")
        self.image_references: list[ImageTk.PhotoImage] = []
        self.result_labels: list[Label | Frame] = []
        self.setup_ui()

    def setup_ui(self):
        btn = Button(
            self.root,
            text="Select an image",
            command=self.select_image,
            font=("Arial", 12),
        )
        btn.pack(side="bottom", fill="both", expand=True, padx=10, pady=10)

    def clear_previous_results(self):
        for label in self.result_labels:
            label.destroy()
        self.result_labels.clear()
        self.image_references.clear()

    def select_image(self):
        path = tkinter.filedialog.askopenfilename(
            title="Select Eye Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")],
        )

        if not path:
            return

        try:
            self.clear_previous_results()
            img = cv2.imread(path)

            if img is None:
                raise ValueError("Failed to read image")

            # Get dimensions with type annotations
            height: int
            width: int
            height, width = img.shape[:2]
            new_width = 500
            new_height = int((new_width / width) * height)
            img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Apply filtering
            kernel = np.ones((5, 5), np.float32) / 25
            imgfiltered = cv2.filter2D(gray, -1, kernel)

            # Create morphological kernels
            kernelOp = np.ones((10, 10), np.uint8)

            # Thresholding
            _, thresh_image = cv2.threshold(imgfiltered, 50, 255, cv2.THRESH_BINARY_INV)

            # Morphological opening
            morpho = cv2.morphologyEx(thresh_image, cv2.MORPH_OPEN, kernelOp)

            # Detect circles using Hough transform
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

            if circles is None:
                raise ValueError("No circles detected")

            circles = np.round(circles[0, :]).astype("int")
            if len(circles) == 0:
                raise ValueError("No valid circles found")

            # Get the first circle with type annotations
            x: int
            y: int
            r: int
            x, y, r = circles[0]

            # Create mask for iris
            img_morpho_copy = morpho.copy()

            # Get dimensions with type annotations
            rows: int
            cols: int
            rows, cols = img_morpho_copy.shape

            # Optimized circular mask using vectorized operations
            y_grid, x_grid = np.ogrid[:rows, :cols]
            mask = (x_grid - x) ** 2 + (y_grid - y) ** 2 <= r**2
            img_morpho_copy[~mask] = 0

            # Invert image
            imgg_inv = cv2.bitwise_not(img_morpho_copy)

            # Find contours for pupil area
            contours_pupil, _ = cv2.findContours(
                img_morpho_copy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )

            pupil_area = 0
            if contours_pupil:
                largest_contour = max(contours_pupil, key=cv2.contourArea)
                pupil_area = cv2.contourArea(largest_contour)

            if pupil_area == 0:
                raise ValueError("Could not detect pupil area")

            # Find contours for cataract detection
            contours_cat, _ = cv2.findContours(
                imgg_inv, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE
            )

            # Create visualization image
            cimg_cat = img_rgb.copy()
            cat_area = 0

            for cnt in contours_cat:
                area = cv2.contourArea(cnt)
                if area < pupil_area and area > 50:
                    _ = cv2.drawContours(cimg_cat, [cnt], -1, (0, 255, 0), 2)
                    cat_area += area

            # Calculate cataract percentage
            if cat_area > 0:
                cataract_percentage = (cat_area / (pupil_area + cat_area)) * 100
            else:
                cataract_percentage = 0.0

            self.display_results(pupil_area, cat_area, cataract_percentage)
            self.display_images(
                img_rgb, imgfiltered, thresh_image, img_morpho_copy, imgg_inv, cimg_cat
            )

        except Exception as e:
            error_label = Label(
                self.root, text=f"Error: {str(e)}", fg="red", font=("Arial", 10)
            )
            error_label.pack()
            self.result_labels.append(error_label)

    def display_results(
        self, pupil_area: float, cat_area: float, cataract_percentage: float
    ):
        results_frame = Frame(self.root)
        results_frame.pack(pady=10)

        pupil_label = Label(
            results_frame,
            text=f"Pupil area: {pupil_area:.0f} pixels",
            font=("Arial", 10, "bold"),
        )
        pupil_label.pack()
        self.result_labels.append(pupil_label)

        cat_label = Label(
            results_frame,
            text=f"Cataract area: {cat_area:.0f} pixels",
            font=("Arial", 10, "bold"),
        )
        cat_label.pack()
        self.result_labels.append(cat_label)

        color = (
            "green"
            if cataract_percentage < 10
            else "orange"
            if cataract_percentage < 30
            else "red"
        )
        percentage_label = Label(
            results_frame,
            text=f"Cataract percentage: {cataract_percentage:.2f}%",
            font=("Arial", 12, "bold"),
            fg=color,
        )
        percentage_label.pack()
        self.result_labels.append(percentage_label)
        self.result_labels.append(results_frame)

    def display_images(
        self,
        img_rgb: MatLike,
        imgfiltered: MatLike,
        thresh_image: MatLike,
        img_morpho_copy: MatLike,
        imgg_inv: MatLike,
        cimg_cat: MatLike,
    ):
        # Convert to 3-channel images for display
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

        # Convert arrays to PIL Images
        pil_images = [Image.fromarray(img) for img in images]

        # Resize images
        width, height = 120, 120
        resized_images: list[ImageTk.PhotoImage] = []
        for img in pil_images:
            resized_img = img.resize((width, height), Image.Resampling.LANCZOS)
            resized_images.append(ImageTk.PhotoImage(resized_img))

        self.image_references.extend(resized_images)

        # Create images frame
        images_frame = Frame(self.root)
        images_frame.pack(pady=10)
        self.result_labels.append(images_frame)

        labels = [
            "Original",
            "Filtered",
            "Threshold",
            "Mask",
            "Inverted",
            "Cataract Detection",
        ]

        for tk_image, label_text in zip(resized_images, labels):
            panel_frame = Frame(images_frame)
            panel_frame.pack(side="left", padx=5, pady=5)

            panel = Label(panel_frame, image=tk_image)
            panel.pack()

            text_label = Label(panel_frame, text=label_text, font=("Arial", 8))
            text_label.pack()

            self.result_labels.extend([panel_frame, panel, text_label])

    def run(self):
        self.root.mainloop()


def main():
    app = CataractDetector()
    app.run()


if __name__ == "__main__":
    main()
