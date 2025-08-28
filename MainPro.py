import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
import scipy.ndimage
import seaborn as sns
from tensorflow.keras.models import load_model, Model
import tensorflow as tf

import subprocess
# Global settings
im_size = 224

# Load the model
from tensorflow.keras.models import load_model
model_path = 'saved_model/custom_model'
cam_model = load_model(model_path)
intermediate_model = Model(inputs=cam_model.input, outputs=cam_model.layers[-3].output)

# GUI Setup
class AICanvasApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Forecasting Wildfire Danger with Deep Learning A Location-Aware Adaptive Normalization")
        self.root.geometry("1000x600")
        self.image_path = None

        # Title
        tk.Label(root, text="Forecasting Wildfire Danger with Deep Learning ", font=("Arial", 24)).pack(pady=10)

        # Button Frame
        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=10)

        # Upload Button
        tk.Button(root, text="Upload Image", command=self.upload_image, bg="lightblue").pack(pady=10)

        # Predict Button
        tk.Button(root, text="Predict", command=self.predict_image, bg="green", fg="white").pack(pady=10)

        # Predict Button
        tk.Button(root, text="Logout", command=self.logout, bg="green", fg="white").pack(pady=10)
        # tk.Button(btn_frame, text="Login Page",command=self.logout,  width=15).pack(side=tk.LEFT, padx=5)

        # Image Display
        self.image_label = tk.Label(root)
        self.image_label.pack(pady=10)

    def logout(self):
        self.root.destroy()
        subprocess.call(["python", "LoginPage.py"])

    def upload_image(self):
        filetypes = [("Image Files", "*.jpg *.png *.jpeg")]
        self.image_path = filedialog.askopenfilename(title="Choose an image", filetypes=filetypes)
        if self.image_path:
            img = Image.open(self.image_path)
            img = img.resize((300, 300))
            self.img_tk = ImageTk.PhotoImage(img)
            self.image_label.config(image=self.img_tk)

    def predict_image(self):
        if not self.image_path:
            messagebox.showerror("Error", "Please upload an image first.")
            return

        # Preprocess image
        img = cv2.imread(self.image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (im_size, im_size)) / 255.0
        tensor_image = np.expand_dims(img, axis=0)

        # Get features and predictions
        features = intermediate_model.predict(tensor_image)
        results = cam_model.predict(tensor_image)

        self.show_cam(tensor_image, features, results)

    def show_cam(self, image_value, features, results):
        features_for_img = features[0]
        class_activation_weights = cam_model.layers[-1].get_weights()[0][:, 1]
        class_activation_features = scipy.ndimage.zoom(features_for_img, 
                                                       (im_size / features_for_img.shape[0],
                                                        im_size / features_for_img.shape[1], 1), order=2)
        cam_output = np.dot(class_activation_features, class_activation_weights)

        plt.figure(figsize=(6, 6))
        plt.imshow(cam_output, cmap='jet', alpha=0.5)
        plt.imshow(np.squeeze(image_value), alpha=0.5)
        plt.title('Class Activation Map')
        plt.figtext(.5, .01,
                    f"No Wildfire Probability: {results[0][0] * 100:.2f}%\nWildfire Probability: {results[0][1] * 100:.2f}%",
                    ha="center", fontsize=10,
                    bbox={"facecolor": "green", "alpha": 0.5, "pad": 3})
        plt.colorbar()
        plt.tight_layout()
        plt.show()


# Run App
if __name__ == "__main__":
    root = tk.Tk()
    app = AICanvasApp(root)
    root.mainloop()
