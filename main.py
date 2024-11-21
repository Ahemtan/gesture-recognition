import cv2
import HandDetectorModule
from ClassificationModule import Classifier
import numpy as np
import math
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk

class HandGestureApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Hand Gesture Recognition")
        self.root.geometry("1200x700")

        self.canvas = tk.Canvas(self.root, width=800, height=600)
        self.canvas.pack()

        self.log_text = tk.Text(self.root, height=5, width=80)
        self.log_text.pack()
        self.log_text.insert(tk.END, "Starting Hand Gesture Recognition...\n")  # Initial log message
        self.log_text.config(state=tk.DISABLED)

        self.detector = HandDetectorModule.HandDetector(maxHands=1)
        self.classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

        self.labels = ["A", "B", "C"]

        self.initialize_camera()

        self.update_frame()

    def initialize_camera(self):
        for i in range(3):
            self.cap = cv2.VideoCapture(i)
            if self.cap.isOpened():
                print(f"Successfully opened camera at index {i}.")
                return
        messagebox.showerror("Error", "Unable to access any camera.")
        self.root.quit()

    def update_frame(self):
        success, img = self.cap.read()
        if not success:
            messagebox.showerror("Error", "Unable to access the webcam.")
            self.root.quit()
            return

        img, log_message = self.process_frame(img)

        self.update_log(log_message)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img)
        img_tk = ImageTk.PhotoImage(img_pil)

        self.photo_image = img_tk

        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo_image)

        self.root.after(10, self.update_frame)

    def process_frame(self, img):
        hands, img = self.detector.findHands(img)
        img_output = img.copy()
        log_message = ""

        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']

            offset = 20
            img_size = 300
            x = max(0, x - offset)
            y = max(0, y - offset)
            w = min(img.shape[1] - x, w + 2 * offset)
            h = min(img.shape[0] - y, h + 2 * offset)

            img_crop = img[y:y + h, x:x + w]

            if img_crop.size != 0:
                img_white = np.ones((img_size, img_size, 3), np.uint8) * 255
                aspect_ratio = h / w

                if aspect_ratio > 1:
                    k = img_size / h
                    wcal = math.ceil(k * w)
                    img_resize = cv2.resize(img_crop, (wcal, img_size))
                    w_gap = math.ceil((img_size - wcal) / 2)
                    img_white[:, w_gap:wcal + w_gap] = img_resize

                    prediction, index = self.classifier.getPrediction(img_white)
                else:
                    k = img_size / w
                    hcal = math.ceil(k * h)
                    img_resize = cv2.resize(img_crop, (img_size, hcal))
                    h_gap = math.ceil((img_size - hcal) / 2)
                    img_white[h_gap:hcal + h_gap, :] = img_resize
                    prediction, index = self.classifier.getPrediction(img_white)

                cv2.rectangle(img_output, (x - offset, y - offset - 50),
                              (x - offset + 150, y - offset - 50 + 50), (0, 0, 0), cv2.FILLED)
                cv2.putText(img_output, self.labels[index], (x, y - 26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
                cv2.rectangle(img_output, (x - offset, y - offset),
                              (x + w + offset, y + h + offset), (0, 0, 0), 4)

                log_message = f"Prediction: {self.labels[index]} | Time: {prediction[1]}ms\n"
            else:
                log_message = "Empty crop, skipping frame"

        return img_output, log_message

    def update_log(self, log_message):
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, log_message)
        self.log_text.yview(tk.END)
        self.log_text.config(state=tk.DISABLED)

    def stop(self):
        if self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
        self.root.quit()


def main():
    root = tk.Tk()

    app = HandGestureApp(root)
    root.resizable(False, False)

    root.protocol("WM_DELETE_WINDOW", app.stop)

    root.mainloop()


if __name__ == "__main__":
    main()
