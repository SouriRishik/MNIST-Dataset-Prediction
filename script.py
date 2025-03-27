import tkinter as tk
from tkinter import Canvas
import numpy as np
import cv2
from PIL import Image, ImageTk
import tensorflow as tf
from keras.models import load_model

model = load_model("mnist_digit_recognizer11.h5")

def preprocess_image(image):
    img = cv2.resize(image, (28, 28))
    img = img.astype('float32') / 255.0
    img = img.reshape(1, 784) 
    return img

class DigitRecognizer:
    def __init__(self, root):
        self.root = root
        self.root.title("Handwritten Digit Recognizer")
        self.root.geometry("400x500")
        self.root.configure(bg='#2C2F33')
        
        self.title_label = tk.Label(root, text="Draw a Digit", font=('Arial', 16, 'bold'), fg='white', bg='#2C2F33')
        self.title_label.pack(pady=10)
        
        self.canvas_frame = tk.Frame(root, bg='#2C2F33')
        self.canvas_frame.pack()
        
        self.canvas = Canvas(self.canvas_frame, width=300, height=300, bg='black', relief='ridge', bd=5)
        self.canvas.pack()
        
        self.button_frame = tk.Frame(root, bg='#2C2F33')
        self.button_frame.pack(pady=10)
        
        self.clear_btn = tk.Button(self.button_frame, text='Clear', command=self.clear_canvas, font=('Arial', 12, 'bold'), fg='white', bg='#7289DA', relief='ridge', width=10)
        self.clear_btn.pack(side=tk.LEFT, padx=10)
        
        self.predict_btn = tk.Button(self.button_frame, text='Predict', command=self.predict_digit, font=('Arial', 12, 'bold'), fg='white', bg='#43B581', relief='ridge', width=10)
        self.predict_btn.pack(side=tk.LEFT, padx=10)
        
        self.canvas.bind("<B1-Motion>", self.paint)
        
        self.image = np.zeros((300, 300), dtype=np.uint8)
        
        self.label = tk.Label(root, text='Prediction: ', font=('Arial', 14, 'bold'), fg='white', bg='#2C2F33')
        self.label.pack(pady=10)
    
    def paint(self, event):
        x, y = event.x, event.y
        self.canvas.create_oval(x, y, x+10, y+10, fill='white', outline='white')
        cv2.circle(self.image, (x, y), 10, 255, -1)
    
    def clear_canvas(self):
        self.canvas.delete("all")
        self.image.fill(0)
    
    def predict_digit(self):
        img = preprocess_image(self.image)
        predictions = model.predict(img)[0]
        top2 = np.argsort(predictions)[-2:][::-1]
        
        pred1, pred2 = top2
        self.label.config(text=f'Prediction: 1st Guess = {pred1}, 2nd Guess = {pred2}')

if __name__ == "__main__":
    root = tk.Tk()
    app = DigitRecognizer(root)
    root.mainloop()
