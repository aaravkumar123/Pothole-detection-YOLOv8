import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import os
import threading
import numpy as np
import cvzone
from ultralytics import YOLO

class ObjectDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLO Object Detection System")
        self.root.geometry("1200x800")
        self.root.resizable(True, True)
        
        # Load YOLO model
        self.model = YOLO("best.pt")
        self.class_names = self.model.names
        
        # Variables
        self.cap = None
        self.video_path = ""
        self.image_path = ""
        self.detection_running = False
        self.current_frame = None
        self.selected_option = tk.StringVar(value="webcam")
        self.frame_skip = tk.IntVar(value=3)  # Frame skip counter
        
        # Create GUI
        self.create_widgets()
        
    def create_widgets(self):
        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel (controls)
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding=10)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        # Right panel (display)
        display_frame = ttk.LabelFrame(main_frame, text="Detection Output", padding=10)
        display_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Display area
        self.display_label = ttk.Label(display_frame)
        self.display_label.pack(fill=tk.BOTH, expand=True)
        
        # Mode selection
        mode_frame = ttk.LabelFrame(control_frame, text="Detection Mode", padding=10)
        mode_frame.pack(fill=tk.X, pady=5)
        
        ttk.Radiobutton(mode_frame, text="Webcam", variable=self.selected_option, 
                       value="webcam", command=self.mode_changed).pack(anchor=tk.W)
        ttk.Radiobutton(mode_frame, text="Video File", variable=self.selected_option, 
                       value="video", command=self.mode_changed).pack(anchor=tk.W)
        ttk.Radiobutton(mode_frame, text="Image File", variable=self.selected_option, 
                       value="image", command=self.mode_changed).pack(anchor=tk.W)
        
        # Settings frame
        settings_frame = ttk.LabelFrame(control_frame, text="Detection Settings", padding=10)
        settings_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(settings_frame, text="Frame Skip:").pack(anchor=tk.W)
        ttk.Entry(settings_frame, textvariable=self.frame_skip, width=5).pack(anchor=tk.W)
        
        # File selection (hidden initially)
        self.file_frame = ttk.Frame(control_frame)
        
        self.video_btn = ttk.Button(self.file_frame, text="Select Video", 
                                   command=self.select_video)
        self.video_btn.pack(fill=tk.X, pady=5)
        
        self.image_btn = ttk.Button(self.file_frame, text="Select Image", 
                                  command=self.select_image)
        self.image_btn.pack(fill=tk.X, pady=5)
        
        # Control buttons
        btn_frame = ttk.Frame(control_frame)
        btn_frame.pack(fill=tk.X, pady=10)
        
        self.start_btn = ttk.Button(btn_frame, text="Start Detection", 
                                  command=self.start_detection)
        self.start_btn.pack(fill=tk.X, pady=5)
        
        self.stop_btn = ttk.Button(btn_frame, text="Stop Detection", 
                                 command=self.stop_detection, state=tk.DISABLED)
        self.stop_btn.pack(fill=tk.X, pady=5)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = ttk.Label(control_frame, textvariable=self.status_var, 
                              relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(fill=tk.X, pady=5)
        
        # Initialize mode
        self.mode_changed()
        
    def mode_changed(self):
        mode = self.selected_option.get()
        
        if mode == "webcam":
            self.file_frame.pack_forget()
            self.video_path = ""
            self.image_path = ""
        elif mode == "video":
            self.file_frame.pack(fill=tk.X, pady=5)
            self.video_btn.pack(fill=tk.X, pady=5)
            self.image_btn.pack_forget()
            self.image_path = ""
        elif mode == "image":
            self.file_frame.pack(fill=tk.X, pady=5)
            self.image_btn.pack(fill=tk.X, pady=5)
            self.video_btn.pack_forget()
            self.video_path = ""
            
    def select_video(self):
        file_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[("Video Files", "*.mp4 *.avi *.mov"), ("All Files", "*.*")]
        )
        if file_path:
            self.video_path = file_path
            self.status_var.set(f"Selected: {os.path.basename(file_path)}")
            
    def select_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Image File",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp"), ("All Files", "*.*")]
        )
        if file_path:
            self.image_path = file_path
            self.status_var.set(f"Selected: {os.path.basename(file_path)}")
            
    def start_detection(self):
        mode = self.selected_option.get()
        
        if mode == "video" and not self.video_path:
            messagebox.showerror("Error", "Please select a video file first")
            return
        elif mode == "image" and not self.image_path:
            messagebox.showerror("Error", "Please select an image file first")
            return
            
        self.detection_running = True
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        
        if mode == "webcam":
            self.cap = cv2.VideoCapture(0)
            self.status_var.set("Starting webcam...")
        elif mode == "video":
            self.cap = cv2.VideoCapture(self.video_path)
            self.status_var.set(f"Processing video: {os.path.basename(self.video_path)}")
        elif mode == "image":
            self.status_var.set(f"Processing image: {os.path.basename(self.image_path)}")
            
        # Start detection in a separate thread
        detection_thread = threading.Thread(target=self.run_detection)
        detection_thread.daemon = True
        detection_thread.start()
        
    def stop_detection(self):
        self.detection_running = False
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.status_var.set("Stopped")
        
    def run_detection(self):
        mode = self.selected_option.get()
        count = 0
        
        if mode == "image":
            self.process_image()
            return
            
        while self.detection_running and self.cap.isOpened():
            ret, img = self.cap.read()
            
            if not ret:
                if mode == "video":
                    # Video ended
                    self.status_var.set("Video processing completed")
                    break
                else:
                    # Webcam error
                    self.status_var.set("Error reading from webcam")
                    break
            
            count += 1
            if count % self.frame_skip.get() != 0:
                continue
                
            # Resize image
            img = cv2.resize(img, (1020, 500))
            h, w, _ = img.shape
            
            # Perform YOLO detection
            results = self.model.predict(img)
            
            # Process results
            for r in results:
                boxes = r.boxes
                masks = r.masks
                
                if masks is not None:
                    masks = masks.data.cpu()
                    for seg, box in zip(masks.data.cpu().numpy(), boxes):
                        seg = cv2.resize(seg, (w, h))
                        contours, _ = cv2.findContours((seg).astype(np.uint8), 
                                                     cv2.RETR_EXTERNAL, 
                                                     cv2.CHAIN_APPROX_SIMPLE)

                        for contour in contours:
                            d = int(box.cls)
                            c = self.class_names[d]
                            x, y, x1, y1 = cv2.boundingRect(contour)
                            cv2.polylines(img, [contour], True, color=(0, 0, 255), thickness=2)
                            cv2.putText(img, c, (x, y - 10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                                       (255, 255, 255), 2)
            
            # Display the frame
            self.update_display(img)
            
            # Slow down the loop for webcam
            if mode == "webcam":
                key = cv2.waitKey(1)
                if key == 27:  # ESC key
                    break
                    
        # Clean up
        if self.cap is not None:
            self.cap.release()
        self.detection_running = False
        self.root.after(0, lambda: self.stop_btn.config(state=tk.DISABLED))
        self.root.after(0, lambda: self.start_btn.config(state=tk.NORMAL))
        
    def process_image(self):
        if not self.image_path:
            return
            
        img = cv2.imread(self.image_path)
        if img is None:
            self.status_var.set("Error loading image")
            return
            
        # Resize image
        img = cv2.resize(img, (1020, 500))
        h, w, _ = img.shape
        
        # Perform YOLO detection
        results = self.model.predict(img)
        
        # Process results
        for r in results:
            boxes = r.boxes
            masks = r.masks
            
            if masks is not None:
                masks = masks.data.cpu()
                for seg, box in zip(masks.data.cpu().numpy(), boxes):
                    seg = cv2.resize(seg, (w, h))
                    contours, _ = cv2.findContours((seg).astype(np.uint8), 
                                                 cv2.RETR_EXTERNAL, 
                                                 cv2.CHAIN_APPROX_SIMPLE)

                    for contour in contours:
                        d = int(box.cls)
                        c = self.class_names[d]
                        x, y, x1, y1 = cv2.boundingRect(contour)
                        cv2.polylines(img, [contour], True, color=(0, 0, 255), thickness=2)
                        cv2.putText(img, c, (x, y - 10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                                   (255, 255, 255), 2)
        
        # Display the image
        self.update_display(img)
        self.detection_running = False
        self.root.after(0, lambda: self.stop_btn.config(state=tk.DISABLED))
        self.root.after(0, lambda: self.start_btn.config(state=tk.NORMAL))
        
    def update_display(self, frame):
        # Convert the frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(frame_rgb)
        
        # Convert to ImageTk format
        tk_image = ImageTk.PhotoImage(image=pil_image)
        
        # Update the display
        self.display_label.config(image=tk_image)
        self.display_label.image = tk_image
        
    def on_closing(self):
        self.detection_running = False
        if self.cap is not None:
            self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = ObjectDetectionApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
