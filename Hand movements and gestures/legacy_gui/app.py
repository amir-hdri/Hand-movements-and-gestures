import tkinter as tk
from tkinter import ttk, messagebox, Text
import cv2
import numpy as np
from PIL import Image, ImageTk
import threading
import time
import sys
import os
import queue

# Ensure modules can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from legacy_gui.data_manager import DataCollector
from legacy_gui.model_trainer import GestureTrainer
from legacy_gui.inference import GestureInference
from legacy_gui.robot_interface import MockRobotController, PingPongController, PINGPONG_AVAILABLE

class MockVideoCapture:
    """Mock video capture for testing without camera."""
    def __init__(self):
        self.frame_count = 0

    def isOpened(self):
        return True

    def read(self):
        # Create a dummy image with moving text
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        self.frame_count += 1
        cv2.putText(img, f"No Camera - Mock Frame {self.frame_count}", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Simulate some movement to test processing
        x = (self.frame_count * 5) % 640
        cv2.circle(img, (x, 100), 20, (0, 255, 0), -1)

        time.sleep(0.03) # ~30 FPS
        return True, img

    def release(self):
        pass

class GestureApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Gesture Control System")
        self.geometry("1200x800")

        # Style
        self.style = ttk.Style()
        self.style.theme_use('clam')

        # Data & Logic
        self.data_collector = DataCollector()
        self.trainer = GestureTrainer()
        self.inference = GestureInference()
        self.robot = MockRobotController()

        self.cap = None
        self.is_running = True
        self.video_queue = queue.Queue(maxsize=1)
        self.result_queue = queue.Queue(maxsize=1)

        self.current_action_name = tk.StringVar(value="new_action")
        self.robot_status_var = tk.StringVar(value="Disconnected")
        self.inference_result_var = tk.StringVar(value="Waiting...")
        self.robot_mode_var = tk.StringVar(value="Mock")

        # Build UI
        self.create_widgets()

        # Start Video Thread
        self.video_thread = threading.Thread(target=self.video_loop, daemon=True)
        self.video_thread.start()

        # Start UI Update Loop
        self.update_ui()

        # Protocol
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

    def create_widgets(self):
        # Notebook (Tabs)
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Tabs
        self.tab_inference = ttk.Frame(self.notebook)
        self.tab_data = ttk.Frame(self.notebook)
        self.tab_training = ttk.Frame(self.notebook)
        self.tab_robot = ttk.Frame(self.notebook)

        self.notebook.add(self.tab_inference, text="Inference & Control")
        self.notebook.add(self.tab_data, text="Data Collection")
        self.notebook.add(self.tab_training, text="Model Training")
        self.notebook.add(self.tab_robot, text="Robot Manual Control")

        self.setup_inference_tab()
        self.setup_data_tab()
        self.setup_training_tab()
        self.setup_robot_tab()

    def setup_inference_tab(self):
        # Layout: Video on Left, Controls/Status on Right
        frame = self.tab_inference

        # Left: Video
        self.video_label = ttk.Label(frame)
        self.video_label.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Right: Controls
        control_panel = ttk.Frame(frame, width=300)
        control_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)

        # Status
        ttk.Label(control_panel, text="Prediction:", font=("Arial", 14, "bold")).pack(pady=5)
        ttk.Label(control_panel, textvariable=self.inference_result_var, font=("Arial", 16), foreground="blue").pack(pady=5)

        ttk.Separator(control_panel, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)

        # Robot Toggle
        ttk.Label(control_panel, text="Robot Mode:").pack(pady=5)
        modes = ["Mock"]
        if PINGPONG_AVAILABLE:
            modes.append("PingPong")

        self.robot_mode_combo = ttk.Combobox(control_panel, values=modes, textvariable=self.robot_mode_var, state="readonly")
        self.robot_mode_combo.pack(pady=5)
        self.robot_mode_combo.bind("<<ComboboxSelected>>", self.change_robot_mode)

        ttk.Button(control_panel, text="Connect Robot", command=self.connect_robot).pack(pady=5)
        ttk.Label(control_panel, textvariable=self.robot_status_var).pack(pady=5)

    def setup_data_tab(self):
        frame = self.tab_data

        # Top: Controls
        top_panel = ttk.Frame(frame)
        top_panel.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

        ttk.Label(top_panel, text="Action Name:").pack(side=tk.LEFT, padx=5)
        ttk.Entry(top_panel, textvariable=self.current_action_name).pack(side=tk.LEFT, padx=5)

        self.btn_record = ttk.Button(top_panel, text="Start Recording", command=self.toggle_recording)
        self.btn_record.pack(side=tk.LEFT, padx=5)

        # Video
        self.data_video_label = ttk.Label(frame)
        self.data_video_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    def setup_training_tab(self):
        frame = self.tab_training

        btn_train = ttk.Button(frame, text="Train Model", command=self.start_training)
        btn_train.pack(pady=10)

        self.log_text = Text(frame, height=20, state='disabled')
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    def setup_robot_tab(self):
        frame = self.tab_robot

        lbl = ttk.Label(frame, text="Manual Control", font=("Arial", 14, "bold"))
        lbl.pack(pady=10)

        grid_frame = ttk.Frame(frame)
        grid_frame.pack(pady=10)

        # Directional Pad
        ttk.Button(grid_frame, text="Up", command=lambda: self.robot.move("up")).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(grid_frame, text="Left", command=lambda: self.robot.move("left")).grid(row=1, column=0, padx=5, pady=5)
        ttk.Button(grid_frame, text="Down", command=lambda: self.robot.move("down")).grid(row=1, column=1, padx=5, pady=5)
        ttk.Button(grid_frame, text="Right", command=lambda: self.robot.move("right")).grid(row=1, column=2, padx=5, pady=5)

        # Action Buttons
        action_frame = ttk.Frame(frame)
        action_frame.pack(pady=20)

        ttk.Button(action_frame, text="Grab", command=lambda: self.robot.gripper("close")).pack(side=tk.LEFT, padx=10)
        ttk.Button(action_frame, text="Release", command=lambda: self.robot.gripper("open")).pack(side=tk.LEFT, padx=10)
        ttk.Button(action_frame, text="Takeoff", command=lambda: self.robot.execute_action("takeoff")).pack(side=tk.LEFT, padx=10)
        ttk.Button(action_frame, text="Land", command=lambda: self.robot.execute_action("land")).pack(side=tk.LEFT, padx=10)

    def log(self, message):
        self.log_text.config(state='normal')
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state='disabled')

    def change_robot_mode(self, event=None):
        mode = self.robot_mode_var.get()
        if self.robot:
            self.robot.disconnect()

        if mode == "PingPong":
            try:
                self.robot = PingPongController()
            except ImportError:
                messagebox.showerror("Error", "PingPong library not available.")
                self.robot_mode_var.set("Mock")
                self.robot = MockRobotController()
        else:
            self.robot = MockRobotController()

        self.robot_status_var.set("Disconnected")

    def connect_robot(self):
        if self.robot.connect():
            self.robot_status_var.set("Connected")
        else:
            self.robot_status_var.set("Connection Failed")

    def toggle_recording(self):
        if self.data_collector.is_recording:
            count = self.data_collector.stop_recording()
            self.btn_record.config(text="Start Recording")
            messagebox.showinfo("Info", f"Saved {count} frames.")
        else:
            action = self.current_action_name.get()
            if not action:
                messagebox.showwarning("Warning", "Enter action name first.")
                return
            self.data_collector.start_recording(action)
            self.btn_record.config(text="Stop Recording")

    def start_training(self):
        self.log("Starting training...")
        threading.Thread(target=self._run_training, daemon=True).start()

    def _run_training(self):
        try:
            result = self.trainer.train()
            # Update UI from thread
            self.after(0, lambda: self.log(result))
            self.after(0, lambda: messagebox.showinfo("Training", "Training Complete"))
            # Reload inference model
            self.inference.load_model()
        except Exception as e:
            self.after(0, lambda: self.log(f"Error: {e}"))

    def video_loop(self):
        # Try real camera first, then mock
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Cannot open camera, using MockVideoCapture")
            self.cap = MockVideoCapture()

        last_pred_time = 0

        while self.is_running:
            ret, frame = self.cap.read()
            if not ret:
                break

            # 1. Process for Data Collection (Draws landmarks)
            processed_frame, results = self.data_collector.process_frame(frame)

            # 2. Inference (if landmarks detected)
            if results.multi_hand_landmarks:
                action, conf = self.inference.predict(results)

                # Update UI Variable via queue
                if action:
                    try:
                         if self.result_queue.full():
                             self.result_queue.get_nowait()
                         self.result_queue.put(f"{action} ({conf:.2f})")
                    except:
                        pass

                    # Execute Robot Action (Rate limited)
                    if time.time() - last_pred_time > 1.0:
                        self.robot.execute_action(action)
                        last_pred_time = time.time()
                else:
                    pass
            else:
                try:
                     if self.result_queue.full():
                         self.result_queue.get_nowait()
                     self.result_queue.put("No Hand Detected")
                except:
                    pass

            # Convert to RGB for display
            rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb_frame)
            # DO NOT CREATE ImageTk here

            try:
                # Put PIL Image in queue for main thread (dropping old frames if full)
                if self.video_queue.full():
                    try:
                        self.video_queue.get_nowait()
                    except queue.Empty:
                        pass
                self.video_queue.put(img)
            except:
                pass

            # If using mock, self.cap.read already sleeps
            # If real camera, read() blocks for next frame, but sometimes not.
            # Adding small sleep prevents 100% CPU in loop if read() is fast (e.g. video file)
            # but for webcam it's fine.
            # time.sleep(0.001)

    def update_ui(self):
        # Update video
        try:
            img = self.video_queue.get_nowait()

            if not self.is_running:
                return

            # Create ImageTk on Main Thread
            imgtk = ImageTk.PhotoImage(image=img)

            try:
                current_tab_idx = self.notebook.index(self.notebook.select())
            except tk.TclError:
                current_tab_idx = 0

            if current_tab_idx == 0: # Inference
                self.video_label.configure(image=imgtk)
                self.video_label.image = imgtk
            elif current_tab_idx == 1: # Data
                self.data_video_label.configure(image=imgtk)
                self.data_video_label.image = imgtk
        except queue.Empty:
            pass

        # Update result label
        try:
            res = self.result_queue.get_nowait()
            self.inference_result_var.set(res)
        except queue.Empty:
            pass

        if self.is_running:
            self.after(30, self.update_ui)

    def on_closing(self):
        self.is_running = False
        if self.cap:
            self.cap.release()
        self.destroy()

if __name__ == "__main__":
    app = GestureApp()
    app.mainloop()
