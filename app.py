"""Windows / VS Code desktop GUI for the Arduino-connected research prototype.

Run: python app.py --model models/benign_malignant_model_fine_knn.pkl
No pretrained weights are downloaded: provide local .keras and .pkl files.
"""
from __future__ import annotations

import argparse
import queue
import sys
import threading
import time
from collections import deque
from pathlib import Path

import cv2
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk, messagebox

from skin_screening.core import SkinLesionModel, crop_roi
from skin_screening.serial_interface import ArduinoLink, available_ports


class SkinScreeningApp:
    def __init__(self, root: tk.Tk, args):
        self.root, self.args = root, args
        self.root.title('AI-Assisted Skin Lesion Screening | Research Prototype')
        self.root.geometry('1100x760')
        self.camera = None
        self.serial = None
        self.model = None
        self.latest_frame = None
        self.events: queue.Queue = queue.Queue()
        self.capture_frames = []
        self.capture_pending = False
        self.inference_busy = False
        self.closing = False
        self.image_ref = None

        self.state_text = tk.StringVar(value='Starting local model loader ...')
        self.prediction_text = tk.StringVar(value='READY — No image classified')
        self.probability_text = tk.StringVar(value='The reported score is not a clinical probability.')
        self.port_choice = tk.StringVar()
        self.camera_choice = tk.StringVar(value=str(args.camera))

        style = ttk.Style()
        style.theme_use('clam')
        frame = ttk.Frame(root, padding=12)
        frame.pack(fill='both', expand=True)
        ttk.Label(frame, text='SKIN LESION SCREENING — RESEARCH DEMONSTRATOR',
                  font=('Segoe UI', 18, 'bold')).pack(anchor='w', pady=(0, 6))
        ttk.Label(frame, text='Not for diagnosis or clinical triage. A clinician must evaluate suspicious lesions.',
                  foreground='#8b1e1e', font=('Segoe UI', 11)).pack(anchor='w', pady=(0, 10))

        controls = ttk.Frame(frame)
        controls.pack(fill='x', pady=(0, 8))
        ttk.Label(controls, text='Camera index:').pack(side='left')
        ttk.Entry(controls, textvariable=self.camera_choice, width=5).pack(side='left', padx=(4, 10))
        ttk.Button(controls, text='Open camera', command=self.open_camera).pack(side='left', padx=3)
        ttk.Button(controls, text='Capture 10 frames', command=self.start_capture).pack(side='left', padx=3)
        ttk.Button(controls, text='Reset (blue)', command=self.reset).pack(side='left', padx=3)
        ttk.Label(controls, text='Arduino COM:').pack(side='left', padx=(18, 4))
        self.port_combo = ttk.Combobox(controls, textvariable=self.port_choice, width=11, state='readonly')
        self.port_combo.pack(side='left')
        ttk.Button(controls, text='Refresh', command=self.refresh_ports).pack(side='left', padx=3)
        ttk.Button(controls, text='Connect', command=self.connect_serial).pack(side='left', padx=3)

        self.video_label = tk.Label(frame, bg='#121820', relief='sunken', bd=1)
        self.video_label.pack(fill='both', expand=True, pady=(6, 8))
        ttk.Label(frame, textvariable=self.prediction_text,
                  font=('Segoe UI', 18, 'bold')).pack(anchor='w', pady=(4, 2))
        ttk.Label(frame, textvariable=self.probability_text,
                  font=('Segoe UI', 10)).pack(anchor='w')
        ttk.Label(frame, textvariable=self.state_text,
                  font=('Segoe UI', 10), foreground='#174d6a').pack(anchor='w', pady=(6, 0))

        self.refresh_ports()
        self.root.protocol('WM_DELETE_WINDOW', self.close)
        self.root.after(40, self.poll_events)
        self.root.after(40, self.update_camera)
        threading.Thread(target=self.load_model_worker, daemon=True).start()
        self.open_camera()

    def load_model_worker(self):
        try:
            model = SkinLesionModel(self.args.extractor, self.args.model,
                                    self.args.metadata, self.args.threshold)
            self.events.put(('model_loaded', model))
        except Exception as exc:
            self.events.put(('error', f'Failed to load local ML model: {exc}'))

    def refresh_ports(self):
        ports = available_ports()
        self.port_combo['values'] = ports
        if ports and self.port_choice.get() not in ports:
            self.port_choice.set(ports[0])

    def connect_serial(self):
        port = self.port_choice.get().strip()
        if not port:
            messagebox.showwarning('Arduino port', 'Select an Arduino COM port first.')
            return
        if self.serial:
            self.serial.close()
            self.serial = None
        try:
            self.serial = ArduinoLink(port)
            self.state_text.set(f'Connected to {port} at 115200 baud')
            threading.Thread(target=self.serial_reader, args=(self.serial,), daemon=True).start()
        except Exception as exc:
            messagebox.showerror('Serial connection', str(exc))

    def serial_reader(self, link: ArduinoLink):
        while not self.closing and self.serial is link:
            try:
                line = link.read_line()
                if line:
                    self.events.put(('serial', line))
            except Exception as exc:
                self.events.put(('error', f'Serial receive error: {exc}'))
                break

    def open_camera(self):
        try:
            index = int(self.camera_choice.get())
        except ValueError:
            messagebox.showerror('Camera index', 'Camera index must be an integer.')
            return
        if self.camera is not None:
            self.camera.release()
        backend = cv2.CAP_DSHOW if sys.platform.startswith('win') else cv2.CAP_ANY
        self.camera = cv2.VideoCapture(index, backend)
        if not self.camera.isOpened():
            self.camera.release()
            self.camera = None
            self.state_text.set(f'Camera {index} unavailable; try index 0 or 1.')
            return
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.state_text.set(f'Camera {index} active. Press Arduino button or Capture.')

    def update_camera(self):
        if self.closing:
            return
        if self.camera is not None:
            ok, frame = self.camera.read()
            if ok:
                self.latest_frame = frame
                if self.capture_pending:
                    self.capture_frames.append(crop_roi(frame, self.args.roi))
                    if len(self.capture_frames) >= self.args.frames:
                        self.capture_pending = False
                        frames = self.capture_frames
                        self.capture_frames = []
                        threading.Thread(target=self.infer_worker, args=(frames,), daemon=True).start()
                shown = frame.copy()
                h, w = shown.shape[:2]
                x1, y1, x2, y2 = self.args.roi
                cv2.rectangle(shown, (max(0,x1), max(0,y1)),
                              (min(w-1,x2), min(h-1,y2)), (0, 230, 230), 2)
                cv2.putText(shown, 'ROI | Press button to capture', (15, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                # Display smaller than source if necessary; preserve aspect ratio.
                display = cv2.cvtColor(shown, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(display)
                image.thumbnail((1050, 540))
                self.image_ref = ImageTk.PhotoImage(image)
                self.video_label.configure(image=self.image_ref)
        self.root.after(40, self.update_camera)

    def start_capture(self):
        if self.model is None:
            self.state_text.set('Model still loading or unavailable; no classification performed.')
            self.send_arduino('UNCERTAIN')
            return
        if self.camera is None:
            self.state_text.set('Please open a webcam first.')
            self.send_arduino('UNCERTAIN')
            return
        if self.capture_pending or self.inference_busy:
            self.state_text.set('Please finish the current capture first.')
            return
        self.inference_busy = True
        self.capture_frames = []
        self.capture_pending = True
        self.prediction_text.set('CAPTURING...')
        self.state_text.set(f'Collecting {self.args.frames} new webcam frames after the button event.')

    def infer_worker(self, frames):
        try:
            result = self.model.predict_frames(frames)
            self.events.put(('prediction', result))
        except Exception as exc:
            self.events.put(('error', f'Inference failed: {exc}'))
            self.events.put(('inference_failed', None))

    def send_arduino(self, message: str):
        if self.serial:
            try:
                self.serial.send(message)
            except Exception as exc:
                self.state_text.set(f'Arduino send error: {exc}')

    def reset(self):
        self.send_arduino('RESET')
        self.prediction_text.set('READY — Awaiting another capture')
        self.probability_text.set('A displayed score is not a validated medical risk estimate.')

    def poll_events(self):
        if self.closing:
            return
        try:
            while True:
                kind, data = self.events.get_nowait()
                if kind == 'model_loaded':
                    self.model = data
                    self.state_text.set('Local MobileNetV2 + classifier loaded; no downloads performed.')
                elif kind == 'serial':
                    if data == 'CAPTURE':
                        self.start_capture()
                    elif data == 'ARDUINO_READY':
                        self.state_text.set('Arduino ready — default blue ring.')
                elif kind == 'prediction':
                    self.inference_busy = False
                    self.prediction_text.set(f'RESEARCH CLASSIFIER OUTPUT: {data.label}')
                    self.probability_text.set(
                        f'Average model scores — benign {data.mean_benign_score:.1%}; '
                        f'malignant {data.mean_malignant_score:.1%}; '
                        f'frame agreement {data.stability:.0%} ({data.sample_count} frames). '
                        f'Uncertain threshold {self.args.threshold:.0%}.')
                    self.send_arduino(data.label)
                    self.state_text.set('Inference complete; feedback sent to Arduino.')
                elif kind == 'inference_failed':
                    self.inference_busy = False
                    self.prediction_text.set('UNCERTAIN — Processing error')
                    self.send_arduino('UNCERTAIN')
                elif kind == 'error':
                    self.state_text.set(data)
        except queue.Empty:
            pass
        self.root.after(60, self.poll_events)

    def close(self):
        self.closing = True
        if self.camera:
            self.camera.release()
        if self.serial:
            self.serial.close()
        self.root.destroy()


def parse_args():
    p = argparse.ArgumentParser(description='USB webcam + Arduino skin-lesion research demonstrator')
    p.add_argument('--extractor', default='models/mobilenet_feature_extractor.keras')
    p.add_argument('--model', default='models/benign_malignant_model_fine_knn.pkl')
    p.add_argument('--metadata', default='models/benign_malignant_model_fine_knn_metadata.pkl')
    p.add_argument('--camera', type=int, default=0)
    p.add_argument('--frames', type=int, default=10)
    p.add_argument('--threshold', type=float, default=0.60)
    p.add_argument('--roi', type=int, nargs=4, default=[150, 100, 450, 400],
                   metavar=('X1', 'Y1', 'X2', 'Y2'))
    args = p.parse_args()
    if args.frames < 1 or args.frames > 100:
        p.error('--frames must be between 1 and 100')
    return args


if __name__ == '__main__':
    root = tk.Tk()
    SkinScreeningApp(root, parse_args())
    root.mainloop()
