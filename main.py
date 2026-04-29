import pyaudio
import numpy as np
import scipy.signal as signal
import threading
import tkinter as tk
from tkinter import messagebox
from pydub import AudioSegment
import wave
import os
import subprocess
from datetime import datetime

# ── Settings ───────────────────────────────────────────────────────────────────
CHUNK = 8192
RATE = 44100
CHANNELS = 1
FORMAT = pyaudio.paFloat32
INPUT_DEVICE = 3

VOICES_FOLDER = 'voice_samples'
OUTPUT_FOLDER = 'cloned_output'
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ── Voice Definitions ──────────────────────────────────────────────────────────
MALE_VOICES = {
    'Male Voice 1 — Deep': ('Male_Voice_01_Deep.wav', -2),
    'Male Voice 2 — Neutral': ('Male_Voice_02_Neutral.wav', 0),
    'Male Voice 3 — Soft': ('Male_Voice_03_Soft.wav', -1),
    'Male Voice 4 — Strong': ('Male_Voice_04_Strong.wav', -3),
    'Male Voice 5 — Young': ('Male_Voice_05_Young.wav', 1),
}

FEMALE_VOICES = {
    'Female Voice 1 — Soft': ('Female_Voice_01_Soft.wav', 2),
    'Female Voice 2 — Neutral':('Female_Voice_02_Neutral.wav', 0),
    'Female Voice 3 — High': ('Female_Voice_03_High.wav', 3),
    'Female Voice 4 — Warm': ('Female_Voice_04_Warm.wav', 1),
    'Female Voice 5 — Clear': ('Female_Voice_05_Clear.wav', 2),
}

ALL_VOICES = {**MALE_VOICES, **FEMALE_VOICES}

# ── Audio Processing ───────────────────────────────────────────────────────────
def pitch_shift(data, semitones):
    if semitones == 0:
        return data
    factor = 2 ** (semitones / 12.0)
    resampled = signal.resample(data, int(len(data) / factor))
    if len(resampled) > len(data):
        return resampled[:len(data)]
    return np.pad(resampled, (0, len(data) - len(resampled)))

def normalize(data):
    max_val = np.max(np.abs(data))
    if max_val > 0:
        return data / max_val * 0.90
    return data

def apply_effect(data, semitones):
    shifted = pitch_shift(data, semitones)
    return normalize(shifted).astype(np.float32)

# ── Recorder ───────────────────────────────────────────────────────────────────
class VoiceRecorder:
    def __init__(self):
        self.audio = pyaudio.PyAudio()
        self.running = False
        self.thread = None
        self.lock = threading.Lock()
        self.recorded_frames = []
        self.current_semitones = 0
        self.status_callback = None

    def set_status_callback(self, cb):
        self.status_callback = cb

    def update_status(self, msg, color='#9ca3af'):
        if self.status_callback:
            self.status_callback(msg, color)

    def set_semitones(self, semitones):
        with self.lock:
            self.current_semitones = semitones

    def start_recording(self):
        if not self.running:
            self.recorded_frames = []
            self.running = True
            self.thread = threading.Thread(target=self.process, daemon=True)
            self.thread.start()
            self.update_status('🔴 Recording... Speak now!', '#ef4444')

    def stop_recording(self, voice_name):
        self.running = False
        self.update_status('💾 Saving...', '#f59e0b')
        # Wait for thread to finish
        if self.thread:
            self.thread.join(timeout=3)
        self.save_recording(voice_name)

    def save_recording(self, voice_name):
        if not self.recorded_frames:
            self.update_status('❌ No audio recorded!', '#ef4444')
            return

        audio_data = np.concatenate(self.recorded_frames)
        audio_int16 = (audio_data * 32767).astype(np.int16)

        timestamp = datetime.now().strftime('%H%M%S')
        safe_name = voice_name.replace(' ', '_').replace('—', '').strip('_')
        filename = f'{OUTPUT_FOLDER}/{safe_name}_{timestamp}.wav'

        with wave.open(filename, 'w') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)
            wf.setframerate(RATE)
            wf.writeframes(audio_int16.tobytes())

        self.update_status(f'✅ Saved: {filename}', '#34d399')
        print(f'Saved: {filename}')

    def process(self):
        try:
            stream_in = self.audio.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                input_device_index=INPUT_DEVICE,
                frames_per_buffer=CHUNK
            )
        except Exception as e:
            self.update_status(f'❌ Mic error: {e}', '#ef4444')
            return

        while self.running:
            try:
                raw = stream_in.read(CHUNK, exception_on_overflow=False)
                data = np.frombuffer(raw, dtype=np.float32).copy()
                with self.lock:
                    semitones = self.current_semitones
                output = apply_effect(data, semitones)
                self.recorded_frames.append(output)
            except Exception as e:
                print(f'Error: {e}')
                continue

        stream_in.stop_stream()
        stream_in.close()

# ── Main App ───────────────────────────────────────────────────────────────────
class App:
    def __init__(self, root):
        self.root = root
        self.root.title('🎙️ Voice Changer')
        self.root.geometry('480x780')
        self.root.configure(bg='#1a1a2e')
        self.root.resizable(False, False)

        self.recorder = VoiceRecorder()
        self.recorder.set_status_callback(self.update_status)

        self.selected_voice = None
        self.selected_file = None
        self.selected_semitones = 0
        self.is_recording = False

        self.build_ui()

    def build_ui(self):
        # ── Title ──
        tk.Label(
            self.root, text='🎙️ Voice Changer',
            font=('Arial', 24, 'bold'),
            bg='#1a1a2e', fg='#a78bfa'
        ).pack(pady=12)

        tk.Label(
            self.root,
            text='Select a voice → Preview → Record & Save',
            font=('Arial', 10),
            bg='#1a1a2e', fg='#6b7280'
        ).pack()

        # ── Male Voices ──
        tk.Label(
            self.root, text='👨 Male Voices',
            font=('Arial', 13, 'bold'),
            bg='#1a1a2e', fg='#60a5fa'
        ).pack(pady=(14, 3))

        male_frame = tk.Frame(self.root, bg='#1a1a2e')
        male_frame.pack()

        self.voice_buttons = {}
        for name, (file, semitones) in MALE_VOICES.items():
            btn = tk.Button(
                male_frame, text=name,
                font=('Arial', 11),
                bg='#1e3a5f', fg='white',
                activebackground='#2563eb',
                relief='flat', width=30, pady=5,
                command=lambda n=name, f=file, s=semitones: self.select_voice(n, f, s)
            )
            btn.pack(pady=2)
            self.voice_buttons[name] = btn

        # ── Female Voices ──
        tk.Label(
            self.root, text='👩 Female Voices',
            font=('Arial', 13, 'bold'),
            bg='#1a1a2e', fg='#f472b6'
        ).pack(pady=(14, 3))

        female_frame = tk.Frame(self.root, bg='#1a1a2e')
        female_frame.pack()

        for name, (file, semitones) in FEMALE_VOICES.items():
            btn = tk.Button(
                female_frame, text=name,
                font=('Arial', 11),
                bg='#4a1942', fg='white',
                activebackground='#db2777',
                relief='flat', width=30, pady=5,
                command=lambda n=name, f=file, s=semitones: self.select_voice(n, f, s)
            )
            btn.pack(pady=2)
            self.voice_buttons[name] = btn

        # ── Status ──
        self.status_label = tk.Label(
            self.root, text='⚪ Select a voice to begin',
            font=('Arial', 11),
            bg='#1a1a2e', fg='#9ca3af'
        )
        self.status_label.pack(pady=10)

        # ── Buttons ──
        btn_frame = tk.Frame(self.root, bg='#1a1a2e')
        btn_frame.pack(pady=5)

        # Preview button
        tk.Button(
            btn_frame, text='▶ Preview',
            font=('Arial', 12, 'bold'),
            bg='#065f46', fg='white',
            activebackground='#047857',
            relief='flat', width=10, pady=8,
            command=self.preview_voice
        ).grid(row=0, column=0, padx=6)

        # Record button
        self.record_btn = tk.Button(
            btn_frame, text='⏺ Record',
            font=('Arial', 12, 'bold'),
            bg='#ef4444', fg='white',
            activebackground='#dc2626',
            relief='flat', width=10, pady=8,
            command=self.toggle_record
        )
        self.record_btn.grid(row=0, column=1, padx=6)

        # Save button
        tk.Button(
            btn_frame, text='⏹ Stop & Save',
            font=('Arial', 12, 'bold'),
            bg='#7c3aed', fg='white',
            activebackground='#6d28d9',
            relief='flat', width=12, pady=8,
            command=self.stop_recording
        ).grid(row=0, column=2, padx=6)

        tk.Label(
            self.root,
            text='💡 Saved in cloned_output/ folder',
            font=('Arial', 10),
            bg='#1a1a2e', fg='#f59e0b'
        ).pack(pady=8)

    def highlight_button(self, selected):
        for name, btn in self.voice_buttons.items():
            if name == selected:
                btn.configure(bg='#7c3aed')
            elif name in MALE_VOICES:
                btn.configure(bg='#1e3a5f')
            else:
                btn.configure(bg='#4a1942')

    def select_voice(self, name, file, semitones):
        self.selected_voice = name
        self.selected_file = file
        self.selected_semitones = semitones
        self.recorder.set_semitones(semitones)
        self.highlight_button(name)
        self.update_status(f'✅ Selected: {name}', '#34d399')

    def preview_voice(self):
        if not self.selected_voice:
            messagebox.showwarning('No Voice', 'Please select a voice first!')
            return
        filepath = os.path.join(VOICES_FOLDER, self.selected_file)
        if not os.path.exists(filepath):
            messagebox.showerror('Error', f'File not found: {filepath}')
            return
        self.update_status(f'▶ Playing: {self.selected_voice}', '#60a5fa')
        # Play in background thread so UI doesn't freeze
        threading.Thread(
            target=lambda: subprocess.run(['aplay', filepath],
            capture_output=True),
            daemon=True
        ).start()

    def toggle_record(self):
        if not self.selected_voice:
            messagebox.showwarning('No Voice', 'Please select a voice first!')
            return
        if not self.is_recording:
            self.is_recording = True
            self.record_btn.configure(bg='#991b1b', text='⏺ Recording')
            self.recorder.start_recording()
        else:
            messagebox.showinfo('Info', 'Click Stop & Save to finish recording.')

    def stop_recording(self):
        if not self.is_recording:
            messagebox.showwarning('Not Recording', 'Please start recording first!')
            return
        self.is_recording = False
        self.record_btn.configure(bg='#ef4444', text='⏺ Record')
        threading.Thread(
            target=lambda: self.recorder.stop_recording(self.selected_voice),
            daemon=True
        ).start()

    def update_status(self, msg, color='#9ca3af'):
        self.status_label.configure(text=msg, fg=color)

if __name__ == '__main__':
    root = tk.Tk()
    app = App(root)
    root.mainloop()



