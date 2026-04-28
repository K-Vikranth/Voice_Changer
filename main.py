import pyaudio
import numpy as np
import librosa
import scipy.signal as signal
import threading
import tkinter as tk
import wave
import os
from datetime import datetime

# ── Audio Settings ─────────────────────────────────────────────────────────────
CHUNK = 8192 # Larger chunk = cleaner audio
RATE = 44100
CHANNELS = 1
FORMAT = pyaudio.paFloat32
INPUT_DEVICE = 3

OUTPUT_FOLDER = 'voice_samples'
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ── Voice Effects ──────────────────────────────────────────────────────────────
def pitch_shift(data, semitones):
    """Clean pitch shift using librosa."""
    try:
        shifted = librosa.effects.pitch_shift(
            data.astype(np.float32),
            sr=RATE,
            n_steps=semitones
        )
        return shifted
    except Exception as e:
        print(f'Pitch shift error: {e}')
        return data

def robot_effect(data):
    t = np.arange(len(data)) / RATE
    carrier = np.sin(2 * np.pi * 80 * t)
    modulated = data * carrier
    return np.clip(modulated * 2.0, -1.0, 1.0)

def echo_effect(data, delay_samples=8000, decay=0.4):
    output = data.copy()
    if len(data) > delay_samples:
        output[delay_samples:] += data[:-delay_samples] * decay
    return output

def wobble_effect(data):
    t = np.arange(len(data)) / RATE
    wobble = np.sin(2 * np.pi * 5 * t) * 0.3 + 1.0
    return data * wobble

def growl_effect(data):
    t = np.arange(len(data)) / RATE
    growl = np.sin(2 * np.pi * 40 * t) * 0.3
    return np.clip(data + growl, -1.0, 1.0)

def normalize(data):
    max_val = np.max(np.abs(data))
    if max_val > 0:
        return data / max_val * 0.85
    return data

def apply_voice(data, voice):
    data = data.astype(np.float32)
    try:
        if voice == 'Normal':
            result = data
        elif voice == 'Female':
            result = pitch_shift(data, semitones=4)
        elif voice == 'Male Deep':
            result = pitch_shift(data, semitones=-4)
        elif voice == 'Kid':
            result = pitch_shift(data, semitones=8)
        elif voice == 'Grandpa':
            shifted = pitch_shift(data, semitones=-5)
            result = echo_effect(shifted)
        elif voice == 'Robot 🤖':
            result = robot_effect(data)
        elif voice == 'Ghost 👻':
            shifted = pitch_shift(data, semitones=-3)
            result = echo_effect(shifted, delay_samples=6000, decay=0.5)
        elif voice == 'Alien 👽':
            shifted = pitch_shift(data, semitones=7)
            result = wobble_effect(shifted)
        elif voice == 'Monster 👹':
            shifted = pitch_shift(data, semitones=-8)
            result = growl_effect(shifted)
        elif voice == 'Chipmunk 🐿️':
            result = pitch_shift(data, semitones=10)
        else:
            result = data
    except Exception as e:
        print(f'Effect error: {e}')
        result = data

    return normalize(result).astype(np.float32)

# ── Recorder Engine ────────────────────────────────────────────────────────────
class VoiceRecorder:
    def __init__(self):
        self.audio = pyaudio.PyAudio()
        self.running = False
        self.current_voice = 'Normal'
        self.thread = None
        self.lock = threading.Lock()
        self.recorded_frames = []
        self.status_callback = None

    def set_voice(self, voice):
        with self.lock:
            self.current_voice = voice
        print(f'Voice: {voice}')

    def set_status_callback(self, callback):
        self.status_callback = callback

    def update_status(self, msg, color='#9ca3af'):
        if self.status_callback:
            self.status_callback(msg, color)

    def start_recording(self):
        if not self.running:
            self.recorded_frames = []
            self.running = True
            self.thread = threading.Thread(target=self.process, daemon=True)
            self.thread.start()
            self.update_status('🔴 Recording... Speak now!', '#ef4444')

    def stop_recording(self):
        self.running = False
        self.update_status('💾 Saving...', '#f59e0b')

    def save_recording(self, voice_name):
        if not self.recorded_frames:
            self.update_status('❌ No audio recorded!', '#ef4444')
            return None

        audio_data = np.concatenate(self.recorded_frames)
        audio_int16 = (audio_data * 32767).astype(np.int16)

        timestamp = datetime.now().strftime('%H%M%S')
        safe_name = voice_name.replace(' ', '_')\
            .replace('🤖', 'Robot')\
            .replace('👻', 'Ghost')\
            .replace('👽', 'Alien')\
            .replace('👹', 'Monster')\
            .replace('🐿️', 'Chipmunk')

        filename = f'{OUTPUT_FOLDER}/{safe_name}_{timestamp}.wav'
        with wave.open(filename, 'w') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)
            wf.setframerate(RATE)
            wf.writeframes(audio_int16.tobytes())

        self.update_status(f'✅ Saved: {filename}', '#34d399')
        print(f'Saved: {filename}')
        return filename

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

        print('🎙️ Recording started...')
        while self.running:
            try:
                raw = stream_in.read(CHUNK, exception_on_overflow=False)
                data = np.frombuffer(raw, dtype=np.float32).copy()
                with self.lock:
                    voice = self.current_voice
                output = apply_voice(data, voice)
                self.recorded_frames.append(output)
            except Exception as e:
                print(f'Processing error: {e}')
                continue

        stream_in.stop_stream()
        stream_in.close()
        print('🛑 Recording stopped.')

        with self.lock:
            voice = self.current_voice
        self.save_recording(voice)

# ── Tkinter UI ─────────────────────────────────────────────────────────────────
class App:
    def __init__(self, root):
        self.root = root
        self.root.title('🎙️ Voice Changer Recorder')
        self.root.geometry('420x700')
        self.root.configure(bg='#1a1a2e')
        self.root.resizable(False, False)
        self.recorder = VoiceRecorder()
        self.recorder.set_status_callback(self.update_status)
        self.voices = [
            'Normal', 'Female', 'Male Deep', 'Kid', 'Grandpa',
            'Robot 🤖', 'Ghost 👻', 'Alien 👽', 'Monster 👹', 'Chipmunk 🐿️',
        ]
        self.build_ui()

    def build_ui(self):
        tk.Label(
            self.root, text='🎙️ Voice Changer',
            font=('Arial', 22, 'bold'),
            bg='#1a1a2e', fg='#a78bfa'
        ).pack(pady=12)

        tk.Label(
            self.root,
            text='Select voice → Record → Stop & Save',
            font=('Arial', 10),
            bg='#1a1a2e', fg='#9ca3af'
        ).pack()

        btn_frame = tk.Frame(self.root, bg='#1a1a2e')
        btn_frame.pack(pady=8)

        self.voice_buttons = {}
        for voice in self.voices:
            btn = tk.Button(
                btn_frame, text=voice,
                font=('Arial', 11),
                bg='#2d2b55', fg='white',
                activebackground='#7c3aed',
                relief='flat', width=22, pady=5,
                command=lambda v=voice: self.select_voice(v)
            )
            btn.pack(pady=2)
            self.voice_buttons[voice] = btn

        self.highlight_button('Normal')

        self.status_label = tk.Label(
            self.root, text='⚪ Ready to record',
            font=('Arial', 11),
            bg='#1a1a2e', fg='#9ca3af'
        )
        self.status_label.pack(pady=8)

        ctrl_frame = tk.Frame(self.root, bg='#1a1a2e')
        ctrl_frame.pack(pady=5)

        tk.Button(
            ctrl_frame, text='⏺ Record',
            font=('Arial', 13, 'bold'),
            bg='#ef4444', fg='white',
            relief='flat', width=10, pady=8,
            command=self.start_recording
        ).grid(row=0, column=0, padx=10)

        tk.Button(
            ctrl_frame, text='⏹ Stop & Save',
            font=('Arial', 13, 'bold'),
            bg='#7c3aed', fg='white',
            relief='flat', width=12, pady=8,
            command=self.stop_recording
        ).grid(row=0, column=1, padx=10)

        tk.Label(
            self.root,
            text='💡 Files saved in voice_samples/ folder',
            font=('Arial', 10),
            bg='#1a1a2e', fg='#f59e0b'
        ).pack(pady=8)

    def highlight_button(self, voice):
        for v, btn in self.voice_buttons.items():
            btn.configure(bg='#7c3aed' if v == voice else '#2d2b55')

    def select_voice(self, voice):
        self.recorder.set_voice(voice)
        self.highlight_button(voice)

    def update_status(self, msg, color='#9ca3af'):
        self.status_label.configure(text=msg, fg=color)

    def start_recording(self):
        self.recorder.start_recording()

    def stop_recording(self):
        self.recorder.stop_recording()

if __name__ == '__main__':
    root = tk.Tk()
    app = App(root)
    root.mainloop()


