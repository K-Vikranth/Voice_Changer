import pyaudio
import numpy as np
import scipy.signal as signal
import threading
import tkinter as tk

# ── Audio Settings ─────────────────────────────────────────────────────────────
CHUNK = 4096 # Increased from 1024 to reduce grinding
RATE = 44100
CHANNELS = 1
FORMAT = pyaudio.paFloat32

# ── Voice Effects ──────────────────────────────────────────────────────────────
def pitch_shift(data, semitones):
    """Smooth pitch shift using scipy resample."""
    factor = 2 ** (semitones / 12.0)
    resampled = signal.resample(data, int(len(data) / factor))
    return pad_or_trim(resampled, len(data))

def robot_effect(data):
    t = np.arange(len(data)) / RATE
    carrier = np.sin(2 * np.pi * 80 * t)
    return np.clip(data * carrier * 2.0, -1.0, 1.0)

def echo_effect(data, delay_samples=4000, decay=0.4):
    output = data.copy()
    if len(data) > delay_samples:
        output[delay_samples:] += data[:-delay_samples] * decay
    return output

def wobble_effect(data):
    t = np.arange(len(data)) / RATE
    wobble = np.sin(2 * np.pi * 6 * t) * 0.3 + 1.0
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

def pad_or_trim(data, target_len):
    if len(data) >= target_len:
        return data[:target_len]
    return np.pad(data, (0, target_len - len(data)))

def apply_voice(data, voice):
    data = data.astype(np.float32)
    original_len = len(data)

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
            result = echo_effect(shifted, delay_samples=3000, decay=0.3)

        elif voice == 'Robot 🤖':
            result = robot_effect(data)

        elif voice == 'Ghost 👻':
            shifted = pitch_shift(data, semitones=-3)
            result = echo_effect(shifted, delay_samples=5000, decay=0.5)

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

    result = pad_or_trim(result, original_len)
    return normalize(result).astype(np.float32)

# ── Voice Changer Engine ───────────────────────────────────────────────────────
class VoiceChanger:
    def __init__(self):
        self.audio = pyaudio.PyAudio()
        self.stream_in = None
        self.stream_out = None
        self.running = False
        self.current_voice = 'Normal'
        self.thread = None
        self.lock = threading.Lock()

    def set_voice(self, voice):
        with self.lock:
            self.current_voice = voice
        print(f'Voice changed to: {voice}')

    def process(self):
        try:
            self.stream_in = self.audio.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK
            )
            self.stream_out = self.audio.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                output=True,
                frames_per_buffer=CHUNK
            )
        except Exception as e:
            print(f'Stream error: {e}')
            return

        print('🎙️ Voice changer running...')
        while self.running:
            try:
                raw = self.stream_in.read(CHUNK, exception_on_overflow=False)
                data = np.frombuffer(raw, dtype=np.float32).copy()
                with self.lock:
                    voice = self.current_voice
                output = apply_voice(data, voice)
                self.stream_out.write(output.tobytes())
            except Exception as e:
                print(f'Processing error: {e}')
                continue

        self.stream_in.stop_stream()
        self.stream_in.close()
        self.stream_out.stop_stream()
        self.stream_out.close()
        print('🛑 Stopped.')

    def start(self):
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self.process, daemon=True)
            self.thread.start()

    def stop(self):
        self.running = False

# ── Tkinter UI ─────────────────────────────────────────────────────────────────
class App:
    def __init__(self, root):
        self.root = root
        self.root.title('🎙️ Voice Changer')
        self.root.geometry('400x620')
        self.root.configure(bg='#1a1a2e')
        self.root.resizable(False, False)
        self.vc = VoiceChanger()
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
        ).pack(pady=15)

        tk.Label(
            self.root, text='Select a Voice:',
            font=('Arial', 12),
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
                relief='flat', width=22, pady=6,
                command=lambda v=voice: self.select_voice(v)
            )
            btn.pack(pady=2)
            self.voice_buttons[voice] = btn

        self.highlight_button('Normal')

        self.status_label = tk.Label(
            self.root, text='⚪ Stopped',
            font=('Arial', 12),
            bg='#1a1a2e', fg='#9ca3af'
        )
        self.status_label.pack(pady=10)

        ctrl_frame = tk.Frame(self.root, bg='#1a1a2e')
        ctrl_frame.pack()

        tk.Button(
            ctrl_frame, text='▶ Start',
            font=('Arial', 13, 'bold'),
            bg='#7c3aed', fg='white',
            relief='flat', width=10, pady=8,
            command=self.start
        ).grid(row=0, column=0, padx=10)

        tk.Button(
            ctrl_frame, text='⏹ Stop',
            font=('Arial', 13, 'bold'),
            bg='#dc2626', fg='white',
            relief='flat', width=10, pady=8,
            command=self.stop
        ).grid(row=0, column=1, padx=10)

    def highlight_button(self, voice):
        for v, btn in self.voice_buttons.items():
            btn.configure(bg='#7c3aed' if v == voice else '#2d2b55')

    def select_voice(self, voice):
        self.vc.set_voice(voice)
        self.highlight_button(voice)

    def start(self):
        self.vc.start()
        self.status_label.configure(text='🟢 Running...', fg='#34d399')

    def stop(self):
        self.vc.stop()
        self.status_label.configure(text='⚪ Stopped', fg='#9ca3af')

if __name__ == '__main__':
    root = tk.Tk()
    app = App(root)
    root.mainloop()

