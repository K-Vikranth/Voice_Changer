import numpy as np
import scipy.signal as signal
from pydub import AudioSegment
import tkinter as tk
from tkinter import messagebox
import os

# ── Settings ───────────────────────────────────────────────────────────────────
RATE = 44100
VOICES_FOLDER = 'voice_samples'
OUTPUT_FOLDER = 'cloned_output'
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ── Voice definitions ──────────────────────────────────────────────────────────
MALE_VOICES = {
    'Male Voice 1 — Deep': ('Male_Voice_01.mp3', -2),
    'Male Voice 2 — Neutral': ('Male_Voice_02.mp3', 0),
    'Male Voice 3 — Soft': ('Male_Voice_03.mp3', -1),
    'Male Voice 4 — Strong': ('Male_Voice_04.mp3', -3),
    'Male Voice 5 — Young': ('Male_Voice_05.mp3', 1),
}

FEMALE_VOICES = {
    'Female Voice 1 — Soft': ('Female_Voice_01.mp3', 2),
    'Female Voice 2 — Neutral': ('Female_Voice_02.mp3', 0),
    'Female Voice 3 — High': ('Female_Voice_03.mp3', 3),
    'Female Voice 4 — Warm': ('Female_Voice_04.mp3', 1),
    'Female Voice 5 — Clear': ('Female_Voice_05.mp3', 2),
}

# ── Audio Processing ───────────────────────────────────────────────────────────
def load_audio(filepath):
    audio = AudioSegment.from_file(filepath)
    audio = audio.set_channels(1).set_frame_rate(RATE)
    samples = np.array(audio.get_array_of_samples()).astype(np.float32)
    return samples / 32767.0

def save_audio(samples, filename):
    samples = np.clip(samples, -1.0, 1.0)
    audio_int16 = (samples * 32767).astype(np.int16)
    audio_seg = AudioSegment(
        audio_int16.tobytes(),
        frame_rate=RATE,
        sample_width=2,
        channels=1
    )
    audio_seg.export(filename, format='wav')
    print(f'✅ Saved: {filename}')

def normalize(data):
    max_val = np.max(np.abs(data))
    if max_val > 0:
        return data / max_val * 0.90
    return data

def pitch_shift(data, semitones):
    if semitones == 0:
        return data
    factor = 2 ** (semitones / 12.0)
    resampled = signal.resample(data, int(len(data) / factor))
    if len(resampled) > len(data):
        return resampled[:len(data)]
    return np.pad(resampled, (0, len(data) - len(resampled)))

def clone_voice(source_file, semitones, output_name):
    """Load voice sample, apply effect, save output."""
    filepath = os.path.join(VOICES_FOLDER, source_file)
    if not os.path.exists(filepath):
        print(f'❌ File not found: {filepath}')
        return None
    data = load_audio(filepath)
    shifted = pitch_shift(data, semitones)
    normalized = normalize(shifted)
    output_path = os.path.join(OUTPUT_FOLDER, f'{output_name}.wav')
    save_audio(normalized, output_path)
    return output_path

# ── App ────────────────────────────────────────────────────────────────────────
class App:
    def __init__(self, root):
        self.root = root
        self.root.title('🎙️ Voice Cloner')
        self.root.geometry('460x720')
        self.root.configure(bg='#1a1a2e')
        self.root.resizable(False, False)
        self.selected_voice = None
        self.selected_file = None
        self.selected_semitones = 0
        self.build_ui()

    def build_ui(self):
        # Title
        tk.Label(
            self.root, text='🎙️ Voice Cloner',
            font=('Arial', 22, 'bold'),
            bg='#1a1a2e', fg='#a78bfa'
        ).pack(pady=12)

        tk.Label(
            self.root,
            text='Select a voice → Clone & Save',
            font=('Arial', 10),
            bg='#1a1a2e', fg='#9ca3af'
        ).pack()

        # Male voices
        tk.Label(
            self.root, text='👨 Male Voices',
            font=('Arial', 13, 'bold'),
            bg='#1a1a2e', fg='#60a5fa'
        ).pack(pady=(15, 4))

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

        # Female voices
        tk.Label(
            self.root, text='👩 Female Voices',
            font=('Arial', 13, 'bold'),
            bg='#1a1a2e', fg='#f472b6'
        ).pack(pady=(15, 4))

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

        # Status
        self.status_label = tk.Label(
            self.root, text='⚪ Select a voice to clone',
            font=('Arial', 11),
            bg='#1a1a2e', fg='#9ca3af'
        )
        self.status_label.pack(pady=10)

        # Clone button
        tk.Button(
            self.root, text='🎙️ Clone & Save',
            font=('Arial', 14, 'bold'),
            bg='#7c3aed', fg='white',
            activebackground='#6d28d9',
            relief='flat', width=20, pady=10,
            command=self.clone
        ).pack(pady=5)

        tk.Label(
            self.root,
            text='💡 Cloned files saved in cloned_output/ folder',
            font=('Arial', 10),
            bg='#1a1a2e', fg='#f59e0b'
        ).pack(pady=5)

    def highlight_button(self, selected):
        for name, btn in self.voice_buttons.items():
            if name == selected:
                btn.configure(bg='#7c3aed')
            elif name in [n for n in MALE_VOICES]:
                btn.configure(bg='#1e3a5f')
            else:
                btn.configure(bg='#4a1942')

    def select_voice(self, name, file, semitones):
        self.selected_voice = name
        self.selected_file = file
        self.selected_semitones = semitones
        self.highlight_button(name)
        self.status_label.configure(
            text=f'✅ Selected: {name}',
            fg='#34d399'
        )

    def clone(self):
        if not self.selected_voice:
            messagebox.showwarning('No Voice', 'Please select a voice first!')
            return

        self.status_label.configure(text='⏳ Cloning...', fg='#f59e0b')
        self.root.update()

        safe_name = self.selected_voice.replace(' ', '_').replace('—', '')
        result = clone_voice(
            self.selected_file,
            self.selected_semitones,
            safe_name
        )

        if result:
            self.status_label.configure(
                text=f'✅ Saved: {result}',
                fg='#34d399'
            )
        else:
            self.status_label.configure(
                text='❌ Error cloning voice!',
                fg='#ef4444'
            )

if __name__ == '__main__':
    root = tk.Tk()
    app = App(root)
    root.mainloop()
