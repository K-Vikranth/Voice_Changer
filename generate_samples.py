import numpy as np
import scipy.signal as signal
from pydub import AudioSegment
import os

INPUT_FILE = 'input.mp3'
OUTPUT_FOLDER = 'voice_samples'
RATE = 44100
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def load_audio(filepath):
    audio = AudioSegment.from_file(filepath)
    audio = audio.set_channels(1).set_frame_rate(RATE)
    samples = np.array(audio.get_array_of_samples()).astype(np.float32)
    samples = samples / 32767.0
    return samples

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
    factor = 2 ** (semitones / 12.0)
    resampled = signal.resample(data, int(len(data) / factor))
    if len(resampled) > len(data):
        return resampled[:len(data)]
    return np.pad(resampled, (0, len(data) - len(resampled)))

def echo_effect(data, delay_samples=8000, decay=0.4):
    output = data.copy()
    if len(data) > delay_samples:
        output[delay_samples:] += data[:-delay_samples] * decay
    return output

def robot_effect(data):
    t = np.arange(len(data)) / RATE
    carrier = np.sin(2 * np.pi * 80 * t)
    return np.clip(data * carrier * 2.0, -1.0, 1.0)

def wobble_effect(data):
    t = np.arange(len(data)) / RATE
    wobble = np.sin(2 * np.pi * 5 * t) * 0.3 + 1.0
    return data * wobble

def growl_effect(data):
    t = np.arange(len(data)) / RATE
    growl = np.sin(2 * np.pi * 40 * t) * 0.3
    return np.clip(data + growl, -1.0, 1.0)

# Load input
print(f'Loading {INPUT_FILE}...')
data = load_audio(INPUT_FILE)
print(f'✅ Loaded! Duration: {len(data)/RATE:.1f} seconds')

# Generate all 10 voices
voices = {
    '01_Normal': data,
    '02_Female': pitch_shift(data, semitones=4),
    '03_Male_Deep': pitch_shift(data, semitones=-4),
    '04_Kid': pitch_shift(data, semitones=8),
    '05_Grandpa': echo_effect(pitch_shift(data, semitones=-5)),
    '06_Robot': robot_effect(data),
    '07_Ghost': echo_effect(pitch_shift(data, semitones=-3), delay_samples=6000, decay=0.5),
    '08_Alien': wobble_effect(pitch_shift(data, semitones=7)),
    '09_Monster': growl_effect(pitch_shift(data, semitones=-8)),
    '10_Chipmunk': pitch_shift(data, semitones=10),
}

print('\nGenerating voice samples...')
for name, audio in voices.items():
    normalized = normalize(audio)
    save_audio(normalized, f'{OUTPUT_FOLDER}/{name}.wav')

print(f'\n🎉 All 10 voices saved in {OUTPUT_FOLDER}/ folder!')
print('Play them with: aplay voice_samples/01_Normal.wav')
