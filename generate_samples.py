import subprocess
import os
from pydub import AudioSegment
import numpy as np
import scipy.signal as signal

RATE = 44100
OUTPUT_FOLDER = 'voice_samples'
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ── 10 Voices with espeak-ng ───────────────────────────────────────────────────
VOICES = {
    'Male_Voice_01_Deep': ('mb-en1', 'The storm is coming. Prepare yourself for what lies ahead.'),
    'Male_Voice_02_Neutral': ('mb-us2', 'Good morning! Today is going to be a great day. Lets get started.'),
    'Male_Voice_03_Soft': ('mb-us3', 'I understand how you feel. Everything will be alright in the end.'),
    'Male_Voice_04_Strong': ('en-us+m3', 'Listen carefully. This is your one and only chance to make it right.'),
    'Male_Voice_05_Young': ('en+m5', 'Hey! Did you see that? That was absolutely amazing!'),
    'Female_Voice_01_Soft': ('mb-us1', 'Welcome home. I missed you so much today.'),
    'Female_Voice_02_Neutral': ('en-us+f3', 'The meeting is scheduled for 3pm. Please be on time.'),
    'Female_Voice_03_High': ('en-us+f4', 'Oh my God! I cannot believe this is actually happening right now!'),
    'Female_Voice_04_Warm': ('en-us+f5', 'Dont worry about it. I am always here for you no matter what.'),
    'Female_Voice_05_Clear': ('en+f2', 'Ladies and gentlemen, welcome to todays presentation.'),
}

def normalize(data):
    max_val = np.max(np.abs(data))
    if max_val > 0:
        return data / max_val * 0.90
    return data

def generate_voice(name, espeak_voice, dialogue):
    print(f'Generating {name}...')
    temp_wav = f'{OUTPUT_FOLDER}/temp_{name}.wav'
    output_wav = f'{OUTPUT_FOLDER}/{name}.wav'

    # Generate using espeak-ng directly to WAV
    result = subprocess.run([
        'espeak-ng',
        '-v', espeak_voice,
        '-w', temp_wav,
        dialogue
    ], capture_output=True)

    if result.returncode != 0:
        print(f'❌ Error: {result.stderr.decode()}')
        return

    # Load and normalize
    audio = AudioSegment.from_wav(temp_wav)
    audio = audio.set_channels(1).set_frame_rate(RATE)
    samples = np.array(audio.get_array_of_samples()).astype(np.float32)
    samples = samples / 32767.0
    normalized = normalize(samples)

    # Save final WAV
    output_int16 = (normalized * 32767).astype(np.int16)
    final_audio = AudioSegment(
        output_int16.tobytes(),
        frame_rate=RATE,
        sample_width=2,
        channels=1
    )
    final_audio.export(output_wav, format='wav')

    # Remove temp file
    os.remove(temp_wav)
    print(f'✅ Saved: {output_wav}')

# ── Generate all 10 ────────────────────────────────────────────────────────────
print('🎙️ Generating 10 voice samples...\n')
for name, (voice, dialogue) in VOICES.items():
    try:
        generate_voice(name, voice, dialogue)
    except Exception as e:
        print(f'❌ Error on {name}: {e}')

print('\n🎉 All done! Check voice_samples/ folder')
print('\nPlay samples:')
print('aplay voice_samples/Male_Voice_01_Deep.wav')
print('aplay voice_samples/Female_Voice_01_Soft.wav')
