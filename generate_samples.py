from gtts import gTTS
from pydub import AudioSegment
import numpy as np
import scipy.signal as signal
import os

RATE = 44100
OUTPUT_FOLDER = 'voice_samples'
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ── 10 Dialogues ───────────────────────────────────────────────────────────────
VOICES = {
    # Name, dialogue, gender(tld), pitch semitones
    'Male_Voice_01_Deep': ('The storm is coming. Prepare yourself for what lies ahead.', 'com', -4),
    'Male_Voice_02_Neutral': ('Good morning! Today is going to be a great day. Lets get started.', 'com.au', -2),
    'Male_Voice_03_Soft': ('I understand how you feel. Everything will be alright in the end.', 'co.uk', -1),
    'Male_Voice_04_Strong': ('Listen carefully. This is your one and only chance to make it right.','ca', -3),
    'Male_Voice_05_Young': ('Hey! Did you see that? That was absolutely amazing!', 'ie', 0),
    'Female_Voice_01_Soft': ('Welcome home. I missed you so much today.', 'co.in', 3),
    'Female_Voice_02_Neutral':('The meeting is scheduled for 3pm. Please be on time.', 'com', 2),
    'Female_Voice_03_High': ('Oh my God! I cannot believe this is actually happening right now!', 'com.au', 4),
    'Female_Voice_04_Warm': ('Dont worry about it. I am always here for you no matter what.', 'co.uk', 3),
    'Female_Voice_05_Clear': ('Ladies and gentlemen, welcome to todays presentation.', 'ca', 2),
}

def pitch_shift(data, semitones):
    if semitones == 0:
        return data
    factor = 2 ** (semitones / 12.0)
    from scipy.signal import resample
    resampled = resample(data, int(len(data) / factor))
    if len(resampled) > len(data):
        return resampled[:len(data)]
    return np.pad(resampled, (0, len(data) - len(resampled)))

def normalize(data):
    max_val = np.max(np.abs(data))
    if max_val > 0:
        return data / max_val * 0.90
    return data

def generate_voice(name, dialogue, tld, semitones):
    print(f'Generating {name}...')

    # Step 1 — Generate TTS audio
    tts = gTTS(text=dialogue, lang='en', tld=tld)
    temp_path = f'{OUTPUT_FOLDER}/temp_{name}.mp3'
    tts.save(temp_path)

    # Step 2 — Load and convert
    audio = AudioSegment.from_file(temp_path)
    audio = audio.set_channels(1).set_frame_rate(RATE)
    samples = np.array(audio.get_array_of_samples()).astype(np.float32)
    samples = samples / 32767.0

    # Step 3 — Apply pitch shift
    shifted = pitch_shift(samples, semitones)
    normalized = normalize(shifted)

    # Step 4 — Save final WAV
    output_int16 = (normalized * 32767).astype(np.int16)
    final_audio = AudioSegment(
        output_int16.tobytes(),
        frame_rate=RATE,
        sample_width=2,
        channels=1
    )
    output_path = f'{OUTPUT_FOLDER}/{name}.wav'
    final_audio.export(output_path, format='wav')

    # Step 5 — Remove temp file
    os.remove(temp_path)
    print(f'✅ Saved: {output_path}')

# ── Generate all 10 voices ─────────────────────────────────────────────────────
print('🎙️ Generating 10 voice samples...\n')
for name, (dialogue, tld, semitones) in VOICES.items():
    try:
        generate_voice(name, dialogue, tld, semitones)
    except Exception as e:
        print(f'❌ Error on {name}: {e}')

print('\n🎉 All done! Check voice_samples/ folder')
print('Play with: aplay voice_samples/Male_Voice_01_Deep.wav')
