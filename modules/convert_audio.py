
import torchaudio
from pathlib import Path

input_path = Path("data/test_audio/sample_depression.wav")
output_path = Path("data/test_audio/sample_depression_mono.wav")

waveform, sample_rate = torchaudio.load(str(input_path))

# Convert stereo to mono
if waveform.shape[0] > 1:
    waveform = waveform.mean(dim=0, keepdim=True)

torchaudio.save(str(output_path), waveform, sample_rate)

print("Audio converted successfully!")
print("Output:", output_path)
print("Channels:", waveform.shape[0])
