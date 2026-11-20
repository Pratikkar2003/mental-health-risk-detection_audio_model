
import sys
from pathlib import Path

print("DAM RUNNER STARTED")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DAM_DIR = PROJECT_ROOT / "models" / "dam"

sys.path.insert(0, str(DAM_DIR))

print("DAM directory:", DAM_DIR)

from pipeline import Pipeline

def main():

    audio_path = PROJECT_ROOT / "data" / "test_audio" / "sample_depression_mono.wav"
    checkpoint_path = DAM_DIR / "dam3.1.ckpt"

    print("Checking audio file...")

    if not audio_path.exists():
        print("ERROR: Audio file not found!")
        return

    if not checkpoint_path.exists():
        print("ERROR: Checkpoint not found!")
        return

    print("Loading DAM model...")

    pipeline = Pipeline(checkpoint=checkpoint_path)

    print("Model loaded successfully!")

    print("Running audio prediction...")

    result = pipeline.run_on_file(str(audio_path), quantize=True)
    

    print("\nPrediction Result:")
    print(result)


if __name__ == "__main__":
    main()
