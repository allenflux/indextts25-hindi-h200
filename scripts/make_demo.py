#!/usr/bin/env python3
"""Create tiny synthetic PCM tones for CPU pipeline checks, never speech training."""
import argparse
import json
import math
import struct
import wave
from pathlib import Path

TEXTS = ["नमस्ते मित्र।", "आज मौसम सुहाना है।", "कृपया दरवाज़ा खोलिए।", "यह एक छोटी कहानी है।", "मैं घर जा रहा हूँ।", "पानी साफ़ होना चाहिए।"]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    for i, text in enumerate(TEXTS):
        name = f"tone_{i}.wav"
        with wave.open(str(args.output / name), "wb") as audio:
            audio.setparams((1, 2, 24000, 0, "NONE", "not compressed"))
            audio.writeframes(b"".join(struct.pack("<h", int(1000 * math.sin(2 * math.pi * (200 + i * 50) * t / 24000))) for t in range(28800)))
        rows.append(dict(id=f"demo_{i}", speaker_id=f"synthetic_{i // 2}", audio=name, text=text, language="hi", source="synthetic-test-tone-no-speech-alignment", license="CC0-1.0", consent=True))
    (args.output / "input.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    print("Created six synthetic tones; these are NOT aligned speech or training data.")

if __name__ == "__main__":
    main()
