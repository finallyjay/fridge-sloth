"""Add an ElevenLabs voice-over to the demo video.

Each line is generated separately and placed at the second its moment appears on screen,
then mixed under the silent recording with ffmpeg. Lines are stitched (previous/next text)
so they sound like one continuous take.

Setup: pip install -r requirements-demo.txt, and put ELEVENLABS_API_KEY in .env
(`npx -y @elevenlabs/cli@latest onboard init` does that for you).

Usage: python scripts/narrate_demo.py SILENT_VIDEO OUTPUT_VIDEO
"""

import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv
from elevenlabs import ElevenLabs, VoiceSettings

ROOT = Path(__file__).resolve().parent.parent
VOICE_ID = "JBFqnCBsd6RMkjVDRZzb"  # George: warm, narrative
MODEL_ID = "eleven_v4"
NARRATION = VoiceSettings(stability=0.7, similarity_boost=0.5)  # steady narrator tone

# (start second in the silent video, text). Timeline of the demo: title 0–3.5 s,
# staples ~8 s, photo ~13 s, scan (sped up) 17.8–22 s, grocery list 22–29 s,
# recipes (sped up) 29–32 s, meals 32–46 s, "Add to list" ~46.5 s, end card 56.9 s.
LINES = [
    (0.4, "Meet Fridge Sloth. Snap your fridge, and get a grocery list and healthy meals."),
    (5.9, "First, the things I always want at home. One tap for spinach, and I'll type feta."),
    (12.8, "Now, one photo of the fridge."),
    (17.9, "Gemma, running locally on my six-year-old MacBook, lists what it can actually see."),
    (23.8, "Everything I usually have that's missing goes straight onto the grocery list."),
    (29.3, "Then it suggests three healthy meals, using what's already in the fridge."),
    (
        34.6,
        "A pineapple and lettuce omelette in twenty minutes. A cheesy lettuce wrap. "
        "A quick yogurt bowl.",
    ),
    (45.6, "Missing an ingredient? One tap adds it to the list."),
    (50.2, "No account. No cloud. The photo never leaves my laptop."),
    (57.2, "Fridge Sloth. Snap your fridge, stay on the couch."),
]


def duration(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(out.stdout)


def generate_lines(client: ElevenLabs, out_dir: Path) -> list[tuple[float, Path]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    clips = []
    for i, (start, text) in enumerate(LINES):
        audio = client.text_to_speech.convert(
            text=text,
            voice_id=VOICE_ID,
            model_id=MODEL_ID,
            voice_settings=NARRATION,
            output_format="mp3_44100_128",
            previous_text=LINES[i - 1][1] if i > 0 else None,
            next_text=LINES[i + 1][1] if i + 1 < len(LINES) else None,
        )
        clip = out_dir / f"line{i:02d}.mp3"
        with clip.open("wb") as f:
            for chunk in audio:
                f.write(chunk)
        clips.append((start, clip))
    return clips


def check_timing(clips: list[tuple[float, Path]], video_length: float) -> None:
    for i, (start, clip) in enumerate(clips):
        end = start + duration(clip)
        limit = clips[i + 1][0] if i + 1 < len(clips) else video_length
        warning = "  <-- runs into the next line" if end > limit else ""
        print(f"{start:5.1f}s → {end:5.1f}s  {LINES[i][1][:60]}{warning}")


def mix(video: Path, clips: list[tuple[float, Path]], output: Path) -> None:
    inputs: list[str] = []
    filters = []
    for i, (start, clip) in enumerate(clips):
        inputs += ["-i", str(clip)]
        ms = int(start * 1000)
        filters.append(f"[{i + 1}:a]adelay={ms}|{ms}[a{i}]")
    labels = "".join(f"[a{i}]" for i in range(len(clips)))
    filters.append(f"{labels}amix=inputs={len(clips)}:normalize=0,apad[voice]")
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-y", "-i", str(video), *inputs,
         "-filter_complex", ";".join(filters), "-map", "0:v", "-map", "[voice]",
         "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", str(output)],
        check=True,
    )  # fmt: skip


def main(video: Path, output: Path) -> None:
    load_dotenv(ROOT / ".env")
    # Passed explicitly: ElevenLabs() reads the variable as a default argument, which is
    # evaluated at import time, before load_dotenv() has run.
    client = ElevenLabs(api_key=os.environ["ELEVENLABS_API_KEY"])
    clips = generate_lines(client, output.parent / f"{output.stem}-voice")
    check_timing(clips, duration(video))
    mix(video, clips, output)
    print("saved", output)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve())
