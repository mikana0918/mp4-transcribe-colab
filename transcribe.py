#!/usr/bin/env python3
"""Extract audio, separate vocals, and transcribe video files with Whisper."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def extract_source_wav(video: Path, out_dir: Path) -> Path:
    source = out_dir / "source.wav"
    run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(video),
            "-vn",
            "-ac",
            "2",
            "-ar",
            "44100",
            str(source),
        ]
    )
    return source


def separate_vocals(source: Path, out_dir: Path, device: str) -> Path:
    demucs_out = out_dir / "demucs"
    run(
        [
            sys.executable,
            "-m",
            "demucs",
            "--two-stems",
            "vocals",
            "-d",
            device,
            "-o",
            str(demucs_out),
            str(source),
        ]
    )
    vocals = demucs_out / "htdemucs" / source.stem / "vocals.wav"
    if not vocals.exists():
        raise FileNotFoundError(f"Vocals not found: {vocals}")
    return vocals


def to_whisper_wav(vocals: Path, out_dir: Path) -> Path:
    whisper_wav = out_dir / "vocals_16k_mono.wav"
    run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(vocals),
            "-ac",
            "1",
            "-ar",
            "16000",
            str(whisper_wav),
        ]
    )
    return whisper_wav


def transcribe(audio: Path, out_dir: Path, model: str, device: str) -> None:
    cmd = [
        sys.executable,
        "-m",
        "whisper",
        str(audio),
        "--model",
        model,
        "--language",
        "ja",
        "--device",
        device,
        "--output_dir",
        str(out_dir),
        "--output_format",
        "all",
    ]
    if device == "cuda":
        cmd.extend(["--fp16", "False"])
    run(cmd)


def process_video(
    video: Path,
    output_root: Path,
    model: str,
    device: str,
    skip_separation: bool,
) -> Path:
    out_dir = output_root / video.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    whisper_wav = out_dir / "vocals_16k_mono.wav"
    if whisper_wav.exists():
        print(f"skip extract/separate: {whisper_wav}")
    elif skip_separation:
        run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(video),
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                str(whisper_wav),
            ]
        )
    else:
        source = extract_source_wav(video, out_dir)
        vocals = separate_vocals(source, out_dir, device)
        whisper_wav = to_whisper_wav(vocals, out_dir)

    transcribe(whisper_wav, out_dir, model, device)
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="Video files")
    parser.add_argument(
        "-o",
        "--output-root",
        type=Path,
        default=Path("static/transcripts/preprocessed"),
        help="Output root directory",
    )
    parser.add_argument("--model", default="small", help="Whisper model")
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device for demucs and whisper",
    )
    parser.add_argument(
        "--skip-separation",
        action="store_true",
        help="Only extract mono 16kHz audio (no demucs)",
    )
    args = parser.parse_args()

    for video in args.inputs:
        if not video.exists():
            raise SystemExit(f"Not found: {video}")
        print(f"\n=== {video.name} ===")
        out_dir = process_video(
            video,
            args.output_root,
            args.model,
            args.device,
            args.skip_separation,
        )
        print(f"done: {out_dir}")


if __name__ == "__main__":
    main()
