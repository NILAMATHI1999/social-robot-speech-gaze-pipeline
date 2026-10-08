#!/usr/bin/env python3

import subprocess
import time
import wave
from collections import deque

import numpy as np
from faster_whisper import WhisperModel


AUDIO_FILE = "/home/subash/thesis-social-robot/test_audio/live_respeaker.wav"
PIPER_MODEL = "/home/subash/thesis-social-robot/piper_voices/en_US-lessac-medium.onnx"
PIPER_AUDIO_FILE = "/home/subash/thesis-social-robot/test_audio/piper_echo.wav"

SAMPLE_RATE = 16000
CHANNELS = 6
SAMPLE_WIDTH = 2

CHUNK_SECONDS = 0.1
CHUNK_FRAMES = int(SAMPLE_RATE * CHUNK_SECONDS)
CHUNK_BYTES = CHUNK_FRAMES * CHANNELS * SAMPLE_WIDTH

# Stop after approximately 0.8 seconds of silence
SILENCE_CHUNKS_TO_STOP = 8

MAX_WAIT_SECONDS = 15
MAX_SPEECH_SECONDS = 20


def calculate_rms(raw_audio):
    samples = np.frombuffer(raw_audio, dtype=np.int16)

    stereo = samples.reshape(-1, CHANNELS)

    mono = stereo[:, 0].astype(np.float32)
    return float(
        np.sqrt(np.mean(mono * mono))
    )


def main():
    print("Loading faster-whisper...")

    model = WhisperModel(
        "base",
        device="cpu",
        compute_type="int8",
    )

    print("Model ready.")

    recorder = subprocess.Popen(
        [
            "arecord",
            "-D", "hw:1,0",
            "-f", "S16_LE",
            "-r", str(SAMPLE_RATE),
            "-c", str(CHANNELS),
            "-t", "raw",
        ],
        stdout=subprocess.PIPE,
    )

    try:
        print("Stay quiet for one second.")

        noise_levels = []

        for _ in range(10):
            chunk = recorder.stdout.read(CHUNK_BYTES)

            if len(chunk) != CHUNK_BYTES:
                raise RuntimeError(
                    "Microphone returned incomplete audio during calibration."
                )

            noise_levels.append(
                calculate_rms(chunk)
            )

        noise_level = float(
            np.median(noise_levels)
        )

        speech_threshold = max(
            300.0,
            noise_level * 3.0,
        )

        print(
            "Background level:",
            round(noise_level, 1),
        )

        print(
            "Speech threshold:",
            round(speech_threshold, 1),
        )

        print("SPEAK NOW")

        pre_roll = deque(maxlen=5)
        recorded_chunks = []

        speech_started = False
        silent_chunks = 0

        waiting_started = time.perf_counter()

        speech_started_at = None
        last_speech_at = None

        while True:
            chunk = recorder.stdout.read(
                CHUNK_BYTES
            )

            if len(chunk) != CHUNK_BYTES:
                raise RuntimeError(
                    "Microphone recording stopped unexpectedly."
                )

            now = time.perf_counter()
            level = calculate_rms(chunk)

            if not speech_started:
                pre_roll.append(chunk)

                if level >= speech_threshold:
                    speech_started = True

                    speech_started_at = now
                    last_speech_at = now

                    recorded_chunks.extend(
                        pre_roll
                    )

                    print("Speech detected.")

                elif (
                    now - waiting_started
                    >= MAX_WAIT_SECONDS
                ):
                    raise RuntimeError(
                        "No speech detected within 15 seconds."
                    )

            else:
                recorded_chunks.append(chunk)

                if level >= speech_threshold:
                    last_speech_at = now
                    silent_chunks = 0

                else:
                    silent_chunks += 1

                if (
                    silent_chunks
                    >= SILENCE_CHUNKS_TO_STOP
                ):
                    print("Speech ended.")
                    break

                if (
                    now - speech_started_at
                    >= MAX_SPEECH_SECONDS
                ):
                    print(
                        "Maximum speech duration reached."
                    )
                    break

    finally:
        recorder.terminate()
        recorder.wait()

    if not recorded_chunks:
        raise RuntimeError(
            "No speech audio was recorded."
        )

    raw_audio = b"".join(
        recorded_chunks
    )

    with wave.open(AUDIO_FILE, "wb") as wav_file:
        wav_file.setnchannels(CHANNELS)
        wav_file.setsampwidth(SAMPLE_WIDTH)
        wav_file.setframerate(SAMPLE_RATE)
        wav_file.writeframes(raw_audio)

    stereo = np.frombuffer(
        raw_audio,
        dtype=np.int16,
    ).reshape(-1, CHANNELS)

    mono = (
        stereo[:, 0].astype(np.float32)
        / 32768.0
    )

    print("Transcribing...")

    transcription_started = (
        time.perf_counter()
    )

    segments_generator, info = (
        model.transcribe(
            mono,
            language="en",
            beam_size=5,
        )
    )

    segments = list(
        segments_generator
    )

    transcription_finished = (
        time.perf_counter()
    )

    text = " ".join(
        segment.text.strip()
        for segment in segments
    )

    inference_time = (
        transcription_finished
        - transcription_started
    )

    utterance_duration = (
        last_speech_at
        - speech_started_at
    )

    speech_start_to_text = (
        transcription_finished
        - speech_started_at
    )

    speech_end_to_text = (
        transcription_finished
        - last_speech_at
    )

    print()

    print(
        "Transcription:",
        text,
    )

    print(
        "Utterance duration:",
        round(utterance_duration, 2),
        "seconds",
    )

    print(
        "Inference time:",
        round(inference_time, 2),
        "seconds",
    )

    print(
        "Speech-start-to-text time:",
        round(speech_start_to_text, 2),
        "seconds",
    )

    print(
        "Speech-end-to-text latency:",
        round(speech_end_to_text, 2),
        "seconds",
    )

    print(
        "Saved audio:",
        AUDIO_FILE,
    )
    if text:
        print("Generating speech with Piper...")

        subprocess.run(
            [
                "piper",
                "--model",
                PIPER_MODEL,
                "--output-file",
                PIPER_AUDIO_FILE,
            ],
            input=text,
            text=True,
            check=True,
        )

        print("Playing generated speech...")

        subprocess.run(
            [
                "aplay",
                PIPER_AUDIO_FILE,
            ],
            check=True,
        )
    else:
        print("No transcription to speak.")

if __name__ == "__main__":
    main()
