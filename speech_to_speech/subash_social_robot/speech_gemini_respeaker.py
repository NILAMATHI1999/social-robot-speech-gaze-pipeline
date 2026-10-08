#!/usr/bin/env python3

import json
import os
import subprocess
import time
import urllib.request
import wave
from collections import deque

import numpy as np
from faster_whisper import WhisperModel


AUDIO_FILE = "/home/subash/thesis-social-robot/test_audio/live_respeaker.wav"

PIPER_MODEL = (
    "/home/subash/thesis-social-robot/"
    "piper_voices/en_US-lessac-medium.onnx"
)

PIPER_AUDIO_FILE = (
    "/home/subash/thesis-social-robot/"
    "test_audio/piper_echo.wav"
)

GEMINI_MODEL = "gemini-3.5-flash-lite"

GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/"
    f"models/{GEMINI_MODEL}:generateContent"
)

SAMPLE_RATE = 16000
CHANNELS = 6
SAMPLE_WIDTH = 2

CHUNK_SECONDS = 0.1
CHUNK_FRAMES = int(SAMPLE_RATE * CHUNK_SECONDS)
CHUNK_BYTES = CHUNK_FRAMES * CHANNELS * SAMPLE_WIDTH

# Stop a turn after approximately 0.8 seconds of silence.
SILENCE_CHUNKS_TO_STOP = 8

MAX_WAIT_SECONDS = 15
MAX_SPEECH_SECONDS = 20

EXIT_PHRASES = {
    "goodbye",
    "good bye",
    "stop conversation",
    "exit",
    "quit",
}


def calculate_rms(raw_audio):
    samples = np.frombuffer(
        raw_audio,
        dtype=np.int16,
    )

    multichannel = samples.reshape(
        -1,
        CHANNELS,
    )

    mono = multichannel[:, 0].astype(
        np.float32
    )

    return float(
        np.sqrt(
            np.mean(
                mono * mono
            )
        )
    )


def generate_gemini_reply(text):
    api_key = os.environ.get(
        "GEMINI_API_KEY"
    )

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. "
            "Set it in the terminal before running."
        )

    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": (
                            "Reply naturally in one short sentence. "
                            "Keep the answer concise for spoken robot "
                            "interaction.\n\n"
                            f"User: {text}"
                        )
                    }
                ]
            }
        ],
        "generationConfig": {
            "maxOutputTokens": 60,
        },
    }

    request = urllib.request.Request(
        GEMINI_URL,
        data=json.dumps(payload).encode(
            "utf-8"
        ),
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": api_key,
        },
        method="POST",
    )

    with urllib.request.urlopen(
        request,
        timeout=60,
    ) as response:
        result = json.loads(
            response.read().decode(
                "utf-8"
            )
        )

    candidates = result.get(
        "candidates",
        [],
    )

    if not candidates:
        raise RuntimeError(
            "Gemini returned no response candidates."
        )

    parts = (
        candidates[0]
        .get("content", {})
        .get("parts", [])
    )

    reply = " ".join(
        part.get("text", "").strip()
        for part in parts
        if part.get("text")
    ).strip()

    if not reply:
        raise RuntimeError(
            "Gemini returned an empty response."
        )

    return reply

def main():
    print("Loading faster-whisper...")

    model = WhisperModel(
        "base",
        device="cpu",
        compute_type="int8",
    )

    print("Model ready.")

    if not os.environ.get(
        "GEMINI_API_KEY"
    ):
        raise RuntimeError(
            "GEMINI_API_KEY is not set."
        )

    print(
        "Gemini API selected. "
        "Internet connection required."
    )
    print()
    print(
        "Continuous conversation started."
    )
    print(
        "Say 'goodbye', 'stop conversation', "
        "'exit', or 'quit' to stop."
    )

    try:
        while True:
            print()
            print(
                "----------------------------------------"
            )

            recorder = subprocess.Popen(
                [
                    "arecord",
                    "-D",
                    "hw:1,0",
                    "-f",
                    "S16_LE",
                    "-r",
                    str(SAMPLE_RATE),
                    "-c",
                    str(CHANNELS),
                    "-t",
                    "raw",
                ],
                stdout=subprocess.PIPE,
            )

            recorded_chunks = []
            speech_started = False
            silent_chunks = 0

            speech_started_at = None
            last_speech_at = None

            no_speech_timeout = False

            try:
                print(
                    "Stay quiet for one second."
                )

                noise_levels = []

                for _ in range(10):
                    chunk = recorder.stdout.read(
                        CHUNK_BYTES
                    )

                    if len(chunk) != CHUNK_BYTES:
                        raise RuntimeError(
                            "Microphone returned incomplete "
                            "audio during calibration."
                        )

                    noise_levels.append(
                        calculate_rms(chunk)
                    )

                noise_level = float(
                    np.median(
                        noise_levels
                    )
                )

                speech_threshold = max(
                    300.0,
                    noise_level * 3.0,
                )

                print(
                    "Background level:",
                    round(
                        noise_level,
                        1,
                    ),
                )

                print(
                    "Speech threshold:",
                    round(
                        speech_threshold,
                        1,
                    ),
                )

                print("SPEAK NOW")

                pre_roll = deque(
                    maxlen=5
                )

                waiting_started = (
                    time.perf_counter()
                )

                while True:
                    chunk = (
                        recorder.stdout.read(
                            CHUNK_BYTES
                        )
                    )

                    if (
                        len(chunk)
                        != CHUNK_BYTES
                    ):
                        raise RuntimeError(
                            "Microphone recording "
                            "stopped unexpectedly."
                        )

                    now = (
                        time.perf_counter()
                    )

                    level = calculate_rms(
                        chunk
                    )

                    if not speech_started:
                        pre_roll.append(
                            chunk
                        )

                        if (
                            level
                            >= speech_threshold
                        ):
                            speech_started = True

                            speech_started_at = (
                                now
                            )

                            last_speech_at = (
                                now
                            )

                            recorded_chunks.extend(
                                pre_roll
                            )

                            print(
                                "Speech detected."
                            )

                        elif (
                            now
                            - waiting_started
                            >= MAX_WAIT_SECONDS
                        ):
                            no_speech_timeout = (
                                True
                            )

                            print(
                                "No speech detected. "
                                "Listening again."
                            )

                            break

                    else:
                        recorded_chunks.append(
                            chunk
                        )

                        if (
                            level
                            >= speech_threshold
                        ):
                            last_speech_at = (
                                now
                            )

                            silent_chunks = 0

                        else:
                            silent_chunks += 1

                        if (
                            silent_chunks
                            >= SILENCE_CHUNKS_TO_STOP
                        ):
                            print(
                                "Speech ended."
                            )

                            break

                        if (
                            now
                            - speech_started_at
                            >= MAX_SPEECH_SECONDS
                        ):
                            print(
                                "Maximum speech "
                                "duration reached."
                            )

                            break

            finally:
                recorder.terminate()
                recorder.wait()

            # If nobody spoke for 15 seconds,
            # simply start listening again.
            if no_speech_timeout:
                continue

            if not recorded_chunks:
                print(
                    "No speech audio recorded. "
                    "Listening again."
                )
                continue

            raw_audio = b"".join(
                recorded_chunks
            )

            with wave.open(
                AUDIO_FILE,
                "wb",
            ) as wav_file:
                wav_file.setnchannels(
                    CHANNELS
                )

                wav_file.setsampwidth(
                    SAMPLE_WIDTH
                )

                wav_file.setframerate(
                    SAMPLE_RATE
                )

                wav_file.writeframes(
                    raw_audio
                )

            multichannel = (
                np.frombuffer(
                    raw_audio,
                    dtype=np.int16,
                ).reshape(
                    -1,
                    CHANNELS,
                )
            )

            mono = (
                multichannel[
                    :, 0
                ].astype(
                    np.float32
                )
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
                round(
                    utterance_duration,
                    2,
                ),
                "seconds",
            )

            print(
                "Inference time:",
                round(
                    inference_time,
                    2,
                ),
                "seconds",
            )

            print(
                "Speech-start-to-text time:",
                round(
                    speech_start_to_text,
                    2,
                ),
                "seconds",
            )

            print(
                "Speech-end-to-text latency:",
                round(
                    speech_end_to_text,
                    2,
                ),
                "seconds",
            )

            print(
                "Saved audio:",
                AUDIO_FILE,
            )

            if not text:
                print(
                    "No transcription detected. "
                    "Listening again."
                )
                continue

            # Check whether the user wants
            # to end the conversation.
            normalized_text = (
                text.lower()
                .strip()
                .rstrip(".!?")
            )

            if (
                normalized_text
                in EXIT_PHRASES
            ):
                print(
                    "Exit command detected."
                )
                print(
                    "Conversation ended."
                )
                break

            print(
                "Generating Gemini reply..."
            )

            llm_started = (
                time.perf_counter()
            )

            reply = generate_gemini_reply(
                text
            )

            llm_finished = (
                time.perf_counter()
            )

            print(
                "Gemini reply:",
                reply,
            )

            print(
                "Generating speech with Piper..."
            )

            piper_started = (
                time.perf_counter()
            )

            subprocess.run(
                [
                    "piper",
                    "--model",
                    PIPER_MODEL,
                    "--output-file",
                    PIPER_AUDIO_FILE,
                ],
                input=reply,
                text=True,
                check=True,
            )

            piper_finished = (
                time.perf_counter()
            )

            print(
                "Playing generated speech..."
            )

            playback_started = (
                time.perf_counter()
            )

            subprocess.run(
                [
                    "aplay",
                    PIPER_AUDIO_FILE,
                ],
                check=True,
            )

            playback_finished = (
                time.perf_counter()
            )

            llm_time = (
                llm_finished
                - llm_started
            )

            piper_time = (
                piper_finished
                - piper_started
            )

            text_to_speaker_start = (
                playback_started
                - transcription_finished
            )

            playback_time = (
                playback_finished
                - playback_started
            )

            speech_end_to_speaker_start = (
                playback_started
                - last_speech_at
            )

            print()

            print(
                "Gemini response time:",
                round(
                    llm_time,
                    2,
                ),
                "seconds",
            )

            print(
                "Piper synthesis time:",
                round(
                    piper_time,
                    2,
                ),
                "seconds",
            )

            print(
                "Whisper-text-to-speaker-start time:",
                round(
                    text_to_speaker_start,
                    2,
                ),
                "seconds",
            )

            print(
                "Speech-end-to-speaker-start latency:",
                round(
                    speech_end_to_speaker_start,
                    2,
                ),
                "seconds",
            )

            print(
                "Playback duration:",
                round(
                    playback_time,
                    2,
                ),
                "seconds",
            )

            print()
            print(
                "Robot reply finished."
            )
            print(
                "Listening for the next turn..."
            )

    except KeyboardInterrupt:
        print()
        print(
            "Conversation stopped with Ctrl+C."
        )


if __name__ == "__main__":
    main()
