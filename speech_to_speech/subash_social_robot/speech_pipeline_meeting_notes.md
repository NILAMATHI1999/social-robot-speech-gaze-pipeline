  # Local Speech-Interaction Pipeline — Meeting Notes

  Date: 24 August 2026

  ## 1. Objective

  The objective was to reproduce and evaluate a local speech-interaction pipeline
  using the ReSpeaker microphone:

  User speech
  → speech-to-text
  → local language model
  → text-to-speech
  → speaker output

  The system runs locally without using the OpenAI API.

  ## 2. Hardware and software

  ### Laptop

  - Operating system: Ubuntu 24.04
  - ROS version: ROS 2 Jazzy
  - CPU: Intel Core i5-1135G7
  - RAM: approximately 8 GB
  - Integrated GPU: Intel Iris Xe
  - Dedicated GPU: NVIDIA GeForce MX330
  - Microphone: ReSpeaker microphone array
  - Audio output: external/laptop speaker

  ### Speech recognition

  - Engine: Faster Whisper
  - Model: base
  - Device: CPU
  - Compute type: int8

  ### Local language model

  - Runtime: Ollama
  - Ollama version: 0.32.15
  - Model: qwen2.5:1.5b
  - Execution mode: CPU AVX2
  - Model keep-alive: 10 minutes

  ### Text-to-speech

  - Engine: Piper
  - Piper version: 1.7.0
  - Voice: en_US-lessac-medium
  - Output format: WAV
  - Piper sample rate: 22,050 Hz
  - Piper output: mono, signed 16-bit audio

  ## 3. Audio configuration

  The ReSpeaker is recorded using:

  ALSA device: hw:1,0
  Sample rate: 16,000 Hz
  Channels: 6
  Sample width: 16-bit
  Format: signed little-endian PCM

  All six ReSpeaker channels are captured, but channel 0 is currently passed to
  Whisper.

  The current speech endpoint rules are:

  - Wait up to 15 seconds for speech to begin.
  - Stop the turn after approximately 0.8 seconds of silence.
  - Maximum utterance duration: 20 seconds.
  - Preserve approximately 0.5 seconds of pre-roll audio.

  ## 4. Development stages completed

  ### Stage 1: ReSpeaker to Whisper

  User speech
  → ReSpeaker
  → Faster Whisper
  → recognized text in terminal

  Status: completed.

  ### Stage 2: Piper test

  Typed text
  → Piper
  → WAV file
  → speaker playback

  Status: completed.

  ### Stage 3: Speech echo

  User speech
  → ReSpeaker
  → Whisper
  → recognized text
  → Piper speaks the same text

  Status: completed.

  ### Stage 4: Local LLM

  Text prompt
  → Ollama
  → Qwen2.5 1.5B
  → generated reply

  Status: completed.

  ### Stage 5: Complete speech-to-speech pipeline

  User speech
  → ReSpeaker
  → Faster Whisper
  → Qwen
  → Piper
  → speaker

  Status: completed.

  ### Stage 6: Continuous interaction

  Listen
  → Whisper
  → Qwen
  → Piper
  → speaker
  → listen again

  Status: completed.

  ## 5. Important latency definitions

  ### Whisper inference time

  Time taken by Faster Whisper to process the recorded audio.

  ### Speech-end-to-text latency

  Time from the last detected speech until the final transcription becomes
  available.

  This includes:

  end-of-speech detection
  + Whisper processing
  + small program overhead

  ### LLM response time

  Time taken by Qwen to generate its complete text response.

  ### Piper synthesis time

  Time taken by Piper to convert the LLM response into a WAV file.

  ### Speech-end-to-speaker-start latency

  Time from the user finishing their speech until the robot begins speaking.

  User finishes speaking
  → endpoint detection
  → Whisper
  → Qwen
  → Piper
  → speaker begins

  This is the primary conversational response-latency measurement.

  ### Playback duration

  Time taken to play the complete generated robot response.

  Playback duration depends strongly on response length, so it is not the primary
  processing-latency measurement.

  ## 6. STT-to-Piper echo baseline

  Three short-sentence tests were collected before adding the LLM.

   Measurement                    Run 1     Run 2     Run 3    Average
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━  ━━━━━━━━  ━━━━━━━━  ━━━━━━━━━
   Utterance duration            3.38 s    3.25 s    3.87 s     3.50 s
  ────────────────────────────  ────────  ────────  ────────  ─────────
   Whisper inference             0.45 s    0.49 s    0.46 s     0.47 s
  ────────────────────────────  ────────  ────────  ────────  ─────────
   Speech start → text           4.68 s    4.72 s    5.19 s     4.86 s
  ────────────────────────────  ────────  ────────  ────────  ─────────
   Speech end → text             1.30 s    1.47 s    1.31 s     1.36 s
  ────────────────────────────  ────────  ────────  ────────  ─────────
   Piper synthesis               2.02 s    1.26 s    1.31 s     1.53 s
  ────────────────────────────  ────────  ────────  ────────  ─────────
   Speech end → speaker start    3.33 s    2.73 s    2.62 s     2.89 s
  ────────────────────────────  ────────  ────────  ────────  ─────────
   Playback duration             3.96 s    3.56 s    3.93 s     3.82 s

  ### Echo-test accuracy

  - Two runs reproduced the reference sentence correctly.
  - One run omitted the word “the.”
  - No utterance was cut off.
  - A longer utterance of approximately 16.37 seconds was also transcribed
    successfully.

  ## 7. First complete Whisper-Qwen-Piper tests

  ### Cold or partly loaded test

  Input transcription:

  Hello Robot. Hello Robot, what is your name?

  Qwen response:

  Hello! I'm Qwen, created by Alibaba Cloud.

  Measurements:

   Measurement                    Result
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━
   Utterance duration             2.25 s
  ────────────────────────────  ─────────
   Whisper inference              0.46 s
  ────────────────────────────  ─────────
   Speech end → text              1.33 s
  ────────────────────────────  ─────────
   Qwen response                  7.45 s
  ────────────────────────────  ─────────
   Piper synthesis                4.12 s
  ────────────────────────────  ─────────
   Speech end → speaker start    12.91 s
  ────────────────────────────  ─────────
   Playback duration              4.22 s

  The LLM was not fully warm, so this interaction had a large delay.

  ### Warm test

  Input transcription:

  What is the capital of Germany?

  Qwen response:

  The capital of Germany is Berlin.

  Measurements:

   Measurement                   Result
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━
   Utterance duration            1.63 s
  ────────────────────────────  ────────
   Whisper inference             0.43 s
  ────────────────────────────  ────────
   Speech end → text             1.43 s
  ────────────────────────────  ────────
   Qwen response                 0.71 s
  ────────────────────────────  ────────
   Piper synthesis               2.01 s
  ────────────────────────────  ────────
   Speech end → speaker start    4.16 s
  ────────────────────────────  ────────
   Playback duration             1.93 s

  This demonstrated the difference between cold and warm LLM operation.

  ## 8. Qwen cold versus warm behaviour

  A cold model is not currently loaded in memory. Ollama must load the model from
  disk into RAM before generating a response.

  A warm model is already loaded and can immediately process a request.

   Condition                          Qwen response    Speech end → speaker start
  ━━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   Cold or partly loaded                     7.45 s                       12.91 s
  ──────────────────────────  ──────────────────────  ────────────────────────────
   Warm example                              0.71 s                        4.16 s
  ──────────────────────────  ──────────────────────  ────────────────────────────
   Warm after startup warm-    approximately 0.53 s          approximately 3.07 s
   up

  The program now warms Qwen before displaying:

  SPEAK NOW

  This moves the loading delay to application startup instead of making the user
  wait after asking the first question.

  ## 9. Automatic warm-up tests

  Two tests were performed after automatic warm-up was added.

   Measurement                     Test 1    Test 2    Average
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━  ━━━━━━━━  ━━━━━━━━━
   Utterance duration              1.88 s    2.75 s     2.32 s
  ──────────────────────────────  ────────  ────────  ─────────
   Whisper inference               0.42 s    0.45 s     0.44 s
  ──────────────────────────────  ────────  ────────  ─────────
   Speech start → text             3.15 s    4.18 s     3.67 s
  ──────────────────────────────  ────────  ────────  ─────────
   Speech end → text               1.28 s    1.43 s     1.36 s
  ──────────────────────────────  ────────  ────────  ─────────
   Qwen response                   0.52 s    0.54 s     0.53 s
  ──────────────────────────────  ────────  ────────  ─────────
   Piper synthesis                 1.18 s    1.19 s     1.19 s
  ──────────────────────────────  ────────  ────────  ─────────
   Whisper text → speaker start    1.71 s    1.74 s     1.73 s
  ──────────────────────────────  ────────  ────────  ─────────
   Speech end → speaker start      2.98 s    3.16 s     3.07 s
  ──────────────────────────────  ────────  ────────  ─────────
   Playback duration               2.31 s    2.57 s     2.44 s

  ### Primary result before continuous mode

  After the user finished speaking, the robot began answering after approximately:

  3.07 seconds on average

  The two questions were different, so these results demonstrate normal operation
  rather than a strict same-sentence comparison.

  ## 10. Ollama CPU versus Vulkan comparison

  Ollama initially used a mixed CPU/GPU Vulkan configuration.

  ### Mixed Vulkan result

  The Ollama log showed:

  26 of 29 model layers offloaded to Vulkan GPU
  approximately 852 MB in Vulkan model memory
  approximately 265 MB in host memory

  Controlled request measurements:

  Total duration: 25.65 seconds
  Prompt processing: 2.27 seconds
  Generation: 23.38 seconds
  Generated tokens: 14
  Generation rate: approximately 0.60 tokens/second

  ### CPU AVX2 result

  Cold CPU request:

  Total duration: 28.22 seconds
  Model loading: 27.26 seconds
  Generation: 0.61 seconds
  Generated tokens: 14

  Warm CPU request:

  Total duration: 0.72 seconds
  Generation rate: approximately 22.65 tokens/second

  ### Comparison

   Configuration                     Generation speed    Controlled request total
  ━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━
   Mixed Vulkan CPU/GPU    approximately 0.60 tokens/                     25.65 s
                                                    s
  ──────────────────────  ────────────────────────────  ──────────────────────────
   CPU AVX2, warm                 approximately 22.65                      0.72 s
                                             tokens/s

  CPU AVX2 produced tokens approximately 38 times faster than the Vulkan
  configuration on this laptop.

  Therefore, Ollama is configured to use CPU AVX2 on this machine.

  The current Ollama service environment includes:

  OLLAMA_LLM_LIBRARY=cpu_avx2
  CUDA_VISIBLE_DEVICES=-1
  GGML_VK_VISIBLE_DEVICES=-1

  ## 11. Continuous-conversation implementation

  The program now runs multiple interactions without restarting.

  The process is:

  Start Whisper once
  → warm Qwen once
  → display SPEAK NOW
  → record one user turn
  → stop microphone
  → Whisper transcription
  → Qwen response
  → Piper synthesis
  → speaker playback
  → display SPEAK NOW again
  → repeat

  The microphone is stopped while Piper speaks. This prevents the robot from
  recording and answering its own voice.

  The conversation can be stopped using:

  - “Goodbye”
  - “Stop conversation”
  - “Exit”
  - “Quit”
  - Ctrl+C

  If no speech is detected for 15 seconds, the program starts listening again.

  ## 12. Continuous-conversation test results

  The first continuous test used four question-and-answer turns followed by the
  “Goodbye” command.

  The initial Qwen warm-up took:

  25.94 seconds

  This was a cold startup. The delay occurred before interaction began.

  ### Per-turn measurements

   Measurement                    Turn 1    Turn 2    Turn 3    Turn 4    Average
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━  ━━━━━━━━  ━━━━━━━━  ━━━━━━━━  ━━━━━━━━━
   Utterance duration             1.50 s    2.38 s    3.75 s    1.75 s     2.35 s
  ─────────────────────────────  ────────  ────────  ────────  ────────  ─────────
   Whisper inference              0.43 s    0.47 s    0.46 s    0.43 s     0.45 s
  ─────────────────────────────  ────────  ────────  ────────  ────────  ─────────
   Speech start → text            2.78 s    3.82 s    5.07 s    3.03 s     3.68 s
  ─────────────────────────────  ────────  ────────  ────────  ────────  ─────────
   Speech end → text              1.28 s    1.45 s    1.32 s    1.28 s     1.33 s
  ─────────────────────────────  ────────  ────────  ────────  ────────  ─────────
   Qwen response                  0.45 s    0.64 s    0.46 s    0.67 s     0.56 s
  ─────────────────────────────  ────────  ────────  ────────  ────────  ─────────
   Piper synthesis                2.37 s    1.20 s    1.18 s    1.20 s     1.49 s
  ─────────────────────────────  ────────  ────────  ────────  ────────  ─────────
   Whisper text → speaker         2.82 s    1.84 s    1.64 s    1.87 s     2.04 s
   start
  ─────────────────────────────  ────────  ────────  ────────  ────────  ─────────
   Speech end → speaker start     4.10 s    3.29 s    2.96 s    3.15 s     3.38 s
  ─────────────────────────────  ────────  ────────  ────────  ────────  ─────────
   Playback duration              2.06 s    2.32 s    1.80 s    2.82 s     2.25 s

  The “Goodbye” exit-command turn was excluded from these averages.

  ### Main continuous result

  During continuous interaction, the robot began answering approximately:

  3.38 seconds after the user finished speaking

  Qwen’s average warm response time was:

  0.56 seconds

  ## 13. Continuous-test interaction results

  ### Turn 1

  Transcription:

  What is the capital of Germany?

  Qwen response:

  The capital of Germany is Berlin.

  Result: clear transcription and appropriate response.

  ### Turn 2

  Intended question:

  How many states does Germany have?

  Whisper transcription:

  How many stairs does Germany have?

  Qwen response:

  Germany has 14,000 stairs.

  Result: Whisper misrecognized “states” as “stairs.” Qwen trusted the incorrect
  transcription and generated an unreliable response.

  ### Turn 3

  Transcription:

  Sorry, sorry, sorry, how many states does Germany have?

  Qwen response:

  Germany has 16 states.

  Result: the repeated/corrected question was understood.

  ### Turn 4

  Transcription:

  Who is the president of Germany?

  Qwen produced a response from its internal knowledge.

  Result: the complete pipeline worked, but offline LLM answers about current
  public figures should not automatically be treated as up-to-date.

  ### Exit turn

  Transcription:

  Goodbye.

  Program output:

  Exit command detected.
  Conversation ended.

  Result: spoken exit control worked correctly.

  ## 14. Accuracy and reliability observations

  ### Successful behaviour

  - Live six-channel ReSpeaker recording works.
  - Faster Whisper produces usable English transcriptions.
  - Qwen generates short responses.
  - Piper produces intelligible speech.
  - Continuous interaction works without restarting.
  - The microphone does not record Piper’s output.
  - The spoken “Goodbye” command terminates the conversation.
  - Warm Qwen generation is fast on CPU AVX2.

  ### Error propagation

  The most important failure was:

  User says “states”
  → Whisper detects “stairs”
  → Qwen trusts “stairs”
  → Piper speaks the incorrect answer

  This demonstrates that errors can propagate through the complete pipeline:

  STT error
  → LLM error
  → spoken misinformation

  A future version should detect uncertain or unclear transcriptions and ask the
  user to repeat the question instead of answering confidently.

  ### Current-information limitation

  Qwen is an offline model. It has no automatic access to current internet
  information.

  Therefore:

  - Current political information may be outdated.
  - Questions containing “now,” “today” or “current” require caution.
  - A live knowledge source or retrieval system would be required for reliable
    current information.

  ## 15. Current limitations

  - Qwen does not remember previous conversation turns.
  - Each question is currently processed independently.
  - STT confidence is not checked before sending text to Qwen.
  - The system does not ask for clarification after uncertain transcription.
  - The current threshold uses only ReSpeaker channel 0.
  - The microphone is recalibrated for one second before every turn.
  - Piper generates the complete WAV file before playback begins.
  - Generated speech is not streamed.
  - The first cold Qwen startup can take approximately 26 seconds.
  - Camera and eye-contact activation are not integrated.
  - Atul’s camera/eye-contact repository is not currently available.
  - The Social Marketplace architecture has been studied but is currently parked.
  - Unitree integration is intentionally reserved for a later stage.

  ## 16. Work completed

  - Verified Ubuntu and ROS installation
  - Inspected llm_based_speech_interaction
  - Identified available STT implementations
  - Selected Faster Whisper
  - Tested saved-audio transcription
  - Tested laptop microphone transcription
  - Tested ReSpeaker recording
  - Tested six-channel ReSpeaker audio
  - Implemented speech detection and endpoint detection
  - Measured Whisper latency
  - Installed Piper safely inside the Python virtual environment
  - Downloaded and tested a Piper voice
  - Measured Piper synthesis
  - Implemented STT-to-Piper speech echo
  - Installed and tested Ollama
  - Downloaded Qwen2.5 1.5B
  - Measured Qwen generation speed
  - Compared Vulkan and CPU AVX2
  - Selected CPU AVX2 based on measured performance
  - Connected Whisper, Qwen and Piper
  - Added automatic Qwen warm-up
  - Measured the complete speech-to-speech pipeline
  - Implemented continuous interaction
  - Implemented spoken exit commands
  - Tested multiple consecutive interactions

  ## 17. Current status

  ReSpeaker capture                         ✅
  Voice/end-of-speech detection             ✅
  Live Faster Whisper transcription         ✅
  Whisper latency measurement               ✅
  Local Qwen response generation            ✅
  Piper speech generation                    ✅
  Speaker playback                           ✅
  Complete speech-to-speech pipeline         ✅
  Automatic LLM warm-up                      ✅
  Continuous listen-and-answer loop          ✅
  Spoken conversation exit                   ✅
  Conversation memory                        ❌
  STT-error confirmation                     ❌
  Camera/eye-contact activation              ⏳ waiting for repository
  Social Marketplace runtime integration     ⏳ parked
  Unitree integration                        ⏳ much later

  ## 18. Questions for the next professor meeting

  1. Can I receive Atul’s camera/eye-contact repository?
  2. Is approximately 3.38 seconds of warm continuous response latency acceptable?
  3. Should the next optimisation focus on speech endpoint detection, Piper
     synthesis or STT accuracy?

  4. Should uncertain transcriptions cause the robot to request repetition?
  5. Is conversation memory required for the next milestone?
  6. Should the offline system answer current-information questions, or should it
     explicitly state that current information is unavailable?

  7. Should the next stage integrate multimodal activation before further speech-
     pipeline improvements?

