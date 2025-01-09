import argparse
import io
import os
import speech_recognition as sr
from datetime import datetime, timedelta
from queue import Queue
from tempfile import NamedTemporaryFile
from time import sleep
from sys import platform
from faster_whisper import WhisperModel
import pyautogui  # Import pyautogui for key press simulation
from pynput.mouse import Controller, Button
mouse = Controller()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="tiny", help="Model to use",
                        choices=["tiny", "base", "small", "medium", "large"])
    parser.add_argument("--device", default="auto", help="Device for inference",
                        choices=["auto", "cuda", "cpu"])
    parser.add_argument("--compute_type", default="auto", help="Type of quantization to use",
                        choices=["auto", "int8", "int8_floatt16", "float16", "int16", "float32"])
    parser.add_argument("--threads", default=0, help="Number of threads for CPU inference", type=int)
    parser.add_argument("--energy_threshold", default=1000, help="Energy level for mic to detect.", type=int)
    parser.add_argument("--record_timeout", default=2, help="How real-time the recording is in seconds.", type=float)
    parser.add_argument("--phrase_timeout", default=3, help="Pause duration to consider a new phrase.", type=float)

    if 'linux' in platform:
        parser.add_argument("--default_microphone", default='pulse',
                            help="Default microphone name. Use 'list' to view available microphones.", type=str)

    args = parser.parse_args()

    # State variables
    phrase_time = None
    last_sample = bytes()
    data_queue = Queue()

    recorder = sr.Recognizer()
    recorder.energy_threshold = args.energy_threshold
    recorder.dynamic_energy_threshold = False

    # Microphone setup
    if 'linux' in platform:
        mic_name = args.default_microphone
        if not mic_name or mic_name == 'list':
            print("Available microphone devices are: ")
            for index, name in enumerate(sr.Microphone.list_microphone_names()):
                print(f"Microphone with name \"{name}\" found")
            return
        else:
            for index, name in enumerate(sr.Microphone.list_microphone_names()):
                if mic_name in name:
                    source = sr.Microphone(sample_rate=16000, device_index=index)
                    break
    else:
        source = sr.Microphone(sample_rate=16000)

    # Load Whisper model
    if args.model == "large":
        args.model = "large-v2"
    model_name = args.model
    device = args.device
    compute_type = args.compute_type if device != "cpu" else "int8"
    cpu_threads = args.threads
    audio_model = WhisperModel(model_name, device=device, compute_type=compute_type, cpu_threads=cpu_threads)

    temp_file = NamedTemporaryFile().name
    transcription = ""

    with source:
        recorder.adjust_for_ambient_noise(source)

    def record_callback(_, audio: sr.AudioData) -> None:
        data = audio.get_raw_data()
        data_queue.put(data)

    recorder.listen_in_background(source, record_callback, phrase_time_limit=args.record_timeout)

    print("Model loaded. Start speaking.\n")

    while True:
        try:
            now = datetime.utcnow()
            if not data_queue.empty():
                phrase_complete = False
                if phrase_time and now - phrase_time > timedelta(seconds=args.phrase_timeout):
                    last_sample = bytes()
                    phrase_complete = True

                phrase_time = now
                while not data_queue.empty():
                    data = data_queue.get()
                    last_sample += data

                audio_data = sr.AudioData(last_sample, source.SAMPLE_RATE, source.SAMPLE_WIDTH)
                wav_data = io.BytesIO(audio_data.get_wav_data())

                with open(temp_file, 'w+b') as f:
                    f.write(wav_data.read())

                text = ""
                segments, _ = audio_model.transcribe(temp_file)
                for segment in segments:
                    text += segment.text.strip()

                if phrase_complete:
                    transcription += "\n" + text
                else:
                    transcription += text

                print(f"\r{text}", end="", flush=True)

                # Check if the transcribed text contains specific words and simulate key presses
                if "left" in text.lower():
                    pyautogui.keyDown("a")
                    sleep(0.1)
                    pyautogui.keyUp("a")
                elif "right" in text.lower():
                    pyautogui.keyDown("d")
                    sleep(0.1)
                    pyautogui.keyUp("d")  # Simulate "D" key press for "right"
                elif "jump" in text.lower():
                    pyautogui.press("space")  # Simulate "W" key press for "up"

                sleep(0.75)
        except KeyboardInterrupt:
            break

    print("\n\nFinal Transcription:")
    print(transcription)

if __name__ == "__main__":
    main()
