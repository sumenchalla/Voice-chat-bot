import pyaudio
import wave
import streamlit as st
import os
from src.LLM_integration import model,tokenizer
from src.Speach_recognition import model_spe
import whisper
import torchaudio
from gtts import gTTS
from utlies.support import vb_collection
import pygame
import time
from pydub import AudioSegment


def record_audio(output_file, sample_rate=16000, duration=5):
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1

    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=sample_rate,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("Recording...")
    st.write('<i class="bi bi-mic"></i> Recording...', unsafe_allow_html=True)
    frames = []

    # Record audio
    for _ in range(0, int(sample_rate / CHUNK * duration)):
        
        audio_data = stream.read(CHUNK)
        frames.append(audio_data)

    print("Finished recording.")
    st.write('<i class="bi bi-mic"></i> Recordingccompleted', unsafe_allow_html=True)

    # Stop stream
    stream.stop_stream()
    stream.close()
    p.terminate()

    # Save audio to file
    wf = wave.open(output_file, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(sample_rate)
    wf.writeframes(b''.join(frames))
    wf.close()



st.title("Audio chatbot")
chat =False
 
st.write("You you have ask a question regarding Indain states and record button will be active for next 5 seconds")
start = st.button("Start")
stop = st.button("Stop")

if start:
    chat =True
    while chat:
        file_path = "recorded_audio.wav"
        record_audio(file_path)
        #print("Generated Summary:", summary)
        st.spinner("loading...")
        waveform, sample_rate = torchaudio.load("recorded_audio.wav")
        audio = whisper.pad_or_trim(waveform.flatten()).to("cuda")
        mel = whisper.log_mel_spectrogram(audio)

        options = whisper.DecodingOptions( without_timestamps=True,language="en")

        results=model_spe.decode(mel,options)
        question =results.text
        context = vb_collection.query(
            query_texts=f"{question}",
            n_results=1)
        input_text = f"""
            Based on the below data,
            answer this question {question}.
            and the data is {context["documents"]}
        """

        # Tokenize input text
        inputs = tokenizer(input_text, max_length=1024, return_tensors="pt", truncation=True)

        # Generate summary
        summary_ids = model.generate(inputs["input_ids"], max_length=150, min_length=40, num_beams=4, early_stopping=True)

        # Decode and print the summary
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        myobj = gTTS(text=summary, lang="en", slow=False)
        myobj.save("welcome.mp3")
        pygame.mixer.init()
        # Load the audio file
        pygame.mixer.music.load("welcome.mp3")

        # Play the audio file
        pygame.mixer.music.play()
        # Replace with the path to your audio file
        while pygame.mixer.music.get_busy():
            time.sleep(1)
        # Get the duration of the audio file in seconds
            
        pygame.mixer.quit()
if stop:
    chat =False

