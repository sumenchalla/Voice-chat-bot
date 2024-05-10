import whisper
import torchaudio


model_spe = whisper.load_model("base.en")
waveform, sample_rate = torchaudio.load("recorded_audio.wav")
audio = whisper.pad_or_trim(waveform.flatten()).to("cuda")
mel = whisper.log_mel_spectrogram(audio)

options = whisper.DecodingOptions( without_timestamps=True,language="en")

results=model_spe.decode(mel,options)