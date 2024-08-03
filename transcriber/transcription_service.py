import torch
from pyannote.audio import Pipeline
import whisper
import re

class TranscriptionService:
    def __init__(self):
        self.hugging_face_token = "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"
        self.pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=self.hugging_face_token)
        self.whisper_model = whisper.load_model("base")

    def transcribe_audio(self, audio_file_path):
        diarization = self.pipeline(audio_file_path)

        results = []
        speaker_names = {}

        for turn, _, speaker in diarization.itertracks(yield_label=True):
            transcription_text = self.transcribe_segment(audio_file_path, turn.start, turn.end)
            
            if speaker not in speaker_names:
                name = self.extract_name(transcription_text)
                if name:
                    speaker_names[speaker] = name
                else:
                    speaker_names[speaker] = f"Speaker {len(speaker_names) + 1}"

            results.append((speaker_names[speaker], transcription_text))

        return results

    def transcribe_segment(self, audio_file, start, end):
        audio = whisper.load_audio(audio_file)
        audio_segment = audio[int(start * 16000):int(end * 16000)]
        result = self.whisper_model.transcribe(audio_segment, language="english")
        return result["text"]

    def extract_name(self, introduction):
        patterns = [r"my name is (\w+)", r"i am (\w+)", r"this is (\w+)"]
        for pattern in patterns:
            match = re.search(pattern, introduction, re.IGNORECASE)
            if match:
                return match.group(1)
        return None