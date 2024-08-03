from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .transcription_service import TranscriptionService
import tempfile
import os

class TranscribeAudioView(APIView):
    def post(self, request):
        audio_file = request.FILES.get('audio_file')
        if not audio_file:
            return Response({"error": "No audio file provided"}, status=status.HTTP_400_BAD_REQUEST)

        # Save the uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            for chunk in audio_file.chunks():
                temp_file.write(chunk)
            temp_file_path = temp_file.name

        try:
            # Process the audio file
            transcription_service = TranscriptionService()
            results = transcription_service.transcribe_audio(temp_file_path)

            # Format the results
            formatted_results = []
            for speaker, text in results:
                formatted_results.append({
                    "speaker": speaker,
                    "text": text
                })

            return Response({"transcription": formatted_results}, status=status.HTTP_200_OK)
        finally:
            # Clean up the temporary file
            os.unlink(temp_file_path)