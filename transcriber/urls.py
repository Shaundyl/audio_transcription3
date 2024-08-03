from django.urls import path
from .views import TranscribeAudioView

urlpatterns = [
    path('transcribe/', TranscribeAudioView.as_view(), name='transcribe'),
]