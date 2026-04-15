import logging
from faster_whisper import WhisperModel
from app.core.config import settings

logger = logging.getLogger(__name__)

class STTEngine:
    def __init__(self):
        # We load base model on CPU for high compatibility, but allow configuration if needed.
        self.model_size = settings.WHISPER_MODEL_SIZE
        try:
            logger.info(f"Loading faster-whisper model: {self.model_size}")
            # To be broadly compatible, defaulting to compute_type="int8"
            self.model = WhisperModel(self.model_size, device="cpu", compute_type="int8")
            logger.info("STT Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading STT model: {str(e)}")
            self.model = None

    def transcribe(self, audio_path: str) -> str:
        """
        Transcribe the audio file using faster-whisper.
        """
        if not self.model:
            return "Error: STT model failed to load. Cannot transcribe."
        
        try:
            segments, info = self.model.transcribe(audio_path, beam_size=5)
            text = " ".join([segment.text for segment in segments])
            return text.strip()
        except Exception as e:
            logger.error(f"Transcription failed: {str(e)}")
            return f"Error during transcription: {str(e)}"

stt_engine = STTEngine()
