"""Audio/video -> text loading for RAG over spoken content.

Two paths, both native (no hand-rolled transcription math):

1. Preferred, fully local: LlamaIndex's built-in `VideoAudioReader`
   (llama_index.readers.file), which runs local OpenAI Whisper (and, for
   .mp4, extracts the audio track with pydub first). Enabled by installing
   requirements-multimodal.txt.
2. Default fallback, free-tier remote: the Hugging Face Inference API's
   automatic-speech-recognition pipeline (Whisper, `HUGGINGFACE_API_KEY`),
   so the demo works with only the core requirements.txt installed.

Video files are transcribed the same way as audio files: both readers
extract/consume the audio track and return the spoken-word transcript as
the Document text.
"""

from __future__ import annotations

import os
from pathlib import Path

from llama_index.core.schema import Document

AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".flac"}
VIDEO_EXTS = {".mp4", ".mov", ".mkv"}

DEFAULT_HF_ASR_MODEL = "openai/whisper-large-v3"


def _transcribe_locally(path: Path) -> str | None:
    try:
        from llama_index.readers.file import VideoAudioReader
    except ImportError:
        return None

    try:
        reader = VideoAudioReader()
        docs = reader.load_data(file=path)
    except ImportError:
        # whisper / pydub not installed.
        return None

    return docs[0].text if docs else None


def _transcribe_via_hf_inference(path: Path) -> str | None:
    token = os.environ.get("HUGGINGFACE_API_KEY")
    if not token:
        return None

    from huggingface_hub import InferenceClient

    client = InferenceClient(api_key=token)
    result = client.automatic_speech_recognition(str(path), model=DEFAULT_HF_ASR_MODEL)
    return getattr(result, "text", None) or str(result)


def load_audio_video(directory: str | Path) -> list[Document]:
    """Transcribe every audio/video file in `directory` into a Document."""
    directory = Path(directory)
    documents: list[Document] = []

    for path in sorted(directory.iterdir()) if directory.exists() else []:
        suffix = path.suffix.lower()
        if suffix not in AUDIO_EXTS and suffix not in VIDEO_EXTS:
            continue

        transcript = _transcribe_locally(path) or _transcribe_via_hf_inference(path)
        if not transcript:
            continue

        documents.append(
            Document(
                text=transcript,
                metadata={
                    "modality": "video" if suffix in VIDEO_EXTS else "audio",
                    "file_name": path.name,
                    "file_path": str(path),
                },
            )
        )

    return documents
