# Audio / video sample data

Drop any `.mp3`, `.wav`, `.m4a`, `.flac`, `.mp4`, `.mov`, or `.mkv` file here
and `src/loaders/audio_video_loader.py` will transcribe it (local Whisper if
`requirements-multimodal.txt` is installed, otherwise the free Hugging Face
Inference API) and index the transcript like any other document.

No sample audio/video file is committed to this repo (binary media bloats
git history for little benefit) — supply your own, e.g. a short voice memo
or a podcast clip, to try the pipeline end to end.
