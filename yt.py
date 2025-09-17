import streamlit as st
import openai
import yt_dlp
import os
import re
import subprocess
import shutil
from pytube import YouTube
from moviepy import AudioFileClip

# =========================
# Config & Helpers
# =========================

# OpenAI API key
openai.api_key = st.secrets.get("OPENAI_API_KEY", None)

def ensure_temp_dir():
    os.makedirs("temp", exist_ok=True)

def standardize_youtube_url(url: str) -> str:
    """Normalize YouTube URLs (youtu.be -> youtube.com & strip junk)."""
    if "youtu.be/" in url:
        video_id = url.split("youtu.be/")[-1].split("?")[0]
        return f"https://www.youtube.com/watch?v={video_id}"

    if "youtube.com/watch" in url:
        m = re.search(r"v=([a-zA-Z0-9_-]{11})", url)
        if m:
            return f"https://www.youtube.com/watch?v={m.group(1)}"

    return url

def has_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None

# =========================
# Download & Convert (robust with fallbacks)
# =========================
def download_and_convert_audio(video_url: str, audio_format="mp3", retries=2, debug=False) -> str | None:
    """
    Returns path to an audio file (mp3) in ./temp or None on failure.
    """
    ensure_temp_dir()
    url = standardize_youtube_url(video_url)
    st.info(f"🔗 URL standardisée : {url}")

    if not has_ffmpeg():
        st.error("FFmpeg introuvable. Installe-le (ex: `apt-get install ffmpeg` ou Homebrew sur macOS).")
        return None

    # 1) Try yt_dlp (Python API) with User-Agent
    ydl_opts = {
        "format": "bestaudio/best",
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": audio_format,
            "preferredquality": "192",
        }],
        "outtmpl": "temp/audio.%(ext)s",
        "noplaylist": True,
        "quiet": not debug,     # show logs if debug
        "verbose": debug,
        "ignoreerrors": True,
        "http_headers": {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        }
    }

    last_error = None
    for attempt in range(1, retries + 1):
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                if info:
                    output_path = f"temp/audio.{audio_format}"
                    if os.path.exists(output_path):
                        return output_path
        except Exception as e:
            last_error = e
            st.warning(f"⚠️ Tentative yt_dlp {attempt}/{retries} échouée : {e}")

    # 2) Fallback: pytube
    try:
        st.info("⏳ Fallback avec pytube...")
        yt = YouTube(url)
        stream = yt.streams.filter(only_audio=True).first()
        if not stream:
            raise RuntimeError("Aucun flux audio disponible via pytube.")
        fallback_path = "temp/audio_fallback.mp4"
        stream.download(filename=fallback_path)

        # Convert to mp3
        audio_clip = AudioFileClip(fallback_path)
        final_path = "temp/audio_fallback.mp3"
        audio_clip.write_audiofile(final_path, logger=None if not debug else "bar")
        audio_clip.close()
        try:
            os.remove(fallback_path)
        except Exception:
            pass
        return final_path
    except Exception as e:
        st.warning(f"⚠️ Fallback pytube échoué : {e}")

    # 3) Final fallback: yt-dlp CLI
    try:
        st.info("⏳ Dernier recours : yt-dlp CLI...")
        # we keep template to capture the right final name after conversion
        final_template = "temp/audio_cli.%(ext)s"
        cmd = [
            "yt-dlp",
            "-x",
            "--audio-format", audio_format,
            "-o", final_template,
            url
        ]
        if debug:
            cmd.append("--verbose")

        # Run
        completed = subprocess.run(cmd, capture_output=debug, text=debug, check=True)
        if debug and completed:
            st.code((completed.stdout or "") + "\n" + (completed.stderr or ""), language="bash")

        final_path = final_template.replace("%(ext)s", audio_format)
        if os.path.exists(final_path):
            return final_path
        else:
            # Fallback guess: sometimes container names slightly differ; list temp files
            for f in os.listdir("temp"):
                if f.startswith("audio_cli.") and f.endswith(f".{audio_format}"):
                    return os.path.join("temp", f)
            raise FileNotFoundError("Sortie CLI yt-dlp introuvable.")
    except subprocess.CalledProcessError as e:
        # Show CLI output if debug
        if debug:
            st.code((e.stdout or "") + "\n" + (e.stderr or ""), language="bash")
        st.error(f"❌ Échec CLI yt-dlp : {e}")
    except Exception as e:
        st.error(f"❌ Échec complet (yt_dlp + pytube + CLI) : {e}")

    return None

# =========================
# Audio Splitting
# =========================
def split_audio(audio_path: str, chunk_length=60) -> list[str]:
    ensure_temp_dir()
    audio = AudioFileClip(audio_path)
    chunks = []
    duration = int(audio.duration)

    for i in range(0, duration, chunk_length):
        chunk_path = f"temp/chunk_{i}.mp3"
        # subclipped(start, end)
        audio.subclipped(i, min(i + chunk_length, duration)).write_audiofile(chunk_path, logger=None)
        chunks.append(chunk_path)

    audio.close()
    return chunks

# =========================
# Transcription (Whisper-1, multilingual as-spoken)
# =========================
WHISPER_PROMPT = (
    "Transcribe the audio exactly in the spoken language of the speaker. "
    "If the audio contains multiple languages (e.g., English, French, Arabic, or others), "
    "switch dynamically and write each segment in its original spoken language without translation. "
    "Do not normalize or convert languages; preserve the natural mix exactly as spoken."
)

def transcribe_audio(audio_chunk_path: str) -> str:
    with open(audio_chunk_path, "rb") as audio_file:
        # response_format="text" returns a raw string in the v1 SDK
        result = openai.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format="text",
            prompt=WHISPER_PROMPT
        )
    # Ensure we return a string (SDK may return PlainText or object-like)
    return result if isinstance(result, str) else str(result)

# =========================
# UI
# =========================
st.set_page_config(page_title="YouTube → Whisper Transcriber", page_icon="🎙️", layout="centered")
st.title("🎙️ Transcripteur de Vidéos YouTube (multi-langue)")

with st.sidebar:
    st.header("⚙️ Options")
    debug = st.checkbox("Activer le mode Debug (yt-dlp verbose)", value=False)
    chunk_len = st.number_input("Taille des morceaux (sec)", min_value=30, max_value=600, value=60, step=10)
    st.caption("Astuce : 60–120s par chunk est souvent un bon compromis.")

st.write("Entrez un lien YouTube à transcrire. La transcription conserve chaque langue telle que parlée (ar, fr, en, …).")
video_url = st.text_input("🔗 Lien YouTube", "")

if st.button("Transcrire la vidéo"):
    if not openai.api_key:
        st.error("⚠️ OPENAI_API_KEY manquant. Ajoute-le dans `st.secrets`.")
    elif not video_url.strip():
        st.error("❌ Veuillez entrer un lien YouTube valide.")
    else:
        st.info("📥 Téléchargement de l'audio…")
        audio_path = download_and_convert_audio(video_url, retries=2, debug=debug)

        if audio_path:
            st.success("✅ Audio téléchargé.")
            try:
                st.info("✂️ Découpage de l'audio…")
                audio_chunks = split_audio(audio_path, chunk_length=int(chunk_len))

                st.info("📝 Transcription en cours…")
                full_transcript = []
                for chunk in audio_chunks:
                    text = transcribe_audio(chunk)
                    full_transcript.append(text)
                    # Clean up each chunk to save space
                    try:
                        os.remove(chunk)
                    except Exception:
                        pass

                final_text = "\n".join(full_transcript).strip()
                st.success("✅ Transcription terminée !")
                st.text_area("📜 Texte Transcrit", final_text, height=350)
                st.download_button("⬇️ Télécharger la transcription", final_text, file_name="transcription.txt")
            finally:
                # Clean original audio
                try:
                    os.remove(audio_path)
                except Exception:
                    pass
