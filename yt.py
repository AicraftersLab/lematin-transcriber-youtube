import streamlit as st
import openai
import yt_dlp
import os
import re
import subprocess
import shutil
from pytube import YouTube
from moviepy import AudioFileClip

# Configurer la clé API OpenAI via Streamlit secrets
openai.api_key = st.secrets.get("OPENAI_API_KEY")

# S'assurer que le dossier temporaire existe
def ensure_temp_dir():
    os.makedirs("temp", exist_ok=True)

# Vérifier si FFmpeg est disponible
def has_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None

# Normaliser les URL YouTube
def standardize_youtube_url(url: str) -> str:
    if "youtu.be/" in url:
        video_id = url.split("youtu.be/")[-1].split("?")[0]
        return f"https://www.youtube.com/watch?v={video_id}"
    if "youtube.com/watch" in url:
        m = re.search(r"v=([a-zA-Z0-9_-]{11})", url)
        if m:
            return f"https://www.youtube.com/watch?v={m.group(1)}"
    return url

# Télécharger et convertir l'audio avec plusieurs fallbacks
def download_and_convert_audio(video_url: str, audio_format="mp3", retries=2) -> str | None:
    ensure_temp_dir()
    url = standardize_youtube_url(video_url)
    st.info(f"🔗 URL standardisée : {url}")

    if not has_ffmpeg():
        st.error("FFmpeg est requis mais n’est pas installé. Installez-le (apt-get install ffmpeg ou brew install ffmpeg).")
        return None

    # Tentatives avec yt_dlp (API Python), en changeant de format si besoin
    formats_to_try = [None, "140", "251"]  # None équivaut à 'bestaudio/best'
    last_error = None
    for fmt in formats_to_try:
        for attempt in range(1, retries + 1):
            opts = {
                "format": fmt if fmt else "bestaudio/best",
                "postprocessors": [{
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": audio_format,
                    "preferredquality": "192",
                }],
                "outtmpl": "temp/audio.%(ext)s",
                "noplaylist": True,
                "quiet": True,
                "ignoreerrors": True,
                "nocheckcertificate": True,
                "geo_bypass": True,
                "http_headers": {
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/120.0.0.0 Safari/537.36"
                    )
                },
            }
            try:
                with yt_dlp.YoutubeDL(opts) as ydl:
                    info = ydl.extract_info(url, download=True)
                    if info:
                        output_path = f"temp/audio.{audio_format}"
                        if os.path.exists(output_path):
                            return output_path
            except Exception as e:
                last_error = e
                st.warning(f"⚠️ Échec yt_dlp (format {fmt or 'best'}) tentative {attempt}/{retries} : {e}")

    # Fallback avec pytube
    try:
        st.info("⏳ Fallback avec pytube…")
        yt = YouTube(url)
        stream = yt.streams.filter(only_audio=True).first()
        if stream:
            fallback_path = "temp/audio_fallback.mp4"
            stream.download(filename=fallback_path)
            audio_clip = AudioFileClip(fallback_path)
            final_path = "temp/audio_fallback.mp3"
            audio_clip.write_audiofile(final_path)
            audio_clip.close()
            os.remove(fallback_path)
            return final_path
    except Exception as e:
        st.warning(f"⚠️ Fallback pytube échoué : {e}")

    # Dernier recours : yt-dlp en CLI
    try:
        st.info("⏳ Dernier recours : yt-dlp CLI…")
        final_template = "temp/audio_cli.%(ext)s"
        cmd = [
            "yt-dlp",
            "-x",
            "--audio-format", audio_format,
            "--no-check-certificate",
            "--geo-bypass",
            "-o", final_template,
            url
        ]
        subprocess.run(cmd, check=True)
        # Remplacer le template par le format final
        cli_final = final_template.replace("%(ext)s", audio_format)
        if os.path.exists(cli_final):
            return cli_final
        # Chercher un fichier compatible dans temp/
        for f in os.listdir("temp"):
            if f.startswith("audio_cli.") and f.endswith(f".{audio_format}"):
                return os.path.join("temp", f)
    except Exception as e:
        st.error(f"❌ Échec complet (yt_dlp + pytube + CLI) : {e}")
        return None

    st.error(f"❌ Impossible de télécharger la vidéo (dernière erreur : {last_error})")
    return None

# Diviser l’audio en morceaux fixes (1 minute)
def split_audio(audio_path: str, chunk_length=60) -> list[str]:
    ensure_temp_dir()
    audio = AudioFileClip(audio_path)
    chunks = []
    duration = int(audio.duration)
    for i in range(0, duration, chunk_length):
        chunk_path = f"temp/chunk_{i}.mp3"
        audio.subclipped(i, min(i + chunk_length, duration)).write_audiofile(chunk_path, logger=None)
        chunks.append(chunk_path)
    audio.close()
    return chunks

# Transcrire un fichier audio avec Whisper-1 en respectant la langue parlée
WHISPER_PROMPT = (
    "Transcribe the audio exactly in the spoken language of the speaker. "
    "If the audio contains multiple languages (e.g., English, French, Arabic, or others), "
    "switch dynamically and write each segment in its original spoken language without translation. "
    "Do not normalize or convert languages; preserve the natural mix exactly as spoken."
)

def transcribe_audio(audio_chunk_path: str) -> str:
    with open(audio_chunk_path, "rb") as audio_file:
        result = openai.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format="text",
            prompt=WHISPER_PROMPT
        )
    return result if isinstance(result, str) else str(result)

# Interface Streamlit simplifiée
st.title("🎙️ Transcripteur de Vidéos YouTube (multi-langue)")
st.write("Entrez un lien YouTube à transcrire. La transcription reflètera chaque langue telle que parlée (ar, fr, en, …).")

video_url = st.text_input("🔗 Lien YouTube", "")

if st.button("Transcrire la vidéo"):
    if not openai.api_key:
        st.error("⚠️ OPENAI_API_KEY manquante dans les secrets Streamlit.")
    elif not video_url.strip():
        st.error("❌ Veuillez entrer un lien YouTube valide.")
    else:
        st.info("📥 Téléchargement de l'audio…")
        audio_path = download_and_convert_audio(video_url)
        if audio_path:
            st.success("✅ Audio téléchargé.")
            st.info("✂️ Découpage de l'audio en segments de 60 secondes…")
            audio_chunks = split_audio(audio_path)
            st.info("📝 Transcription des segments…")
            full_transcript = []
            for chunk in audio_chunks:
                full_transcript.append(transcribe_audio(chunk))
                try:
                    os.remove(chunk)
                except Exception:
                    pass
            final_text = "\n".join(full_transcript).strip()
            st.success("✅ Transcription terminée !")
            st.text_area("📜 Texte transcrit", final_text, height=350)
            st.download_button("⬇️ Télécharger la transcription", final_text, file_name="transcription.txt")
            try:
                os.remove(audio_path)
            except Exception:
                pass
