import streamlit as st
import openai
import yt_dlp
import os
import re
import subprocess
from pytube import YouTube
from moviepy import AudioFileClip

# Clé API OpenAI
openai.api_key = st.secrets["OPENAI_API_KEY"]

# ---------------------------
# Fonction utilitaire : standardiser l’URL YouTube
# ---------------------------
def standardize_youtube_url(url: str) -> str:
    if "youtu.be/" in url:
        video_id = url.split("youtu.be/")[-1].split("?")[0]
        return f"https://www.youtube.com/watch?v={video_id}"

    if "youtube.com/watch" in url:
        video_id_match = re.search(r"v=([a-zA-Z0-9_-]{11})", url)
        if video_id_match:
            return f"https://www.youtube.com/watch?v={video_id_match.group(1)}"

    return url

# ---------------------------
# Fonction principale : téléchargement robuste avec fallbacks
# ---------------------------
def download_and_convert_audio(video_url: str, audio_format="mp3", retries=2) -> str:
    url = standardize_youtube_url(video_url)
    st.info(f"🔗 URL standardisée : {url}")

    # --- 1) yt_dlp avec headers + cookiesfrombrowser ---
    ydl_opts = {
        "format": "bestaudio/best",
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": audio_format,
            "preferredquality": "192",
        }],
        "outtmpl": f"temp/audio.%(ext)s",
        "noplaylist": True,
        "quiet": True,
        "ignoreerrors": True,
        "http_headers": {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        },
        "cookiesfrombrowser": ("chrome",),  # utiliser cookies Chrome si dispo
    }

    last_error = None
    for attempt in range(retries):
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                if info:
                    output_path = f"temp/audio.{audio_format}"
                    if os.path.exists(output_path):
                        return output_path
        except Exception as e:
            last_error = e
            st.warning(f"⚠️ Tentative yt_dlp {attempt+1} échouée : {e}")

    # --- 2) Fallback avec pytube ---
    try:
        st.info("⏳ Fallback avec pytube...")
        yt = YouTube(url)
        stream = yt.streams.filter(only_audio=True).first()
        fallback_path = "temp/audio_fallback.mp4"
        stream.download(filename=fallback_path)

        audio_clip = AudioFileClip(fallback_path)
        final_path = "temp/audio_fallback.mp3"
        audio_clip.write_audiofile(final_path)
        audio_clip.close()
        os.remove(fallback_path)
        return final_path
    except Exception as e:
        st.warning(f"⚠️ Fallback pytube échoué : {e}")

    # --- 3) Fallback CLI yt-dlp ---
    try:
        st.info("⏳ Dernier recours : yt-dlp CLI...")
        final_path = "temp/audio_cli.%(ext)s"
        cmd = [
            "yt-dlp",
            "-x",
            "--audio-format", audio_format,
            "-o", final_path,
            url
        ]
        subprocess.run(cmd, check=True)
        cli_final = final_path.replace("%(ext)s", audio_format)
        if os.path.exists(cli_final):
            return cli_final
    except Exception as e:
        st.error(f"❌ Échec complet (yt_dlp + pytube + CLI) : {e}")
        return None

# ---------------------------
# Fonction pour diviser l'audio en morceaux
# ---------------------------
def split_audio(audio_path, chunk_length=60):
    audio = AudioFileClip(audio_path)
    chunks = []
    duration = int(audio.duration)
    
    for i in range(0, duration, chunk_length):
        chunk_path = f"temp/chunk_{i}.mp3"
        audio.subclipped(i, min(i + chunk_length, duration)).write_audiofile(chunk_path)
        chunks.append(chunk_path)
    
    return chunks

# ---------------------------
# Fonction pour transcrire un fichier audio
# ---------------------------
def transcribe_audio(audio_chunk_path):
    with open(audio_chunk_path, "rb") as audio_file:
        transcription = openai.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format="text",
            prompt="Transcribe the audio exactly in the spoken language of the speaker. If the audio contains multiple languages (e.g., English, French, Arabic, or others), switch dynamically and write each segment in its original spoken language without translation. Do not normalize or convert languages; preserve the natural mix exactly as spoken."
        )
    return transcription

# ---------------------------
# Interface Streamlit
# ---------------------------
st.title("🎙️ Transcripteur de Vidéos YouTube")
st.write("Entrez un lien YouTube pour obtenir la transcription de la vidéo.")

video_url = st.text_input("🔗 Entrez le lien YouTube ici", "")

if st.button("Transcrire la vidéo"):
    if video_url:
        st.info("📥 Téléchargement de l'audio...")
        audio_path = download_and_convert_audio(video_url)
        
        if audio_path:
            st.info("🎙️ Découpage de l'audio...")
            audio_chunks = split_audio(audio_path, chunk_length=60)

            st.info("📝 Transcription en cours...")
            full_transcript = ""
            for chunk in audio_chunks:
                full_transcript += transcribe_audio(chunk) + "\n"
                os.remove(chunk)  # Nettoyage
            
            st.success("✅ Transcription terminée !")
            st.text_area("📜 Texte Transcrit", full_transcript, height=300)
            st.download_button("⬇️ Télécharger la transcription", full_transcript, file_name="transcription.txt")

            os.remove(audio_path)  # Nettoyage
    else:
        st.error("❌ Veuillez entrer un lien YouTube valide.")
