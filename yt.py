import os
import re
import subprocess
import streamlit as st
import yt_dlp
from pytube import YouTube
from moviepy import AudioFileClip


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
# Fonction principale robuste
# ---------------------------
def download_and_convert_audio(video_url: str, audio_format="mp3", retries=2) -> str:
    url = standardize_youtube_url(video_url)
    st.info(f"🔗 URL standardisée : {url}")

    # --- 1) yt_dlp (fortifié avec headers + cookiesfrombrowser) ---
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
        "cookiesfrombrowser": ("chrome",),  # essaie d'utiliser les cookies du navigateur local
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

    # --- 2) Fallback pytube ---
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
        final_path = "temp/audio_cli.mp3"
        cmd = [
            "yt-dlp",
            "-x",
            "--audio-format", audio_format,
            "-o", final_path,
            url
        ]
        subprocess.run(cmd, check=True)
        if os.path.exists(final_path):
            return final_path
    except Exception as e:
        st.error(f"❌ Échec complet (yt_dlp + pytube + CLI) : {e}")
        return None
