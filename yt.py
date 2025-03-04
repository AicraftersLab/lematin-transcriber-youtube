import streamlit as st
import openai
import yt_dlp
import os
from moviepy import AudioFileClip

# Clé API OpenAI (à configurer via les variables d'environnement)
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Fonction pour télécharger et convertir l'audio depuis YouTube
def download_and_convert_audio(video_url, audio_format="mp3"):
    try:
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': audio_format}],
            'outtmpl': 'temp/audio.%(ext)s',
            'noplaylist': True
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])
        return "temp/audio.mp3"
    except Exception as e:
        st.error(f"Erreur de téléchargement : {e}")
        return None

# def download_and_convert_audio(video_url, audio_format="mp3"):
#     try:
#         ydl_opts = {
#             'format': 'bestaudio/best',
#             'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': audio_format}],
#             'outtmpl': 'temp/audio.%(ext)s',
#             'noplaylist': True,
#             'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
#             'nocheckcertificate': True  # Avoids SSL certificate issues
#         }
        
#         with yt_dlp.YoutubeDL(ydl_opts) as ydl:
#             ydl.download([video_url])
        
#         return "temp/audio.mp3"

#     except yt_dlp.utils.DownloadError as e:
#         st.error(f"Erreur de téléchargement : {e}")
#         return None


# Fonction pour diviser l'audio en morceaux
def split_audio(audio_path, chunk_length=60):
    audio = AudioFileClip(audio_path)
    chunks = []
    duration = int(audio.duration)
    
    for i in range(0, duration, chunk_length):
        chunk_path = f"temp/chunk_{i}.mp3"
        audio.subclipped(i, min(i + chunk_length, duration)).write_audiofile(chunk_path)
        chunks.append(chunk_path)
    
    return chunks

# Fonction pour transcrire un fichier audio
def transcribe_audio(audio_chunk_path):
    
    with open(audio_chunk_path, "rb") as audio_file:
        transcription = openai.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format="text",
        )
    return transcription


# Interface Streamlit
st.title("🎙️ Transcripteur de Vidéos YouTube Le Matin")
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
                full_transcript += transcribe_audio(chunk)
                os.remove(chunk)  # Nettoyage
            
            st.success("✅ Transcription terminée !")
            st.text_area("📜 Texte Transcrit", full_transcript, height=300)
            st.download_button("⬇️ Télécharger la transcription", full_transcript, file_name="transcription.txt")

            os.remove(audio_path)  # Nettoyage
    else:
        st.error("❌ Veuillez entrer un lien YouTube valide.")
