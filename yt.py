import streamlit as st
import openai
import yt_dlp
import os
import tempfile
import shutil
import re
from moviepy import AudioFileClip
from pytube import YouTube
from pydub import AudioSegment

# Clé API OpenAI (à configurer via les variables d'environnement)
openai.api_key = st.secrets["OPENAI_API_KEY"]


# Fonction pour télécharger et convertir l'audio depuis YouTube (wrapper robuste)
def download_and_convert_audio(video_url, audio_format="mp3"):
    try:
        # Utilise la stratégie PyTube -> yt-dlp avec conversion mp3 si nécessaire
        filepath = download_audio(video_url)
        return filepath
    except Exception as e:
        st.error(f"Erreur de téléchargement : {e}")
        return None


# Fonction pour diviser l'audio en morceaux
def split_audio(audio_path, chunk_length=60):
    # Utiliser pydub pour un découpage fiable, en millisecondes
    audio = AudioSegment.from_file(audio_path)
    chunks = []
    duration_ms = len(audio)

    # Crée le répertoire temporaire si nécessaire
    os.makedirs("temp", exist_ok=True)

    step_ms = int(chunk_length * 1000)
    for start_ms in range(0, duration_ms, step_ms):
        end_ms = min(start_ms + step_ms, duration_ms)
        segment = audio[start_ms:end_ms]
        chunk_path = f"temp/chunk_{start_ms // 1000}.mp3"
        segment.export(chunk_path, format="mp3")
        chunks.append(chunk_path)

    return chunks

def _convert_to_wav_16k_mono(src_path: str) -> str:
    """Convert an audio file to 16kHz mono WAV for maximum compatibility."""
    audio = AudioSegment.from_file(src_path)
    audio = audio.set_frame_rate(16000).set_channels(1)
    fd, tmp_path = tempfile.mkstemp(prefix="chunk_", suffix=".wav")
    os.close(fd)
    audio.export(tmp_path, format="wav")
    return tmp_path


# Fonction pour transcrire un fichier audio (robuste)
def transcribe_audio(audio_chunk_path):
    # Skip tiny/empty chunks
    try:
        if not os.path.exists(audio_chunk_path) or os.path.getsize(audio_chunk_path) < 1024:
            print("Skipping invalid/too-small chunk: %s", audio_chunk_path)
            return ""
    except Exception:
        return ""

    prompt_text = "Transcribe the audio exactly in the spoken language of the speaker. If the audio contains multiple languages (e.g., English, French, Arabic, or others), switch dynamically and write each segment in its original spoken language without translation. Do not normalize or convert languages; preserve the natural mix exactly as spoken."

    # First attempt: use the file as-is
    try:
        with open(audio_chunk_path, "rb") as audio_file:
            result = openai.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="text",
                prompt=prompt_text
            )
        return str(result)
    except Exception as first_error:
        print("Direct transcription failed for %s, retrying as 16k WAV: %s", audio_chunk_path, first_error)

    # Retry after converting to 16kHz mono WAV
    safe_path = None
    try:
        safe_path = _convert_to_wav_16k_mono(audio_chunk_path)
        with open(safe_path, "rb") as audio_file:
            result = openai.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="text",
                prompt=prompt_text
            )
        return str(result)
    except Exception as second_error:
        return ""
    finally:
        if safe_path and os.path.exists(safe_path):
            try:
                os.remove(safe_path)
            except Exception:
                pass


# Helper functions for the new download_audio function
def _normalize_youtube_url(url):
    """Normalize YouTube URL to standard format."""
    # Handle various YouTube URL formats
    patterns = [
        r'(?:https?://)?(?:www\.)?youtube\.com/watch\?v=([^&\n?#]+)',
        r'(?:https?://)?(?:www\.)?youtu\.be/([^&\n?#]+)',
        r'(?:https?://)?(?:www\.)?youtube\.com/embed/([^&\n?#]+)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            video_id = match.group(1)
            return f"https://www.youtube.com/watch?v={video_id}"
    
    # If no pattern matches, return as-is
    return url


def _file_is_valid(filepath):
    """Check if the downloaded file is valid (non-trivial size)."""
    try:
        return os.path.exists(filepath) and os.path.getsize(filepath) > 1024  # >1KB
    except Exception:
        return False


def _ensure_supported_audio(filepath):
    """Convert audio file to a supported format (mp3) if needed."""
    try:
        # Get file extension
        _, ext = os.path.splitext(filepath)
        ext = ext.lower()
        
        # If already mp3, return as-is
        if ext == '.mp3':
            return filepath
        
        # Convert to mp3 using pydub
        audio = AudioSegment.from_file(filepath)
        mp3_path = filepath.rsplit('.', 1)[0] + '.mp3'
        audio.export(mp3_path, format="mp3")
        
        # Remove original file if different
        if filepath != mp3_path:
            try:
                os.remove(filepath)
            except Exception:
                pass
        
        return mp3_path
    except Exception as e:
        return filepath


def _yt_dlp_download(url, temp_dir):
    """Download audio using yt-dlp with robust options and dynamic format selection."""

    def _ffmpeg_available():
        try:
            return shutil.which("ffmpeg") is not None
        except Exception:
            return False

    def build_common_opts():
        opts = {
            'outtmpl': os.path.join(temp_dir, 'audio.%(ext)s'),
            'postprocessors': [],
            'noplaylist': True,
            'extract_flat': False,
            'writethumbnail': False,
            'writeinfojson': False,
            'ignoreerrors': False,
            'no_warnings': False,
            'retries': 10,
            'fragment_retries': 10,
            'concurrent_fragment_downloads': 1,
            'nocheckcertificate': True,
            'user_agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
            'quiet': True,
            'restrictfilenames': True,
            'force_overwrites': True,
            'prefer_free_formats': False,
            'geo_bypass': True,
            'source_address': '0.0.0.0',
            'cachedir': False,
            'http_headers': {
                'Accept-Language': 'en-US,en;q=0.9',
            },
        }
        if _ffmpeg_available():
            # Prefer ffmpeg for HLS/DASH merging and audio extraction
            opts['prefer_ffmpeg'] = True
            opts['hls_prefer_native'] = False
        else:
            opts['hls_prefer_native'] = True
        # Optional: cookies from browser
        try:
            browser = st.secrets.get("YTDLP_COOKIES_FROM_BROWSER", "").strip()
        except Exception:
            browser = ""
        if browser:
            opts['cookiesfrombrowser'] = (browser, )
        # Optional: proxy
        try:
            proxy = st.secrets.get("YTDLP_PROXY", "").strip()
        except Exception:
            proxy = ""
        if proxy:
            opts['proxy'] = proxy
        # Optional: player clients
        try:
            clients_raw = st.secrets.get("YTDLP_PLAYER_CLIENTS", "").strip()
        except Exception:
            clients_raw = ""
        if clients_raw:
            clients = [c.strip() for c in clients_raw.split(',') if c.strip()]
        else:
            clients = ['tv', 'ios', 'web']
        extractor_args = {'youtube': {'player_client': clients}}
        # Optional: PO token
        try:
            po_token = st.secrets.get("YTDLP_PO_TOKEN", "").strip()
        except Exception:
            po_token = ""
        if po_token:
            extractor_args['youtube']['po_token'] = po_token
        opts['extractor_args'] = extractor_args
        return opts

    # Probe available formats without downloading
    probe_opts = build_common_opts()
    probe_opts['skip_download'] = True
    with yt_dlp.YoutubeDL(probe_opts) as ydl:
        info = ydl.extract_info(url, download=False)

    formats = info.get('formats') or []
    # Filter to audio-only formats
    audio_formats = [f for f in formats if (f.get('vcodec') in (None, 'none')) and (f.get('acodec') not in (None, 'none'))]

    def ext_priority(ext):
        order = {'m4a': 3, 'mp3': 2, 'webm': 2, 'ogg': 1, 'wav': 1}
        return order.get((ext or '').lower(), 0)

    def bitrate(f):
        return f.get('abr') or f.get('tbr') or 0

    # Sort best first by bitrate then by preferred extension
    audio_formats.sort(key=lambda f: (bitrate(f), ext_priority(f.get('ext'))), reverse=True)

    # Build format_id attempts, then generic fallbacks (include merge of video+audio)
    format_attempts = [f.get('format_id') for f in audio_formats if f.get('format_id')]
    format_attempts += ['bestaudio/best', 'bestaudio', 'bestvideo*+bestaudio/best', 'best']

    def try_once(fmt: str):
        # Remove any previous empty files to avoid confusion
        for f in os.listdir(temp_dir):
            p = os.path.join(temp_dir, f)
            try:
                if os.path.isfile(p) and os.path.getsize(p) == 0:
                    os.remove(p)
            except Exception:
                pass

        ydl_opts = build_common_opts()
        ydl_opts['format'] = fmt
        if _ffmpeg_available():
            # Extract and transcode to mp3 directly with ffmpeg
            ydl_opts['postprocessors'] = [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }]
            ydl_opts['final_ext'] = 'mp3'
            # Enable merging if downloading separate video+audio
            ydl_opts['merge_output_format'] = 'mp4'

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.extract_info(url, download=True)

        # Inspect downloaded file(s)
        for file in os.listdir(temp_dir):
            if any(file.lower().endswith(ext) for ext in ['.mp3', '.m4a', '.webm', '.ogg', '.wav', '.mp4', '.aac']):
                candidate = os.path.join(temp_dir, file)
                if _file_is_valid(candidate):
                    return candidate
        return None

    downloaded = None
    last_err = None
    for fmt in format_attempts:
        try:
            downloaded = try_once(fmt)
            if downloaded:
                break
        except Exception as e:
            last_err = e

    if not downloaded:
        if last_err:
            raise RuntimeError(f"yt-dlp failed: {last_err}")
        raise RuntimeError("No non-empty audio file found after download")

    converted = _ensure_supported_audio(downloaded)
    if not _file_is_valid(converted):
        raise RuntimeError("Downloaded/converted audio file is empty")
    return converted


def download_audio(url):
    """
    Download audio from the given URL.
    Args:
        url (str): The URL of the YouTube video.
    Returns:
        str: The filepath of the downloaded audio file, normalized to mp3 when possible.
    """
    norm_url = _normalize_youtube_url(url)

    temp_dir = tempfile.mkdtemp(prefix="yt_audio_")

    # First try PyTube
    try:
        yt = YouTube(norm_url)
        stream = yt.streams.filter(only_audio=True).first()
        filepath = stream.download(output_path=temp_dir)
        if not _file_is_valid(filepath):
            raise RuntimeError("PyTube produced an empty file")
        filepath = _ensure_supported_audio(filepath)
        return filepath
    except Exception as e:

    # Fallback to yt-dlp with robust options and retries
    try:
        filepath = _yt_dlp_download(norm_url, temp_dir)
        return filepath
    except Exception as e:
        # Cleanup temp dir if nothing was downloaded
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception:
            pass
        raise RuntimeError(f"Failed to download audio: {e}")


# Interface Streamlit
st.title("🎙 Transcripteur de Vidéos YouTube Le Matin")
st.write("Entrez un lien YouTube pour obtenir la transcription de la vidéo.")

video_url = st.text_input("🔗 Entrez le lien YouTube ici", "")

if st.button("Transcrire la vidéo"):
    if video_url:
        st.info("📥 Téléchargement de l'audio...")
        audio_path = download_and_convert_audio(video_url)
        
        if audio_path:
            st.info("🎙 Découpage de l'audio...")
            audio_chunks = split_audio(audio_path, chunk_length=60)

            st.info("📝 Transcription en cours...")
            full_transcript = ""
            for chunk in audio_chunks:
                full_transcript += transcribe_audio(chunk) + "\n"
                os.remove(chunk)  # Nettoyage
            
            st.success("✅ Transcription terminée !")
            st.text_area("📜 Texte Transcrit", full_transcript, height=300)
            st.download_button("⬇ Télécharger la transcription", full_transcript, file_name="transcription.txt")

            os.remove(audio_path)  # Nettoyage
    else:
        st.error("❌ Veuillez entrer un lien YouTube valide.")

