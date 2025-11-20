import streamlit as st
import os
from dotenv import load_dotenv
import openai
import base64
import cv2
from io import BytesIO
from tempfile import NamedTemporaryFile
import numpy as np
import subprocess
import yt_dlp # For downloading video links

# Load environment variables
load_dotenv()
try:
    openai.api_key = os.getenv("OPENAI_API_KEY")
    if not openai.api_key:
        raise ValueError("OPENAI_API_KEY is not set.")
except Exception:
    st.error("FATAL: OpenAI API Key not found. Please set the OPENAI_API_KEY in your .env file.")
    st.stop()


# --- CONFIGURATION ---
FRAME_RATE_SECONDS = 5  
MODEL_WHISPER = "whisper-1"
MODEL_GPT4O = "gpt-4o"
MODEL_TTS = "tts-1"
MAX_VIDEO_SIZE_MB = 150
TEMP_VIDEO_FILE = "downloaded_video_link.mp4"

# --- SESSION STATE INITIALIZATION ---
if 'summary' not in st.session_state: st.session_state.summary = ""
if 'transcript' not in st.session_state: st.session_state.transcript = "Transcript will appear here."
if 'tts_audio' not in st.session_state: st.session_state.tts_audio = None
if 'analysis_run' not in st.session_state: st.session_state.analysis_run = False
if 'category' not in st.session_state: st.session_state.category = None


# --- CORE LOGIC FUNCTIONS ---

def download_video_from_url(url):
    """Downloads video from URL using yt-dlp and returns path."""
    st.info(f"Downloading video from: {url}")
    
    try:
        subprocess.run([
            'yt-dlp', 
            '-f', 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4', 
            '-o', TEMP_VIDEO_FILE,
            '--recode-video', 'mp4',
            url
        ], check=True, capture_output=True)
        
        if os.path.exists(TEMP_VIDEO_FILE):
            st.success("Video downloaded successfully.")
            return TEMP_VIDEO_FILE
        else:
            st.error("Download failed or produced no output file.")
            return None
            
    except subprocess.CalledProcessError as e:
        st.error(f"Download Error: Could not download video. Details: {e.stderr.decode()}")
        return None
    except FileNotFoundError:
        st.error("External Tool Error: 'yt-dlp' not found. Ensure it is installed via Homebrew.")
        return None


def get_audio_frames_from_video(video_file_input):
    """
    Handles file objects or file paths. Extracts audio using ffmpeg, and prepares frames using OpenCV.
    Returns: audio_buffer (BytesIO), frames (list of base64 strings), temp_video_path (str)
    """
    
    if isinstance(video_file_input, str):
        temp_video_path = video_file_input
    else:
        with NamedTemporaryFile(suffix=".mp4", delete=False) as temp_video:
            temp_video.write(video_file_input.read())
            temp_video_path = temp_video.name

    temp_audio_path = temp_video_path.replace(".mp4", "_audio.mp3")
    audio_buffer = None
    frames = []

    try:
        st.info("Extracting audio using ffmpeg...")
        subprocess.run([
            'ffmpeg', 
            '-i', temp_video_path, 
            '-vn', 
            '-acodec', 'libmp3lame', 
            '-b:a', '64k', # Compress audio
            temp_audio_path
        ], check=True, capture_output=True)

        audio_buffer = BytesIO()
        with open(temp_audio_path, 'rb') as f:
            audio_buffer.write(f.read())
        audio_buffer.seek(0)

        st.info("Extracting visual frames using OpenCV...")
        cap = cv2.VideoCapture(temp_video_path)
        if not cap.isOpened():
            st.error("OpenCV Error: Could not open video file for processing frames.")
            return audio_buffer, frames, temp_video_path
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps * FRAME_RATE_SECONDS)
        count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            if count % frame_interval == 0:
                success, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                if success:
                    frames.append(base64.b64encode(buffer).decode('utf-8'))
            count += 1
        cap.release()
        
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
            
        return audio_buffer, frames, temp_video_path
        
    except subprocess.CalledProcessError as e:
        st.error(f"FFmpeg or Subprocess Error: {e.stderr.decode()}. Please check if ffmpeg is correctly installed and in PATH.")
        return None, [], temp_video_path
        
    except Exception as e:
        st.error(f"Media Processing Error: {e}")
        return None, [], temp_video_path

def transcribe_audio_for_whisper(audio_buffer):
    """Transcribes audio data (BytesIO object) using OpenAI Whisper."""
    with NamedTemporaryFile(suffix=".mp3", delete=False) as temp_audio:
        audio_buffer.seek(0)
        temp_audio.write(audio_buffer.read())
        temp_audio_path = temp_audio.name
        
    try:
        with open(temp_audio_path, "rb") as audio_file:
            response = openai.audio.transcriptions.create(
                model=MODEL_WHISPER, 
                file=audio_file
            )
        return response.text
    finally:
        os.remove(temp_audio_path)

def generate_multimodal_summary(transcript, frames_base64, user_question, visual_only=False):
    """Submits the full multimodal query to GPT-4o, adjusting prompt for visual-only analysis."""
    
    if visual_only:
        system_prompt = ("You are an AI Video Assistant. The video had NO AUDIO or silent audio. "
                         "Analyze the provided visual frames exclusively to infer the content, context, and intent of the video based on the user's request. Focus on actions and text visible in the images.")
    else:
        system_prompt = ("You are an AI Video Assistant. Your task is to analyze the video based on the "
                         "provided frames and transcript. Provide a comprehensive, actionable response that addresses the user's specific request.")
    
    content = [{"type": "text", "text": f"USER REQUEST: {user_question} \n\n --- TRANSCRIPT ---: \n{transcript}"}]
    
    for frame_b64 in frames_base64:
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame_b64}", "detail": "low"}})

    try:
        response = openai.chat.completions.create(
            model=MODEL_GPT4O,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content}
            ],
            max_tokens=1000 
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"GPT-4o Query Error: {e}")
        return "Error in generating AI response."
    
def classify_video(transcript, frames_base64):
    """Classifies the video into one of six categories using GPT-4o."""
    
    categories = [
        "Educational/Informational",
        "Entertainment",
        "Promotional/Marketing",
        "News/Commentary",
        "Scenic/Landscape",
        "Others"
    ]
    
    prompt = (
        f"Analyze the visual frames and transcript. Choose the SINGLE BEST category that most accurately describes the video's content from the following list: {', '.join(categories)}. "
        "Your response MUST be only the chosen category name and nothing else."
    )
    
    content = [{"type": "text", "text": prompt + f" --- TRANSCRIPT PREVIEW ---: {transcript[:500]}"}]
    
    for frame_b64 in frames_base64[:5]:
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame_b64}", "detail": "low"}})

    try:
        response = openai.chat.completions.create(
            model=MODEL_GPT4O,
            messages=[{"role": "user", "content": content}],
            max_tokens=50 
        )
        return response.choices[0].message.content.strip().replace('.', '')

    except Exception:
        return "Classification Failed"

def synthesize_speech(text):
    """Converts the AI's text response to raw audio bytes."""
    response = openai.audio.speech.create(
        model=MODEL_TTS, 
        voice="alloy", 
        input=text, 
        response_format="mp3"
    )
    return response.content

# --- STREAMLIT UI & EXECUTION ---

st.set_page_config(page_title="Multimodal Video Assistant", layout="wide")
st.title("🎥 Multimodal Video Analysis (Level 3)")
st.markdown("---")

# --- UI INPUT SELECTION ---
input_method = st.radio("Select Input Method:", ("Upload Local File", "Provide Video Link"), horizontal=True)

video_input = None
uploaded_file = None
video_url = None

if input_method == "Upload Local File":
    uploaded_file = st.file_uploader("Upload MP4/MOV Video File (Max 150MB)", type=["mp4", "mov"])
    if uploaded_file:
        video_input = uploaded_file
elif input_method == "Provide Video Link":
    video_url = st.text_input("Enter YouTube or Instagram Video Link:")
    if video_url:
        video_input = video_url

user_question = st.text_area(
    "Analysis Request:",
    value="Analyze the video's content and give a comprehensive summary and key takeaways, elaborated in three structured paragraphs.",
    height=150
)

# --- EXECUTION BUTTON ---
if video_input and st.button("Start Multimodal Analysis"):
    st.session_state.analysis_run = False
    st.session_state.category = None
    
    video_file_path_for_cleanup = None
    continue_analysis = True
    
    if input_method == "Upload Local File" and video_input.size > MAX_VIDEO_SIZE_MB * 1024 * 1024:
        st.error(f"File size exceeds the {MAX_VIDEO_SIZE_MB}MB limit.")
        continue_analysis = False
        
    st.info("Analysis started. This may take a minute or two depending on video length.")
    
    if continue_analysis:
        try:
            # --- PHASE 0: HANDLE INPUT TYPE ---
            media_input_for_extraction = None
            
            if input_method == "Provide Video Link":
                # Download file and get the temporary file path for processing
                temp_video_path_str = download_video_from_url(video_url)
                if not temp_video_path_str:
                    continue_analysis = False # Set flag if download fails
                else:
                    media_input_for_extraction = temp_video_path_str
                    video_file_path_for_cleanup = temp_video_path_str 
                
            else: # Upload Local File
                media_input_for_extraction = uploaded_file
                
            if continue_analysis:
                # --- PHASE 1: MEDIA EXTRACTION (Audio/Frames) ---
                with st.spinner("1/4: Extracting audio and visual streams..."):
                    audio_buffer, frames_base64, temp_video_path_output = get_audio_frames_from_video(media_input_for_extraction)
                    
                # Set the cleanup path for the uploaded file
                if input_method == "Upload Local File":
                    video_file_path_for_cleanup = temp_video_path_output

                if not audio_buffer:
                    st.error("Media extraction failed. Cannot proceed with transcription.")
                    continue_analysis = False
            
            if continue_analysis:
                # --- PHASE 2: TRANSCRIPTION ---
                with st.spinner("2/4: Transcribing audio with OpenAI Whisper..."):
                    transcript_text = transcribe_audio_for_whisper(audio_buffer)

                is_visual_only = (not transcript_text or transcript_text.strip() == "")
                if is_visual_only:
                    st.warning("⚠️ No meaningful speech detected. Switching to visual-only analysis mode.")
                    transcript_text = "No speech detected in the audio track."
                    
                # --- PHASE 3: CLASSIFICATION & SUMMARY ---
                with st.spinner("Classifying video content..."):
                    category = classify_video(transcript_text, frames_base64)
                    st.session_state.category = category 
                    
                with st.spinner("3/4: Generating comprehensive summary with GPT-4o..."):
                    summary = generate_multimodal_summary(transcript_text, frames_base64, user_question, visual_only=is_visual_only)

                # --- PHASE 4: SPEECH SYNTHESIS ---
                with st.spinner("4/4: Synthesizing audio response..."):
                    tts_audio_bytes = synthesize_speech(summary)

                # --- SAVE RESULTS TO SESSION STATE ---
                st.session_state.summary = summary
                st.session_state.transcript = transcript_text
                st.session_state.tts_audio = tts_audio_bytes
                st.session_state.analysis_run = True
                
                st.rerun()

        except Exception as e:
            st.error(f"An unexpected error occurred during processing: {e}")
            
        finally:
            # FINAL CLEANUP: Ensure temporary video file is deleted
            if video_file_path_for_cleanup and os.path.exists(video_file_path_for_cleanup):
                os.remove(video_file_path_for_cleanup)


# --- DISPLAY RESULTS (Runs after every rerun) ---

if st.session_state.get('category'):
    st.subheader("Video Classification")
    st.markdown(f"**Category:** <span style='background-color:#E0F7FA; padding: 4px 8px; border-radius: 5px; font-weight: bold; color: #00796B;'>{st.session_state.category}</span>", unsafe_allow_html=True)
    st.markdown("---")

if st.session_state.analysis_run:
    st.markdown("## ✅ Analysis Results")
    
    col1, col2 = st.columns(2)
    
    tts_audio_bytes = st.session_state.tts_audio
    tts_audio_buffer = BytesIO(tts_audio_bytes) 
    
    with col1:
        st.markdown("### 🗣️ Spoken Summary")
        st.audio(tts_audio_buffer, format='audio/mp3')
        
        st.download_button(
            label="Download Spoken Summary (MP3)",
            data=tts_audio_bytes, 
            file_name="multimodal_summary.mp3",
            mime="audio/mp3"
        )

    with col2:
        st.markdown("### 📄 Text Summary (GPT-4o)")
        st.text_area("Summary", st.session_state.summary, height=300)
        
    st.markdown("### 📜 Full Transcript (Whisper)")
    with st.expander("Click to view full transcript"):
        st.text(st.session_state.transcript)