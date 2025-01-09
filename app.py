"""
YouTube Video Summarizer Pro
----------------------------
A Streamlit application that generates comprehensive summaries of YouTube videos
using AI-powered transcription (YouTube API and Whisper) and summarization (Groq).

Features:
- Multi-language support using Whisper AI
- Automatic transcription using YouTube API
- Intelligent summarization using Groq
- Support for various YouTube URL formats
- Clean and modern UI with progress tracking
- Built-in error handling and user feedback

"""

import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from youtube_transcript_api import YouTubeTranscriptApi 
from urllib.parse import urlparse, parse_qs
from dotenv import load_dotenv
import yt_dlp
import whisper
import os
import re
from pathlib import Path
import subprocess
import tempfile
from datetime import datetime
import time
import ffmpeg

# ===============================
# Configuration and Setup
# ===============================
st.set_page_config(
        page_title="AI Video Summarizer Pro",
        page_icon="🎥",
        layout="wide",
        initial_sidebar_state="expanded"
    )
def configure_page():
    """Configure Streamlit page settings and styling."""
    
    # Custom CSS for better UI
    st.markdown("""
        <style>
        .main { padding: 0rem 1rem; }
        .stApp {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        }
        .title-container {
            background-color: rgba(255, 255, 255, 0.1);
            padding: 1.5rem;
            border-radius: 10px;
            margin-bottom: 2rem;
            text-align: center;
            backdrop-filter: blur(10px);
        }
        .stButton>button {
            background-color: #ADD8E6 !important;
            color: black !important;
            border-radius: 8px;
            padding: 10px 20px;
            font-size: 16px;
            font-weight: bold;
            transition: 0.3s;
        }
        .stButton>button:first-child:hover {
            background-color: #87CEEB !important; /* Slightly Darker Blue on Hover */
            border-color: #005A9E;
        }
        .stats-box {
            background-color: rgba(255, 255, 255, 0.9);
            padding: 1rem;
            border-radius: 10px;
            margin: 0.5rem 0;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .output-container {
            background-color: white;
            padding: 2rem;
            border-radius: 10px;
            margin-top: 2rem;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        </style>
    """, unsafe_allow_html=True)

# ===============================
# FFmpeg and Whisper Setup
# ===============================

def set_ffmpeg_path():
    """Set FFmpeg path for Whisper"""
    ffmpeg_path = r'C:\ffmpeg_path'  # Update this path for your deployment
    if os.path.exists(ffmpeg_path):
        os.environ["PATH"] = f"{ffmpeg_path};{os.environ['PATH']}"
        os.environ["FFMPEG_BINARY"] = os.path.join(ffmpeg_path, "ffmpeg.exe")
        return True
    return False

@st.cache_resource
def initialize_whisper():
    """Initialize Whisper model with proper device selection and FFmpeg configuration."""
    try:
        if not set_ffmpeg_path():
            st.error("❌ FFmpeg path not found")
            return None

        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        st.info(f"✓ Using device: {device}")
        
        whisper.audio.SAMPLE_RATE = 16000
        whisper.audio.N_FRAMES = 480000
        
        model = whisper.load_model("medium", device=device)
        st.success("✓ Whisper model loaded successfully")
        return model
    except Exception as e:
        st.error(f"❌ Error initializing Whisper: {str(e)}")
        return None

# ===============================
# URL and Video Processing
# ===============================

def extract_video_id(url):
    """
    Extract YouTube video ID from various URL formats including regular videos,
    shorts, and other YouTube URL variants.
    """
    if not url:
        return None

    patterns = [
        r'(?:youtu\.be/|youtube\.com/(?:embed/|v/|shorts/|watch\?v=))([^?&\n]+)',
        r'youtube\.com/watch\?.*v=([^?&\n]+)',
        r'youtube\.com/shorts/([^?&\n]+)',
    ]

    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)

    return None

def get_thumbnail_url(video_id):
    """Get the highest quality thumbnail URL available for a video."""
    if not video_id:
        return None
    
    thumbnail_qualities = [
        'maxresdefault',
        'sddefault',
        'hqdefault',
        '0',
    ]
    
    for quality in thumbnail_qualities:
        thumbnail_url = f"http://img.youtube.com/vi/{video_id}/{quality}.jpg"
        return thumbnail_url
    
    return None

# ===============================
# Audio and Transcription
# ===============================

def download_audio(video_id):
    """Download audio from YouTube video for Whisper processing."""
    try:
        url = f"https://www.youtube.com/watch?v={video_id}"
        
        # Create temporary file
        temp_dir = tempfile.gettempdir()
        output_file = os.path.join(temp_dir, f'temp_{video_id}_{int(time.time())}.mp3')
        
        # st.info(f"✓ Using output path: {output_file}")  # Commented out for cleaner UI
        
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'outtmpl': output_file.replace('.mp3', ''),
            'ffmpeg_location': r'C:\ffmpeg_path',  # Update this path for your deployment
            'quiet': True,  # Added to suppress yt-dlp output
            'no_warnings': True  # Added to suppress warnings
        }
        
        # st.info("⏳ Starting download...")  # Commented out for cleaner UI
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
            
            if os.path.exists(output_file):
                # size = os.path.getsize(output_file)
                # st.success(f"✓ Downloaded file ({size} bytes)")  # Commented out for cleaner UI
                return output_file
            
            st.error("❌ Download failed")
            return None
            
    except Exception as e:
        st.error(f"❌ Download error: {str(e)}")
        return None

def get_whisper_transcript(audio_file):
    """Generate transcript using Whisper model with enhanced error handling."""
    try:
        audio_path = os.path.abspath(audio_file)
        
        if not os.path.exists(audio_path):
            st.error(f"❌ Audio file not found: {audio_path}")
            return None
            
        if not set_ffmpeg_path():
            st.error("❌ FFmpeg path not properly set")
            return None
        
        # st.info("⏳ Starting Whisper transcription...")  # Commented out for cleaner UI
        
        if not whisper_model:
            st.error("❌ Whisper model not initialized")
            return None
            
        result = whisper_model.transcribe(
            audio_path,
            fp16=False,
            language='en'
        )
        
        if result and "text" in result:
            # st.success("✓ Transcription completed")  # Commented out for cleaner UI
            return result["text"]
            
        st.error("❌ Transcription failed")
        return None
            
    except Exception as e:
        st.error(f"❌ Whisper transcription error: {str(e)}")
        return None
    finally:
        try:
            if os.path.exists(audio_path):
                os.remove(audio_path)
                # st.info("✓ Cleaned up audio file")  # Commented out for cleaner UI
        except Exception as e:
            # st.warning(f"⚠️ Cleanup warning: {str(e)}")  # Commented out for cleaner UI
            pass


def extract_transcript_details(youtube_video_url, use_whisper=False):
    """Extract transcript using either YouTube API or Whisper."""
    try:
        video_id = extract_video_id(youtube_video_url)
        if not video_id:
            st.error("Invalid YouTube URL")
            return None
        
        if use_whisper:
            st.info("🎵 Downloading audio for transcription...")
            audio_file = download_audio(video_id)
            
            if not audio_file:
                st.error("Failed to download audio file")
                return None
                
            transcript = get_whisper_transcript(audio_file)
            
            if not transcript:
                st.error("Whisper transcription failed")
                return None
        else:
            try:
                transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
                transcript = " ".join(item["text"] for item in transcript_list)
            except Exception as e:
                st.error(f"Failed to get YouTube transcript: {str(e)}")
                st.info("Falling back to Whisper transcription...")
                return extract_transcript_details(youtube_video_url, use_whisper=True)
        
        if not transcript or len(transcript.strip()) < 10:
            st.error("Generated transcript is too short or empty")
            return None
            
        return transcript
        
    except Exception as e:
        st.error(f"Transcript extraction error: {str(e)}")
        return None

# ===============================
# Text Processing
# ===============================

def chunk_text(text, chunk_size=2000):
    """Split text into manageable chunks for processing."""
    words = text.split()
    chunks = []
    current_chunk = []
    current_size = 0
    
    for word in words:
        current_size += len(word) + 1
        if current_size > chunk_size:
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]
            current_size = len(word)
        else:
            current_chunk.append(word)
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

# ===============================
# AI Processing
# ===============================

# System prompt for Groq
SYSTEM_PROMPT = '''You are a professional content analyst specializing in creating comprehensive video summaries. Your task is to analyze the provided transcript and generate a well-structured summary that captures the essence of the video content.

Output Format:
1. Title & Overview (2-3 sentences capturing the main topic and purpose)
2. Key Points (main arguments or topics discussed)
3. Detailed Summary (organized by topics or chronologically)
4. Notable Quotes or Examples (if any)
5. Technical Details or Specifications (if relevant)
6. Conclusions or Key Takeaways

Guidelines:
- Maintain the original speaker's tone and style
- Preserve technical accuracy in specialized content
- Include specific examples and data points when present
- Break down complex concepts into digestible segments
- Highlight practical applications or actionable insights
- Note any timestamps or section breaks if present in the transcript

Length Adjustments:
- For "Concise" summaries: Focus on key points and takeaways (30% of full length)
- For "Balanced" summaries: Include moderate detail and supporting points (60% of full length)
- For "Detailed" summaries: Comprehensive coverage with examples and context (full length)

Special Considerations:
- For technical videos: Emphasize methodologies, tools, and technical specifications
- For educational content: Structure in a learning-oriented format with clear explanations
- For narrative content: Preserve story arcs and key plot points
- For tutorial videos: Include step-by-step breakdowns of processes

Remember to:
- Maintain objectivity while preserving the video's intended message
- Highlight any disclaimers or important warnings
- Note any calls to action or recommended next steps
- Preserve the context of discussions and debates
'''


def generate_groq_content(transcript_text, summary_length="Balanced"):
    try:
        chunks = chunk_text(transcript_text)
        summaries = []
        
        full_prompt = SYSTEM_PROMPT + f"\n\nGenerate a {summary_length.lower()} summary for this transcript section:"
        

        progress_bar = st.progress(0)
        for i, chunk in enumerate(chunks):
            message = HumanMessage(
                content=f"{full_prompt}\n\nPlease summarize this section of the transcript:\n{chunk}"
            )
            
            response = groq_client.invoke([message])
            if response and response.content:
                summaries.append(response.content)
            
            progress_bar.progress((i + 1) / len(chunks))
        
        if len(summaries) > 1:
            final_message = HumanMessage(
                content=f"""Combine these section summaries into a coherent final summary, 
                organized by the main topics and key points:\n\n{"".join(summaries)}"""
            )
            final_response = groq_client.invoke([final_message])
            return final_response.content
        elif summaries:
            return summaries[0]
        else:
            raise ValueError("No summary generated")
            
    except Exception as e:
        st.error(f"Error generating summary: {str(e)}")
        return None

# ===============================
# UI Components
# ===============================

def display_header():
    """Display application header with styling."""
    st.markdown("""
        <div class="title-container">
            <h1>🎥 AI Video Summarizer Pro</h1>
            <p style="font-size: 1.2em; color: #666;">
                Transform any YouTube video into comprehensive, intelligent notes using advanced AI
            </p>
        </div>
    """, unsafe_allow_html=True)


def display_features():
    """Display key features in the sidebar."""
    st.sidebar.markdown("""
    # 🚀 Features
    
    - 🎯 **Precise Summarization**
    - 🌍 **Multi-language Support**
    - 🎨 **Smart Formatting**
    - ⚡ **Real-time Processing**
    - 📊 **Key Points Extraction**
    - 🔍 **Deep Analysis**
    
    ---
    
    ### 💡 Pro Tips
    - Use Whisper for non-English videos
    - Longer videos may take more time
    - Check both summarization methods
    """)

def display_processing_animation():
    """Display processing animation with custom styling."""
    with st.empty():
        # Use a more subtle progress indicator
        progress_placeholder = st.empty()
        for i in range(100):
            time.sleep(0.01)
            progress_placeholder.progress(i + 1)
        progress_placeholder.empty()


# ===============================
# Main Application
# ===============================

def main():
    """Main application function."""
    # Load environment variables
    load_dotenv()
    
    # Initialize Groq client
    global groq_client
    groq_client = ChatGroq(
        api_key=os.getenv('GROQ_API_KEY'),
        model_name='mixtral-8x7b-32768'
    )
    
    # Initialize Whisper model globally
    global whisper_model
    whisper_model = initialize_whisper()
    
    # Initialize session state variables
    if 'summaries_count' not in st.session_state:
        st.session_state.summaries_count = 0
    if 'processing_history' not in st.session_state:
        st.session_state.processing_history = []
    
    # Configure page layout and styling
    configure_page()
    
    # Display header and features
    display_header()
    display_features()
    
    # Main content area
    st.markdown("### 🎬 Enter Video Details")
    
    # Input container with styling
    with st.container():
        youtube_link = st.text_input(
            "YouTube URL",
            placeholder="Paste your YouTube video link here...",
            help="Support for regular videos, shorts, and playlists"
        )
        
        # Input options in columns
        col1, col2 = st.columns(2)
        with col1:
            transcription_method = st.radio(
                "Choose Transcription Method:",
                ("🎯 YouTube Captions", "🌍 Whisper AI (All Languages)"),
                help="Select Whisper for better accuracy on non-English content"
            )
        
        with col2:
            summary_length = st.select_slider(
                "Summary Length",
                options=["Concise", "Balanced", "Detailed"],
                value="Balanced",
                help="Adjust the level of detail in your summary"
            )
    

    
    # Process button with enhanced styling
    if st.button("🚀 Generate Summary"):
        if not youtube_link:
            st.error("❌ Please enter a YouTube URL")
            return
        
        video_id = extract_video_id(youtube_link)
        if not video_id:
            st.error("❌ Invalid YouTube URL format")
            return
        
        try:
            # Display video thumbnail if available
            thumbnail_url = get_thumbnail_url(video_id)
            if thumbnail_url:
                st.image(thumbnail_url, use_container_width=True)
            
            # Processing with visual feedback
            with st.spinner("Processing your video..."):
                use_whisper = "Whisper" in transcription_method
                transcript_text = extract_transcript_details(youtube_link, use_whisper)
                
                if transcript_text:
                    summary = generate_groq_content(transcript_text, summary_length)
                    
                    if summary:
                        # Show success message and update stats
                        st.success("🎉 Summary generated successfully!")
                        
                        # Update session state
                        st.session_state.summaries_count += 1
                        st.session_state.processing_history.append({
                            'timestamp': datetime.now(),
                            'video_id': video_id,
                            'method': transcription_method
                        })
                        
                        # Display summary in formatted container
                        st.markdown("""
                            <div class="output-container">
                                <h2>📑 Summary</h2>
                                <p>{}</p>
                            </div>
                        """.format(summary.replace('\n', '<br>')), unsafe_allow_html=True)
                        
                        # Add download button for summary
                        st.download_button(
                            label="📥 Download Summary",
                            data=summary,
                            file_name=f"video_summary_{video_id}.txt",
                            mime="text/plain"
                        )
                        
                else:
                    st.error("❌ Failed to extract transcript. Please try a different video or transcription method.")
                    
        except Exception as e:
            st.error(f"❌ An error occurred: {str(e)}")
            st.error("Please try again or contact support if the issue persists.")

if __name__ == "__main__":
    main()