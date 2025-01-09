# YouTube Video Summarizer Pro 🎥

An advanced AI-powered application that generates comprehensive summaries of YouTube videos using Streamlit, combining the power of YouTube's API, OpenAI's Whisper, and Groq's language models for intelligent content analysis.

## 🌟 Features

- **Multi-Language Support**: Transcribe videos in any language using Whisper AI
- **Dual Transcription Methods**: Choose between YouTube's native captions or Whisper AI
- **Intelligent Summarization**: Generate concise, balanced, or detailed summaries using Groq's language models
- **Flexible URL Support**: Compatible with regular YouTube videos, shorts, and various URL formats
- **Interactive UI**: Clean, modern interface with real-time progress tracking
- **Download Options**: Export summaries as text files
- **Error Handling**: Robust error management and fallback options

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- FFmpeg installed on your system
- API keys for:
  - Groq
  - YouTube Data API (optional)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Youtube-Summarizer.git
cd Youtube-Summarizer
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the project root:
```env
GROQ_API_KEY=your_groq_api_key
```

4. Update FFmpeg path in `app.py`:
```python
ffmpeg_path = r'path/to/your/ffmpeg'
```

### Usage

1. Start the application:
```bash
streamlit run app.py
```

2. Enter a YouTube URL in the input field

3. Choose your preferred transcription method:
   - YouTube Captions: Faster, supports native captions
   - Whisper AI: Better accuracy, supports all languages

4. Select summary length:
   - Concise: Key points only
   - Balanced: Moderate detail
   - Detailed: Comprehensive coverage

5. Click "Generate Summary" and wait for processing

## 🛠 Technical Details

### Components

- **Frontend**: Streamlit with custom CSS styling
- **Transcription**: 
  - YouTube Transcript API
  - OpenAI's Whisper (local processing)
- **Summarization**: Groq's Mixtral-8x7b model
- **Video Processing**: yt-dlp for audio extraction
- **Audio Processing**: FFmpeg for format conversion

### Architecture

```
app.py
├── Configuration and Setup
├── FFmpeg and Whisper Setup
├── URL and Video Processing
├── Audio and Transcription
├── Text Processing
├── AI Processing
└── UI Components
```

### Key Functions

- `extract_video_id()`: Handles various YouTube URL formats
- `download_audio()`: Extracts audio for Whisper processing
- `get_whisper_transcript()`: Generates transcripts using Whisper
- `extract_transcript_details()`: Manages transcript extraction workflow
- `generate_groq_content()`: Handles AI summarization
- `chunk_text()`: Splits text for efficient processing

## 🎯 Features In Detail

### Transcription Options

1. **YouTube Captions**
   - Uses native YouTube transcriptions
   - Faster processing
   - Limited to available caption languages

2. **Whisper AI**
   - Local processing using OpenAI's Whisper( used the Small model)
   - Supports all languages
   - Higher accuracy
   - Requires more processing time

**Whisper Model Recommendations**

| Use Case                                       | Best Model | Reason                                      |
|-----------------------------------------------|-----------|---------------------------------------------|
| Real-time transcription (Fast & Light)       | Tiny/Base | Low memory usage, fast                     |
| General transcription (Balanced)             | Small     | Good accuracy & speed                      |
| High-quality transcription                   | Medium    | Very accurate, handles noise               |
| Best accuracy (Complex audio, accents, multiple languages) | Large | Highest accuracy, but slow                 |


### Summary Lengths

1. **Concise**
   - 30% of full length
   - Focus on key points
   - Perfect for quick overview

2. **Balanced**
   - 60% of full length
   - Includes supporting details
   - Ideal for general use

3. **Detailed**
   - Full length
   - Comprehensive coverage
   - Best for in-depth analysis

## 📝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.


