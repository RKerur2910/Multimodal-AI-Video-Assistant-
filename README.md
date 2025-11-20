# 🧠 Multimodal AI Video Assistant

**Tech Stack:**  Streamlit · OpenAI GPT-4o · Whisper · TTS · OpenCV · FFmpeg · Python

## 📘 Overview

The **Multimodal AI Video Assistant** is an AI system that performs **image and audio processing** on video files to generate structured, human-readable insights. It combines **computer vision, speech recognition, and language modeling** to deliver text-based and spoken summaries through a simple Streamlit interface.

## ⚙️ Features

🎞️ **Frame Extraction & Sampling:** Extracts frames efficiently from videos using OpenCV and FFmpeg.

🔊 **Audio Transcription:** Converts speech to text using OpenAI’s Whisper model for accurate transcription.

🧠 **Multimodal Fusion:** Integrates frame and transcript data for context-aware summarization using GPT-4o.

🗣️ **Text-to-Speech:** Generates a playable, downloadable audio summary through OpenAI TTS.

🌐 **Interactive UI:** Streamlit-based interface for uploading videos, running the analysis, and viewing/download results.

## 🧩 System Architecture

 Video Input  
      ↓  
 Audio Extraction (FFmpeg)  
      ↓  
 Frame Sampling (OpenCV)  
      ↓  
 Speech-to-Text (Whisper)  
      ↓  
 GPT-4o Summarization  
      ↓  
 Output: Text Summary + TTS Audio File

## 🧪 Results

- Reduced computational overhead with **frame sampling (1 frame per 5 seconds).**
- Enhanced reliability through optimized **session state management.**
- Produced concise and synchronized **text + audio summaries** from raw video content.

## 📝 Process

- User uploads a video file through the Streamlit interface.
- The system extracts audio and visual frames → performs transcription → generates multimodal summaries.
- Combines results from both modalities to produce reliable and clean insights.
- The final text summary and TTS audio file are provided for download.

## ⚡ Challenges & Fixes

- Blank audio output: Identified root cause as incorrect TTS node mapping → fixed mappings, added blank-summary checks, text chunking, and improved file handling.
- Length constraints: TTS node limited to ~4,096 characters → optimized summarization prompts to fit limits while maintaining content fidelity.

## 🔍 Use Cases

- Automated generation of daily or weekly research summaries.

Converting long-form videos, lectures, or reports into short, digestible audio briefs.

Assisting creators, journalists, and analysts with multimodal content analysis.

## ▶️ Next Steps

Extend to mini-podcast generation with ≤60-second summaries.

Support multi-topic batching and scheduled digest creation.

Add transcript export alongside audio output for accessibility.

Riya Kalyan Kerur
Master’s Student, Computer Engineering — California State University, Sacramento
🌐 LinkedIn
