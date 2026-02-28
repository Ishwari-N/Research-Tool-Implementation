# Research Tool Implementation: Earnings Call Summary

A Streamlit-based application that extracts structured insights from earnings call transcripts using Google Gemini and Groq AI.

## Features
- Structured summary extraction (Positives, Concerns, Guidance, Growth Initiatives).
- Management tone and confidence level assessment.
- Support for PDF and TXT file uploads.
- Dual-provider fallback (Groq + Gemini).

## Setup
1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`.
3. Create a `.env` file with your `GEMINI_API_KEY` and `GROQ_API_KEY`.
4. Run the app: `streamlit run app.py`.
