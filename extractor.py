import os
import json
import time
import re
from enum import Enum
from typing import List
from pydantic import BaseModel, Field
import google.generativeai as genai
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Define Structured Output Schema
class ManagementTone(str, Enum):
    optimistic = "optimistic"
    cautious = "cautious"
    neutral = "neutral"
    pessimistic = "pessimistic"

class ConfidenceLevel(str, Enum):
    high = "high"
    medium = "medium"
    low = "low"

class EarningsSummary(BaseModel):
    tone: ManagementTone = Field(description="Management tone / sentiment based on direct quotes and sentiment words.")
    confidence: ConfidenceLevel = Field(description="Confidence level based on modifiers in direct quotes.")
    key_positives: List[str] = Field(description="3 to 5 key positives mentioned. Return empty list if none.", min_length=0, max_length=5)
    key_concerns: List[str] = Field(description="3 to 5 key concerns or challenges. Return empty list if none.", min_length=0, max_length=5)
    forward_guidance: str = Field(description="Forward guidance (revenue, margin, capex). Quote numbers if present. Summarize qualitative guidance but note lack of specifics if vague. Return 'Not mentioned' if completely omitted.")
    capacity_utilization_trends: str = Field(description="Capacity utilization trends. Return 'Not mentioned' if completely omitted.")
    growth_initiatives: List[str] = Field(description="2 to 3 new growth initiatives described. Return empty list if none.", min_length=0, max_length=3)

SYSTEM_PROMPT = """
You are an expert financial analyst. Your task is to analyze the provided earnings call transcript and extract a structured summary.

CRITICAL INSTRUCTIONS:
1. **Management Tone**: Choose strictly: optimistic, cautious, neutral, pessimistic.
2. **Confidence Level**: Choose strictly: high, medium, low.
3. **Guidance**: Quote specific numbers if present. Note if qualitative guidance is vague.
4. **Missing Information**: Use [] for empty lists and "Not mentioned" for missing text fields.

YOU MUST RESPOND ONLY WITH A VALID JSON OBJECT MATCHING THIS SCHEMA:
{
  "tone": "optimistic",
  "confidence": "high",
  "key_positives": ["item1", "item2"],
  "key_concerns": ["item1"],
  "forward_guidance": "description",
  "capacity_utilization_trends": "description",
  "growth_initiatives": ["item1"]
}
"""

def clean_json_string(content: str) -> str:
    """Extracts JSON from markdown code blocks or raw text."""
    # Remove markdown formatting
    content = content.replace("```json", "").replace("```", "").strip()
    
    # Simple extraction: find first '{' and last '}'
    start_idx = content.find('{')
    end_idx = content.rfind('}')
    
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        return content[start_idx:end_idx + 1]
    return content

def get_key(name: str) -> str:
    # Check env first, then streamlit secrets
    import streamlit as st
    key = os.getenv(name)
    if not key and name in st.secrets:
        key = st.secrets[name]
    return key

def extract_with_groq(transcript_text: str) -> str:
    api_key = get_key("GROQ_API_KEY")
    if not api_key:
        raise ValueError("Groq API Key missing.")
    
    client = Groq(api_key=api_key)
    
    # Using Llama 3.1 8B because it has much higher rate limits (TPM) than 70B models
    models_to_try = ["llama-3.1-8b-instant", "llama-3.3-70b-versatile", "mixtral-8x7b-32768"]
    
    for model_id in models_to_try:
        try:
            print(f"DEBUG: Trying Groq Model: {model_id}")
            completion = client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Extract JSON summary from this text:\n\n{transcript_text[:12000]}"},
                ],
                temperature=0,
            )
            return completion.choices[0].message.content
        except Exception as e:
            if "429" in str(e) or "quota" in str(e).lower():
                print(f"DEBUG: {model_id} rate limited. Retrying next...")
                continue
            raise e
    raise RuntimeError("All Groq models hit rate limits.")

def extract_with_gemini(transcript_text: str) -> str:
    api_key = get_key("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Gemini API Key missing.")
    
    genai.configure(api_key=api_key)
    
    # Test models in order of availability
    models_to_try = ['models/gemini-1.5-flash-8b', 'models/gemini-1.5-flash', 'models/gemini-2.0-flash']
    
    last_err = None
    for m_name in models_to_try:
        try:
            model = genai.GenerativeModel(m_name)
            response = model.generate_content(f"{SYSTEM_PROMPT}\n\nTRANSCRIPT:\n{transcript_text[:15000]}")
            return response.text
        except Exception as e:
            last_err = e
            print(f"DEBUG: Gemini {m_name} failed: {e}")
            continue
            
    raise last_err

def extract_earnings_summary(transcript_text: str) -> EarningsSummary:
    errors = []
    
    # Attempt 1: Groq (Usually more reliable for these specific keys)
    try:
        content = extract_with_groq(transcript_text)
        cleaned = clean_json_string(content)
        return EarningsSummary.model_validate_json(cleaned)
    except Exception as e:
        errors.append(f"Groq: {str(e)}")
    
    # Attempt 2: Gemini
    try:
        content = extract_with_gemini(transcript_text)
        cleaned = clean_json_string(content)
        return EarningsSummary.model_validate_json(cleaned)
    except Exception as e:
        errors.append(f"Gemini: {str(e)}")

    raise RuntimeError(f"Extraction failed.\n\nErrors:\n- " + "\n- ".join(errors))
