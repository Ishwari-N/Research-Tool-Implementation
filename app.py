import streamlit as st
import PyPDF2
from extractor import extract_earnings_summary
import os

from dotenv import load_dotenv

# Load local environment variables
load_dotenv()

st.set_page_config(page_title="Earnings Call Summary App", page_icon="📈", layout="wide")

st.title("📈 Earnings Call Summary App")
st.write("Upload an earnings call transcript (PDF/TXT) or paste it directly to extract a structured commentary.")

# Helper to get API keys from env or Streamlit secrets
def get_api_key(name):
    key = os.getenv(name)
    if not key and name in st.secrets:
        key = st.secrets[name]
    return key

gemini_key = get_api_key("GEMINI_API_KEY")
groq_key = get_api_key("GROQ_API_KEY")

if not gemini_key and not groq_key:
    st.error("⚠️ **API Keys Missing!**")
    st.info("""
    - **Running Locally?** Ensure you have a `.env` file with `GEMINI_API_KEY` or `GROQ_API_KEY`.
    - **Running on Streamlit Cloud?** Add these keys in **Settings > Secrets**.
    """)

st.info("💡 **AI Engine:** Dual-Provider Fallback Active (Trying Groq first, then Gemini).")

# Tabs for input modes
tab_upload, tab_paste = st.tabs(["Upload Document", "Paste Transcript"])

transcript_text = ""

with tab_upload:
    uploaded_file = st.file_uploader("Upload an Earnings Call Transcript or Management Discussion", type=["txt", "pdf"])
    if uploaded_file is not None:
        if uploaded_file.name.endswith(".txt"):
            transcript_text = uploaded_file.getvalue().decode("utf-8")
        elif uploaded_file.name.endswith(".pdf"):
            try:
                pdf_reader = PyPDF2.PdfReader(uploaded_file)
                text = ""
                for page in pdf_reader.pages:
                    extracted = page.extract_text()
                    if extracted:
                        text += extracted + "\n"
                transcript_text = text
            except Exception as e:
                st.error(f"Error reading PDF: {e}")

with tab_paste:
    pasted_text = st.text_area("Paste the transcript text here:", height=300)
    if pasted_text:
        transcript_text = pasted_text

if st.button("Generate Summary", type="primary", use_container_width=True):
    if not transcript_text.strip():
        st.error("Please provide a transcript either by uploading a file or pasting text.")
    else:
        with st.spinner("Analyzing transcript (trying multiple AI providers)..."):
            try:
                summary = extract_earnings_summary(transcript_text)
                
                st.success("Summary Extracted Successfully")
                st.divider()

                # --- UI Display of the Results ---
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Tone & Confidence")
                    st.info(f"**Management Tone:** {summary.tone.value.title()}")
                    st.info(f"**Confidence Level:** {summary.confidence.value.title()}")
                    
                    st.subheader("Key Positives")
                    if summary.key_positives:
                        for pos in summary.key_positives:
                            st.write(f"✅ {pos}")
                    else:
                        st.write("No key positives identified.")

                with col2:
                    st.subheader("Forward Guidance")
                    st.write(summary.forward_guidance)
                    
                    st.subheader("Key Concerns / Challenges")
                    if summary.key_concerns:
                        for con in summary.key_concerns:
                            st.write(f"⚠️ {con}")
                    else:
                        st.write("No key concerns identified.")
                        
                st.divider()
                
                col3, col4 = st.columns(2)
                with col3:
                    st.subheader("Capacity Utilization Trends")
                    st.write(summary.capacity_utilization_trends)
                    
                with col4:
                    st.subheader("Growth Initiatives")
                    if summary.growth_initiatives:
                        for init in summary.growth_initiatives:
                            st.write(f"🚀 {init}")
                    else:
                        st.write("No new growth initiatives identified.")

            except Exception as e:
                st.error("❌ **Extraction Failed**")
                st.warning(f"Technical Details:\n\n{e}")
                if "429" in str(e) or "quota" in str(e).lower():
                    st.info("💡 **Tip:** Free-tier AI limits are very strict. This usually clears up if you wait 1–2 minutes.")
