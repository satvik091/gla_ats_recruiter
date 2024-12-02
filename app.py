
import streamlit as st
import google.generativeai as genai
import PyPDF2
import os
from typing import List, Dict
import pandas as pd

# Configuration for Gemini API
GOOGLE_API_KEY = ("AIzaSyBPDNB9oDlVpJlTdEkEnc7vWv_CsAZiVQ0")
if not GOOGLE_API_KEY:
    st.error("Please set GOOGLE_API_KEY in Streamlit secrets")
    st.stop()

genai.configure(api_key=GOOGLE_API_KEY)

st.set_page_config(page_title="GLA Recruiter-Based Resume Ranking App", page_icon=":guardsman:")

def extract_text_from_pdf(pdf_file):
    """
    Extract text from PDF file
    """
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = " ".join([page.extract_text() for page in pdf_reader.pages])
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return ""

def generate_job_match_score(job_description: str, resume_text: str) -> float:
    """
    Use Gemini model to generate a job match score
    """
    try:
        model = genai.GenerativeModel('gemini-pro')

        prompt = f"""Analyze the following job description and resume,
        and provide a numerical score (0-100) representing how well
        the resume matches the job requirements. Consider:
        1. Relevant skills
        2. Professional experience
        3. Alignment with job responsibilities
        4. Keyword match

        Job Description:
        {job_description}

        Resume:
        {resume_text}

        Return only the numerical score between 0 and 100.
        """

        response = model.generate_content(prompt)

        try:
            score = float(response.text.strip())
        except ValueError:
            # Fallback if direct parsing fails
            score = 50  # Default mid-range score

        return max(0, min(score, 100))  # Ensure score is between 0-100

    except Exception as e:
        st.error(f"Error generating score: {e}")
        return 50  # Default score on error

def main():
    st.title("📄 Resume Ranking Application")

    # Job Description PDF Upload
    st.header("1. Upload Job Description PDF")
    job_description_pdf = st.file_uploader("Choose job description PDF",
                                           type=['pdf'],
                                           key="job_desc_upload")

    # Job Description Text (for verification or manual input)
    job_description = ""
    if job_description_pdf:
        job_description = extract_text_from_pdf(job_description_pdf)
        st.text_area("Extracted Job Description", value=job_description, height=200)

    # Resume PDF Upload
    st.header("2. Upload Resume PDFs")
    uploaded_resumes = st.file_uploader("Choose resume PDF files",
                                        type=['pdf'],
                                        accept_multiple_files=True,
                                        key="resume_upload")

    top_n = st.number_input("Number of Top Resumes", min_value=1, step=1)

    if st.button("Rank Resumes") and job_description and uploaded_resumes:
        with st.spinner('Analyzing Resumes...'):
            # Process resumes
            resume_data = []
            for resume_file in uploaded_resumes:
                resume_text = extract_text_from_pdf(resume_file)
                score = generate_job_match_score(job_description, resume_text)

                resume_data.append({
                    'Filename': resume_file.name,
                    'Match Score': score
                })

            # Create DataFrame and sort
            df = pd.DataFrame(resume_data)
            df_sorted = df.sort_values('Match Score', ascending=False)

            # Display results
            st.header("3. Resume Ranking Results")
            st.dataframe(df_sorted, use_container_width=True)

            # Highlight top candidates
            top_candidates = df_sorted.head(top_n)
            st.subheader("Top Candidates")
            for _, row in top_candidates.iterrows():
                st.metric(label=row['Filename'],
                          value=f"{row['Match Score']:.2f}%")

if __name__ == "__main__":
    main()
