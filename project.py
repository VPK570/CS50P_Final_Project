import re
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------
# Backend Functions
# ----------------------

def clean_text(raw_text):
    """Cleans text: lowercasing, removing HTML, URLs, numbers, stopwords, lemmatization"""
    text = str(raw_text).lower()
    text = re.sub('<.*?>', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub('[0-9]+', '', text)
    
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text)
    
    stop_words = set(stopwords.words('english'))
    filtered_words = [w for w in tokens if len(w) > 2 and w not in stop_words]
    
    lemmatizer = WordNetLemmatizer()
    lemma_words = [lemmatizer.lemmatize(w) for w in filtered_words]
    
    return " ".join(lemma_words)


def generate_embedding(cleaned_texts):
    """
    Converts list of cleaned texts into TF-IDF vectors
    """
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(cleaned_texts)
    return vectors


def compute_similarity(resume_vector, job_vector):
    """Computes cosine similarity between a resume vector and job vector"""
    similarity = cosine_similarity(resume_vector, job_vector)
    return similarity[0][0]


# ----------------------
# Streamlit Frontend
# ----------------------

st.title("Resume Analyzer (ATS Scoring)")

# Job Description Input
job_description = st.text_area(
    "Enter Job Description",
    height=200,
    placeholder="Paste the job description here..."
)

# Resume Upload
uploaded_files = st.file_uploader(
    "Upload Resume Files (.txt)", 
    type=["txt"], 
    accept_multiple_files=True
)

# Analyze Button
if st.button("Analyze Resumes"):

    if not job_description:
        st.warning("Please enter a job description.")
    elif not uploaded_files:
        st.warning("Please upload at least one resume.")
    else:
        # Prepare resumes
        resumes_text = []
        file_names = []
        for file in uploaded_files:
            content = file.read().decode("utf-8")
            cleaned_resume = clean_text(content)
            resumes_text.append(cleaned_resume)
            file_names.append(file.name)

        # Clean job description
        cleaned_job = clean_text(job_description)

        # Generate embeddings (all resumes + job description)
        vectors = generate_embedding(resumes_text + [cleaned_job])
        job_vector = vectors[-1]

        # Compute ATS scores
        st.subheader("ATS Scores")
        for i, resume_vector in enumerate(vectors[:-1]):
            similarity = compute_similarity(resume_vector, job_vector)
            ats_score = round(similarity * 100, 2)  # scale to 0-100
            st.write(f"**{file_names[i]}:** {ats_score}/100")
