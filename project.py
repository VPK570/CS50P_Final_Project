import re
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()


def clean_text(raw_text):
    text = str(raw_text).lower()
    text = re.sub('<.*?>', '', text)
    text = re.sub(r'http\S+', '', text)

    tokenizer = RegexpTokenizer(r'[a-zA-Z][a-zA-Z\+\.\#]*')
    tokens = tokenizer.tokenize(text)

    stop_words = set(stopwords.words('english'))
    filtered_words = [w for w in tokens if w not in stop_words]

    lemmatizer = WordNetLemmatizer()
    lemma_words = [lemmatizer.lemmatize(w) for w in filtered_words]

    return " ".join(lemma_words)


def generate_embedding(texts):
    return model.encode(texts, normalize_embeddings=True)


def compute_similarity(resume_vector, job_vector):
    return cosine_similarity(
        [resume_vector],
        [job_vector]
    )[0][0]


st.title("Resume Analyzer (ATS Scoring)")

job_description = st.text_area(
    "Enter Job Description",
    height=200,
    placeholder="Paste the job description here..."
)

uploaded_files = st.file_uploader(
    "Upload Resume Files (.txt)",
    type=["txt"],
    accept_multiple_files=True
)

if st.button("Analyze Resumes"):

    if not job_description:
        st.warning("Please enter a job description.")
    elif not uploaded_files:
        st.warning("Please upload at least one resume.")
    else:
        resumes_text = []
        file_names = []

        for file in uploaded_files:
            content = file.read().decode("utf-8")
            cleaned_resume = clean_text(content)
            resumes_text.append(cleaned_resume)
            file_names.append(file.name)

        cleaned_job = clean_text(job_description)

        embeddings = generate_embedding(resumes_text + [cleaned_job])
        job_vector = embeddings[-1]

        st.subheader("ATS Scores")
        for i, resume_vector in enumerate(embeddings[:-1]):
            similarity = compute_similarity(resume_vector, job_vector)
            ats_score = round(similarity * 100, 2)
            st.write(f"**{file_names[i]}:** {round(ats_score)}/100")
