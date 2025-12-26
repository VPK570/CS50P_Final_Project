# Resume ATS Analyzer

A simple Applicant Tracking System (ATS) Resume Analyzer built using **Python**, **Streamlit**, and **NLP** techniques.
This project compares resumes against a job description and generates an **ATS compatibility score out of 100**.

---

## Features

- Upload multiple resume files (`.txt`)
- Paste a job description
- Text preprocessing (cleaning, stopword removal, lemmatization)
- TF-IDF based vectorization
- Cosine similarity scoring
- ATS score displayed out of 100
- Simple Streamlit web interface
- Includes unit tests using `pytest`

---

## Project Structure

```
CS50P_FINAL_PROJECT/
│
├── resume/
│   ├── job_desc.txt
│   ├── resume1.txt
│   ├── resume2.txt
│
├── project.py
├── test_project.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Requirements

- Python 3.9 or above
- pip

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/VPK570/CS50P_Final_Project.git
cd CS50P_FINAL_PROJECT
```

### 2. Create & Activate Virtual Environment (Recommended)

```bash
python -m venv venv
```

**Windows**
```bash
venv\Scripts\activate
```

**macOS / Linux**
```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download Required NLTK Data

Run Python once:

```python
import nltk
nltk.download("stopwords")
nltk.download("wordnet")
```

---

## Run the Application

```bash
streamlit run project.py
```

The Streamlit app will open automatically in your browser.

---

## How to Use

1. Paste the **Job Description** into the text box
2. Upload one or more `.txt` resume files
3. Click **Analyze Resumes**
4. View ATS scores out of 100

---

## Example Output

```
ATS Scores
resume1.txt: 87.45/100
resume2.txt: 42.13/100
```

---

## License

This project is for educational purposes.
