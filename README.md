# Resume ATS Analyzer
## Video Demo: 
https://youtu.be/3xJoIspfWBM

A simple Applicant Tracking System (ATS) Resume Analyzer built using **Python**, **Streamlit**, and **NLP** techniques.
This project compares resumes against a job description and generates an **ATS compatibility score out of 100**.

---

## Features

- Upload multiple resume files (`.txt`)
- Paste a job description
- Text preprocessing (cleaning, stopword removal, lemmatization)
- BERT based vectorization
- Cosine similarity scoring
- ATS score displayed out of 100
- Simple Streamlit web interface
- Includes unit tests using `pytest`

---

## Description
The Resume ATS Analyzer is a Python-based application designed to evaluate resumes against a job description using natural language processing (NLP) techniques. The project leverages text cleaning, lemmatization, and semantic similarity to compute an ATS score out of 100, giving users an indication of how well a resume aligns with the requirements of a given job. The goal of the project is to simulate the functionality of an applicant tracking system, commonly used in real-world recruitment, in an educational and interactive manner. It also serves as a practical exploration of Python programming, NLP preprocessing, and vector-based similarity calculations.

The main file of the project, project.py, contains all the functional logic and the Streamlit front-end interface. This file is structured to include a main() function that initializes the app and handles user inputs. Additional functions include clean_text(), which performs text preprocessing such as converting to lowercase, removing numbers, URLs, and punctuation, tokenization, stopword removal, and lemmatization. The generate_embedding() function converts cleaned texts into vector representations, initially using TF-IDF but optionally allowing for more advanced embeddings like BERT. The compute_similarity() function calculates the cosine similarity between resume vectors and the job description vector, which is then scaled to provide an ATS score out of 100. This design allows the application to be easily extended with new preprocessing steps or embeddings in the future.

The test_project.py file contains unit tests written with pytest. These tests validate that core functions behave as expected. For example, the tests check that clean_text() properly removes noise, handles empty input, and lemmatizes words; generate_embedding() returns vectors of correct shape; and compute_similarity() produces scores in the correct range. Writing these tests not only ensures the reliability of the functions but also aligns with best practices in software development and CS50P project requirements.

All resume and job description inputs are expected to be text files, stored in the resume/ directory. Users can upload one or more .txt files and provide a job description in the Streamlit interface. The app then outputs a list of ATS scores, allowing users to quickly identify which resumes are most aligned with the role. The project also includes a .gitignore to prevent committing unnecessary files such as virtual environments, cache files, and temporary logs, keeping the repository clean and manageable.

During development, certain design choices were made deliberately. The decision to keep all core functions in a single file simplifies the project structure and meets CS50P submission requirements. Using Streamlit as the front-end allows immediate interactive feedback without needing complex GUI programming. TF-IDF embeddings were chosen for simplicity and transparency, but the code structure allows for swapping in modern transformer-based embeddings if desired. The text preprocessing pipeline balances thorough cleaning with preserving meaningful words for accurate similarity calculations.

In conclusion, this project demonstrates practical applications of Python programming, NLP techniques, and vector-based similarity measures. It provides an accessible interface for users to analyze resumes, while also serving as a learning tool for anyone interested in text analysis, data processing, and software design. By combining functional Python code, thorough testing, and a user-friendly front end, the Resume ATS Analyzer fulfills the dual goals of education and practical utility.

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
