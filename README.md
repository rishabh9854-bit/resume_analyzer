📄 Resume Analyzer

Resume Analyzer is a web application that helps job seekers evaluate how closely their resumes match a given job description. It extracts text from PDF documents, processes the content using Natural Language Processing (NLP) techniques, and calculates a similarity score to help users understand how relevant their resumes are to a target role.

Built with Python, FastAPI, and Scikit-learn, the project demonstrates how text preprocessing, TF-IDF vectorization, and cosine similarity can be used to automate resume-to-job-description matching.

✨ Features

- 📄 PDF Text Extraction: Extracts text from resume and job description PDFs.
- 🧹 Text Preprocessing: Cleans extracted text by normalizing case, removing unnecessary characters, and handling URLs and email addresses.
- 🔍 TF-IDF Vectorization: Converts text documents into numerical representations based on word importance.
- 📊 Similarity Scoring: Uses cosine similarity to measure the textual similarity between a resume and a job description.
- 🌐 Web Interface: Provides a browser-based interface for uploading PDF files and viewing analysis results.
- ⚡ FastAPI Backend: Handles file uploads and processes documents through a Python-based API.

🛠️ Tech Stack

Technology| Purpose
Python| Core application logic
FastAPI| Backend API and request handling
pdfplumber| Extracting text from PDF files
Scikit-learn| TF-IDF vectorization and cosine similarity
NumPy| Numerical operations
HTML, CSS, JavaScript| Web interface

⚙️ How It Works

1. Upload: Provide a resume and a job description as PDF documents.
2. Extract: Extract readable text from both PDFs.
3. Preprocess: Clean and normalize the extracted text.
4. Vectorize: Transform the text into TF-IDF feature vectors.
5. Compare: Calculate cosine similarity between the two vectors.
6. Analyze: Display a similarity score indicating how closely the documents match textually.

🧠 Core Concepts

- TF-IDF (Term Frequency–Inverse Document Frequency): Represents documents numerically by assigning importance to terms based on their frequency and distribution across the documents.
- Cosine Similarity: Measures the cosine of the angle between two document vectors to estimate their textual similarity.
- NLP Preprocessing: Improves text consistency before converting documents into numerical representations.

🚀 Getting Started

Prerequisites

- Python 3.11 or another compatible Python version
- pip
- Git

Installation

Clone the repository:

git clone https://github.com/rishabh9854-bit/resume_analyzer.git
cd resume_analyzer

Create and activate a virtual environment:

python -m venv venv

On Windows:

venv\Scripts\activate

Install the required dependencies:

pip install fastapi uvicorn python-multipart pdfplumber scikit-learn numpy

Run the Application

Start the FastAPI development server using the appropriate Python module and application object for your project:

uvicorn main:app --reload

Open the application at:

"http://127.0.0.1:8000"

If your entry-point file or frontend setup differs, adjust the command and open the frontend according to your project configuration.

🎯 Use Cases

- Evaluating resume relevance for a specific job opening.
- Identifying whether a resume contains terminology related to a job description.
- Experimenting with NLP-based document comparison.
- Learning how to build and integrate a Python backend with a web interface.

⚠️ Limitations

- The similarity score measures textual similarity, not a candidate's actual qualifications or probability of getting hired.
- Different wording can reduce similarity even when two passages have similar meanings.
- Scanned PDFs may require OCR to extract readable text.
- A high similarity score does not guarantee ATS compatibility or job suitability.

🔮 Future Improvements

- Highlight important keywords present in the job description but missing from the resume.
- Provide a more detailed breakdown of matching and non-matching terms.
- Support additional document formats such as DOCX.
- Add automated tests and improved error handling for invalid or unreadable PDFs.
- Improve the interface with a clearer, more detailed analysis report.

👨‍💻 Author

Rishabh Verma

GitHub: "@rishabh9854-bit" (https://github.com/rishabh9854-bit)

---

Built as a practical project to explore NLP, text similarity, PDF processing, and backend development with FastAPI.
