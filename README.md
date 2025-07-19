git# Candidate Screening and Ranking using NLP and Machine Learning 

The recruitment process is a critical function within organizations, directly impacting their ability to attract, hire, and retain top talent. Traditionally, candidate screening and ranking have been labor-intensive tasks, requiring human resources (HR) professionals to manually sift through large volumes of resumes to identify suitable candidates. This manual process is time-consuming, prone to bias, and often fails to adequately capture the qualifications and potential of candidates due to the sheer volume of applications.
With advancements in Natural Language Processing (NLP) and Machine Learning (ML), there is a significant opportunity to automate and enhance the candidate screening process. By leveraging these technologies, it is possible to develop an automated system that can efficiently analyze, rank, and screen candidates based on their resumes and job descriptions. This not only accelerates the recruitment process but also ensures a more objective and consistent evaluation of candidates.
This is a machine learning-powered web application that helps screen suitable candidates based on the given input parameters. Built using Python and Streamlit.

## 🚀 Features

- Upload candidates data file in csv or xls format
- Give the job description
- It'll predict candidate suitability using a trained ML model and list Top-10 candidates with >50% similarity (Can adjust according to your requirement)
- Easy-to-use Streamlit interface
- Responsive design for desktop

## 🛠 Technologies Used

- Python
- Streamlit
- Scikit-learn
- Pandas, NumPy
- Jupyter Notebooks

## 🧪 How to Run

1. **Clone the Repository**
   git clone https://github.com/ganagalakshmi/candidate-screening-app.git
   cd candidate-screening-app

2. **Create a Virtual Environment**
    python -m venv venv
    venv\Scripts\activate  # for Windows

3. **Install Dependencies**
    pip install -r requirements.txt

4. **Run the App
    streamlit run app.py

## 📂 Dataset
   
    The dataset used in this project can be downloaded from the link below:
    https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset