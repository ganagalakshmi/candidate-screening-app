import streamlit as st
import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# Download NLTK resources
nltk.download('punkt')

def preprocess_text(text):
    # Remove special characters, punctuation, and numbers
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    # Convert to lowercase
    text = text.lower()
    return text

def calculate_similarity(job_description, df):
    # Preprocess text
    df['Resume_str'] = df['Resume_str'].astype(str).apply(preprocess_text)
    job_description = preprocess_text(job_description)

    # Handle missing values by filling with a placeholder
    df['Resume_str'] = df['Resume_str'].fillna('No resume provided')

    # Preserve the original index
    df['Original_Index'] = df.index

    # Initialize TF-IDF vectorizer with expanded parameters
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 3), max_features=15000, min_df=1, max_df=0.85)

    # Fit and transform resumes
    resume_vectors = vectorizer.fit_transform(df['Resume_str'])

    # Transform job description
    job_vector = vectorizer.transform([job_description])

    # Calculate cosine similarity between job description and resumes
    similarity_scores = cosine_similarity(job_vector, resume_vectors).flatten()

    # Add similarity scores to DataFrame
    df['Similarity'] = similarity_scores

    # Sort DataFrame by similarity scores in descending order
    df_sorted = df.sort_values(by='Similarity', ascending=False).reset_index(drop=True)

    # Add rank column
    df_sorted['Rank'] = df_sorted.index + 1

    # Select and return the relevant columns
    return df_sorted[['Original_Index', 'ID', 'Resume_str', 'Similarity', 'Rank']].head(10)

# Streamlit app
def main():
    st.title("📄 Candidate Screening and Ranking App")

    uploaded_file = st.file_uploader("Upload resume data (CSV)", type="csv")

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Error reading the CSV file: {e}")
            return

        # Check for required columns
        if 'ID' not in df.columns or 'Resume_str' not in df.columns:
            st.error("CSV file must contain 'ID' and 'Resume_str' columns.")
            return

        st.write("Uploaded Resume Data Preview", df.head())

        job_description = st.text_area("Enter job description 📝")

        if st.button("Filter Resumes 🔍"):
            if job_description.strip() == "":
                st.error("Job description cannot be empty.")
            else:
                top_matches = calculate_similarity(job_description, df)
                if not top_matches.empty:
                    st.subheader("Top 10 Matches 🚀")
                    st.write(top_matches.set_index('Original_Index'))
                else:
                    st.write("No matches found.")

if __name__ == "__main__":
    main()
