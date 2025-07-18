import streamlit as st
import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download NLTK resources
nltk.download('punkt')

def calculate_similarity(job_description, df):
    # Ensure 'Resume_str' is in lower case
    df['Resume_str'] = df['Resume_str'].astype(str).str.lower()

    # Handle missing values by filling with a placeholder
    df['Resume_str'] = df['Resume_str'].fillna('No resume provided')

    # Initialize TF-IDF vectorizer
    vectorizer = TfidfVectorizer(stop_words='english')

    # Fit and transform resumes
    resume_vectors = vectorizer.fit_transform(df['Resume_str'])

    # Transform job description
    job_vector = vectorizer.transform([job_description.lower()])

    # Calculate cosine similarity between job description and resumes
    similarity_scores = cosine_similarity(job_vector, resume_vectors)

    # Add similarity scores to DataFrame
    df['Similarity'] = similarity_scores.flatten()

    # Sort DataFrame by similarity scores in descending order
    df_sorted = df.sort_values(by='Similarity', ascending=False).reset_index(drop=True)

    # Add rank column
    df_sorted['Rank'] = df_sorted.index + 1

    # Return top 10 matches with rank
    return df_sorted[['ID', 'Resume_str', 'Rank']].head(10)

# Streamlit app
def main():
    st.title("📄 Resume Matching App 🎯")

    uploaded_file = st.file_uploader("Upload resume data (CSV)", type="csv")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        # Check for required columns
        if 'ID' not in df.columns or 'Resume_str' not in df.columns:
            st.error("CSV file must contain 'ID' and 'Resume_str' columns.")
            return

        st.write("Uploaded Resume Data Preview", df.head())

        job_description = st.text_area("Enter job description 📝")

        if st.button("Match Resumes 🔍"):
            if job_description.strip() == "":
                st.error("Job description cannot be empty.")
            else:
                top_matches = calculate_similarity(job_description, df)
                if not top_matches.empty:
                    st.subheader("Top 10 Matches 🚀")
                    st.write(top_matches)
                else:
                    st.write("No matches found.")

if __name__ == "__main__":
    main()
