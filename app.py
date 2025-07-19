import streamlit as st
import pandas as pd
import nltk
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download NLTK resources
nltk.download('punkt')

def calculate_similarity(job_description, df, model):
    df['Resume_str'] = df['Resume_str'].astype(str).str.lower()

    # Handle missing values
    df['Resume_str'] = df['Resume_str'].fillna('No resume provided')

    # Preserve original index
    df['Original_Index'] = df.index

    # TF-IDF vectorizer
    vectorizer = TfidfVectorizer(stop_words='english')
    resume_vectors = vectorizer.fit_transform(df['Resume_str'])
    job_vector = vectorizer.transform([job_description.lower()])

    # Cosine similarity
    similarity_scores = cosine_similarity(job_vector, resume_vectors)
    df['Similarity'] = similarity_scores.flatten()

    # Filter and rank
    df_filtered = df[df['Similarity'] > 0.5]
    df_sorted = df_filtered.sort_values(by='Similarity', ascending=False).reset_index(drop=True)
    df_sorted['Rank'] = df_sorted.index + 1

    return df_sorted[['Original_Index', 'Category', 'Resume_str', 'Rank', 'Similarity']].head(10)

# Streamlit app
def main():
    st.title("📄 Candidate Screening and Ranking App")

    # ✅ Use relative path here for cloud compatibility
    model = joblib.load('best_model.pkl')

    uploaded_file = st.file_uploader("Upload resume data (CSV)", type="csv")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        if 'Category' not in df.columns or 'Resume_str' not in df.columns:
            st.error("CSV file must contain 'Category' and 'Resume_str' columns.")
            return

        st.write("Uploaded Resume Data Preview", df.head())

        job_description = st.text_area("Enter job description 📝")

        if st.button("Filter Resumes 🔍"):
            if job_description.strip() == "":
                st.error("Job description cannot be empty.")
            else:
                top_matches = calculate_similarity(job_description, df, model)
                if not top_matches.empty:
                    st.subheader("Top 10 Matches (Similarity > 50%) 🚀")
                    st.write(top_matches.set_index('Original_Index'))
                else:
                    st.write("No matches found with similarity greater than 50%.")

if __name__ == "__main__":
    main()
