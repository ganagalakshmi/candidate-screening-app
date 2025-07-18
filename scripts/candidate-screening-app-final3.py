import streamlit as st
import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download NLTK resources
nltk.download('punkt')

def preprocess_text(text):
    if isinstance(text, str):
        # Tokenize the text
        tokens = nltk.word_tokenize(text.lower())
        
        # Filter out punctuation and numbers
        tokens = [word for word in tokens if word.isalpha()]
        
        # Join tokens back into a string
        processed_text = ' '.join(tokens)
        
        return processed_text
    else:
        return ''  # Return an empty string for non-string values

def calculate_similarity(job_description, df):
    # Preprocess the resumes and job description
    df['Processed_Resume'] = df['Resume_str'].apply(preprocess_text)
    job_description_processed = preprocess_text(job_description)

    # Initialize TF-IDF vectorizer with n-grams
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_df=0.85)

    # Fit and transform resumes
    resume_vectors = vectorizer.fit_transform(df['Processed_Resume'])

    # Transform job description
    job_vector = vectorizer.transform([job_description_processed])

    # Calculate cosine similarity between job description and resumes
    similarity_scores = cosine_similarity(job_vector, resume_vectors)

    # Add similarity scores to DataFrame
    df['Similarity'] = similarity_scores.flatten()

    # Preserve the original index
    df['Original_Index'] = df.index

    # Sort DataFrame by similarity scores in descending order
    df_sorted = df.sort_values(by='Similarity', ascending=False).reset_index(drop=True)

    # Add rank column
    df_sorted['Rank'] = df_sorted.index + 1

    # Select and return the relevant columns
    return df_sorted[['Original_Index', 'ID', 'Similarity', 'Rank']].head(10)

# Streamlit app
def main():
    st.title("📄 Candidate Screening and Ranking App")

    uploaded_file = st.file_uploader("Upload resume data (CSV)", type="csv")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

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
