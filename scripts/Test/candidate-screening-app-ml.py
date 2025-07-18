import streamlit as st
import pandas as pd
import nltk
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

# Download NLTK resources
nltk.download('punkt')

def calculate_similarity(job_description, df, model):
    # Ensure 'Resume_str' is in lower case
    df['Resume_str'] = df['Resume_str'].astype(str).str.lower()

    # Handle missing values by filling with a placeholder
    df['Resume_str'] = df['Resume_str'].fillna('No resume provided')

    # Preserve the original index
    df['Original_Index'] = df.index

    # Initialize TF-IDF vectorizer
    vectorizer = TfidfVectorizer(stop_words='english')

    # Fit and transform resumes
    resume_vectors = vectorizer.fit_transform(df['Resume_str'])

    # Transform job description
    job_vector = vectorizer.transform([job_description.lower()])

    # Make predictions using the trained Gradient Boosting model
    predictions = model.predict(resume_vectors)

    # Add predictions to DataFrame
    df['Prediction'] = predictions

    # Sort DataFrame by predictions in descending order
    df_sorted = df.sort_values(by='Prediction', ascending=False).reset_index(drop=True)

    # Add rank column
    df_sorted['Rank'] = df_sorted.index + 1

    # Select and return the relevant columns
    return df_sorted[['Original_Index', 'ID', 'Resume_str', 'Rank', 'Prediction']].head(10)

# Streamlit app
def main():
    st.title("📄 Candidate Screening and Ranking App")

    # Load the best model (Gradient Boosting)
    model = joblib.load('C:/Users/Admin/candidate-screening-project/best_model.pkl')


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
                top_matches = calculate_similarity(job_description, df, model)
                if not top_matches.empty:
                    st.subheader("Top 10 Matches 🚀")
                    st.write(top_matches.set_index('Original_Index'))
                else:
                    st.write("No matches found.")

if __name__ == "__main__":
    main()
