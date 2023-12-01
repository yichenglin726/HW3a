import streamlit as st
import camelot
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import tempfile

st.title('PDF Table Extractor')

uploaded_file = st.file_uploader("Choose your PDF file", type="pdf")
query = st.text_input("Enter your query")

if uploaded_file is not None and query:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_pdf:
        tmp_pdf.write(uploaded_file.read())
        tables = camelot.read_pdf(tmp_pdf.name, pages='all')

    # Extract text from tables and vectorize
    table_texts = [table.df.to_string() for table in tables]
    vectorizer = TfidfVectorizer()
    vectorizer.fit(table_texts)
    query_vector = vectorizer.transform([query])

    # Calculate cosine similarity and find the most relevant table
    similarities = cosine_similarity(vectorizer.transform(table_texts), query_vector)
    most_relevant_table_index = similarities[:, 0].argmax()
    most_relevant_table = tables[most_relevant_table_index].df

    # Display the most relevant table
    st.write("Most relevant table:")
    st.dataframe(most_relevant_table)

# Additional logic for handling the case where a file or query is not provided
# can also be added based on your requirements.