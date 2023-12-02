import streamlit as st
import main as m
import tempfile
import os

def main():
    st.title('PDF Table Searching')

    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    search_query = st.text_input("Enter your search query")
    search_engine = m.SearchEngine()

    if st.button('Search'):
        if uploaded_file is not None and search_query != "":
            # Create a temporary file and save the uploaded file to it
            saving_path = os.path.dirname(os.path.abspath(__file__))
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf', dir=saving_path) 
            tfile.write(uploaded_file.read())
            temp_path = tfile.name
            # Perform the search operation here
            st.write('Performing search...')
            table, score = search_engine.search(search_query, temp_path)
            st.write('Most relevant table with score {}:'.format(score[0]))
            st.table(table)
        else:
            st.write('Please upload a file and enter a search query.')

if __name__ == "__main__":
    main()