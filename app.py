# """
# Usage: streamlit run app.py
# """
import streamlit as st
from search import search
import os
import tempfile

# Title of the web app
st.title('Document Intelligence')

# Description of the web app
st.write('Enter a keyword and upload a PDF file to search for the most similar table in the document.')

# Input for the keyword
keyword = st.text_input('Enter the keyword')

# Input for the PDF file
pdf_file = st.file_uploader('Upload a PDF file', type=['pdf'])

# Button to start the process
if st.button('Start'):
    if keyword and pdf_file:
        # Create a temporary file
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') 
        tfile.write(pdf_file.read())

        # Call your main function
        result = search(keyword, tfile.name)

        # Display the result
        st.write('Keyword: ', keyword)
        st.write('Most relevant table: ')
        st.table(result)

        # Delete the temporary file
        os.unlink(tfile.name)
    else:
        st.write('Please enter a keyword and upload a PDF file.')


