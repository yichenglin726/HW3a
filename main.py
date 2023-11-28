import os
import camelot
from sentence_transformers import SentenceTransformer, util
import streamlit as st

# Class to parse PDFs into text tables
class pdf2text:
    def __init__(self):
        pass

    def __call__(self, pdf_file):
        # Reading all pages of the PDF file
        tables = camelot.read_pdf(pdf_file, pages="all")
        texts = []
        for table in tables:
            # Append each table to the texts list
            texts.append(table.df)
        return texts

# Class to convert text to vector using a pre-trained model
class text2vector:
    def __init__(self):
        # Load the SentenceTransformer model
        self.model = SentenceTransformer('distiluse-base-multilingual-cased-v1')

    def __call__(self, text):
        # Convert the input text to a vector
        return self.model.encode(text)

# Class for computing cosine similarity
class cosine_sim:
    def __init__(self):
        pass

    def __call__(self, vector_from_table, vector_from_keyword):
        # Calculate and return the cosine similarity
        return util.cos_sim(vector_from_keyword, vector_from_table)

# Main function to run the Streamlit app
def main():
    st.title("R12922035 Document Intelligence")

    # Input for PDF file name or upload
    pdf = st.text_input("輸入PDF文件名稱或上傳新文件")
    upload_pdf = st.file_uploader("Upload YOUR PDF File", type=['pdf'])
    # Input for keyword
    keyword = st.text_input("輸入keywords")

    # Handling file path for input PDF
    if pdf:
        pdf_file_path = os.path.join('docs', pdf)
    if upload_pdf:
        pdf_file_path = os.path.join("docs", upload_pdf.name)
        with open(pdf_file_path, 'wb') as f:
            f.write(upload_pdf.read())

    # Main logic for processing PDF and keyword
    if (pdf or upload_pdf) and keyword:
        # Initialize the pdf2text class to parse the PDF file
        pdf_parser = pdf2text()
        # Parse the PDF file to extract tables
        tables = pdf_parser(pdf_file_path)

        # Initialize the text2vector class for converting text to vectors
        txt2vec = text2vector()
        # Convert the keyword to its vector representation
        keyword_vector = txt2vec(keyword)

        # Initialize the cosine_sim class for calculating cosine similarity
        cos_similarity = cosine_sim()
        # Variables to keep track of the table with the highest similarity score
        max_similarity = -float('inf')
        best_table = None
        best_i = None

        # Iterate over each table in the extracted tables
        for i, table in enumerate(tables):
            # Convert the text of the table to its vector representation
            vector_table = txt2vec(table.to_string())
            # Calculate the similarity score between the table and the keyword
            similarity_score = cos_similarity(vector_table, keyword_vector).item()

            # If the current table's similarity score is higher than the max found so far
            if similarity_score > max_similarity:
                # Update the max similarity score and store this table as the best match
                max_similarity = similarity_score
                best_table = table
                best_i = i + 1

        # Display the table with the highest similarity score
        st.write(f"Best Matching Table: ")
        # Show the best matching table in the Streamlit app
        st.dataframe(best_table)



if __name__ == "__main__":
    main()
