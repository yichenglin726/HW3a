import streamlit as st
from transformers import BertModel, BertTokenizer
import numpy as np
import pandas as pd
import camelot
import os
import tempfile

# BERT Vectorizer Class
class BERTVectorizer:
    def __init__(self, model_name='shibing624/text2vec-base-chinese'):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)

    def text2vector(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", max_length=128, truncation=True, padding='max_length')
        outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).detach().numpy()

# PDF Table Extractor Class
class PDFTableExtractor:
    def extract_tables(self, pdf_file):
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_pdf:
                temp_pdf.write(pdf_file.read())
                temp_pdf_path = temp_pdf.name

            tables = camelot.read_pdf(temp_pdf_path, pages="all")
            return [table.df for table in tables]
        except Exception as e:
            print(f"Error reading PDF file: {e}")
            return []
        finally:
            os.remove(temp_pdf_path)

# Table Finder Class
class TableFinder:
    def __init__(self):
        self.pdf_extractor = PDFTableExtractor()
        self.vectorizer = BERTVectorizer()
    
    @staticmethod
    def cosine_sim(vec1, vec2):
        vec1, vec2 = vec1.flatten(), vec2.flatten()
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    @staticmethod
    def process_table_text(table):
        return table.to_string().replace("\\n", "")

    @staticmethod
    def get_max_sec(max_sim, sec_sim, similarity):
        if max_sim <= similarity:
            sec_sim = max_sim
            max_sim = similarity
        elif sec_sim <= similarity:
            sec_sim = similarity
        return (max_sim, sec_sim)

    def find_most_relevant_tables(self, keyword, pdf_file, top_n=2, thresh=-1):
        tables = self.pdf_extractor.extract_tables(pdf_file)
        if not tables:
            return "No tables found or error in reading PDF."

        keyword_vector = self.vectorizer.text2vector(keyword)
        table_similarities = []

        for idx, table in enumerate(tables):
            table_text = self.process_table_text(table)
            table_text_list = table_text.split()

            max_sim, sec_sim = -1, -1
            for text in table_text_list:
                text_vector = self.vectorizer.text2vector(text)
                similarity = self.cosine_sim(keyword_vector, text_vector)
                max_sim, sec_sim = self.get_max_sec(max_sim, sec_sim, similarity)

            max_sim = (np.exp(max_sim) + np.exp(sec_sim)) / (2 * np.e)
            print(f"table({idx}) similarities:", max_sim)

            if max_sim >= thresh:
                table_similarities.append((max_sim, table))

        best_tables = [table for _, table in sorted(table_similarities, key=lambda x: x[0], reverse=True)[:top_n]]
        return best_tables if best_tables else ["No table closely matches the keyword."]

# Streamlit UI
def main():
    st.title('PDF Table Finder using text2vec')
    st.write('This app uses text2vec to find the most relevant tables in a PDF document based on a keyword.')

    pdf_file = st.file_uploader("Upload a PDF", type=['pdf'])
    keyword = st.text_input("Enter a keyword")

    if st.button('Find Tables'):
        if pdf_file is not None and keyword:
            table_finder = TableFinder()
            matched_tables = table_finder.find_most_relevant_tables(keyword, pdf_file)
            if matched_tables:
                for table in matched_tables:
                    if isinstance(table, pd.DataFrame):  
                        st.table(table)
                    else:
                        st.text(table)  
            else:
                st.write("No relevant tables found.")
        else:
            st.write("Please upload a PDF file and enter a keyword.")

if __name__ == '__main__':
    main()


