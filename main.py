import os
import camelot
import streamlit as st
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util

class PdfToText:
    def __init__(self):
        pass

    def extract_tables(self, pdf_file):
        return camelot.read_pdf(pdf_file, pages="all")

    def extract_streams(self, pdf_file):
        return camelot.read_pdf(pdf_file, flavor='stream', pages='all')

    def extract_titles(self, streams):
        titles = []
        for stream in streams:
            for value in stream.df.values.tolist():
                if str(value).startswith('ai_tables'):
                    titles.append(value)
        return sorted(set(titles))

    def create_docs(self, tables, pdf_file):
        docs = []
        for table in tables:
            doc = {
                "text": table.df.to_string().replace('\\n', ''),
                "df": table.df.replace('\n', '', regex=True),
                "file": os.path.basename(pdf_file)
            }
            docs.append(doc)
        return docs

    def __call__(self, pdf_file):
        tables = self.extract_tables(pdf_file)
        streams = self.extract_streams(pdf_file)
        self.extract_titles(streams)
        return self.create_docs(tables, pdf_file)


class TextToVector:
    def __init__(self, model_name='sentence-transformers/distiluse-base-multilingual-cased-v2'):
        self.model = self.load_model(model_name)

    def load_model(self, model_name):
        return SentenceTransformer(model_name)

    def encode_text(self, text):
        return self.model.encode(text)

    def __call__(self, text):
        return self.encode_text(text)



class SimilarityScore:
    def __init__(self):
        pass

    def calculate_dot_score(self, vector_from_table, vector_from_keyword):
        return util.dot_score(vector_from_keyword, vector_from_table)

    def __call__(self, vector_from_table, vector_from_keyword):
        return self.calculate_dot_score(vector_from_table, vector_from_keyword)

@st.cache_data
def load_pdf_files(pdf_dir_path):
    pdf_parser = PdfToText()
    pdfs = load_pdfs_from_directory(pdf_dir_path, pdf_parser)

    t2v = TextToVector()
    doc_embeds = create_document_embeddings(pdfs, t2v)

    return doc_embeds

def load_pdfs_from_directory(directory_path, pdf_parser):
    pdfs = []
    for pdf_path in os.listdir(directory_path):
        pdfs.append(pdf_parser(os.path.join(directory_path, pdf_path)))
    return pdfs

def create_document_embeddings(pdfs, text_to_vector):
    doc_embeds = []

    finished, total = 1, sum([len(doc) for doc in pdfs])
    for docs in pdfs:
        for doc in docs:
            doc_embeds.append((text_to_vector(doc['text']), doc))

            finished += 1
    return doc_embeds


def calculate_scores(doc_embeds, sim, keyword_embed):
    scores = []
    finished = 1
    for i, doc in enumerate(doc_embeds):
        scores.append((sim(doc[0], keyword_embed), doc[1]))
        finished += 1
    return scores

def display_intro():
    st.sidebar.title('Doc AI')
    st.sidebar.write('Using AI to search tables in pdfs')

def get_user_input():
    pdf_dir_path = st.sidebar.text_input("PDF Directory", 'docs/')
    keyword = st.sidebar.text_input("Keyword", key='keyword')
    return pdf_dir_path, keyword

def display_top_scores(scores):
    st.title("Top 3 related tables:")
    scores = sorted(scores, key=lambda x: x[0], reverse=True)
    for score in scores[:3]:
        st.subheader('Score: {}'.format(score[0].item()))
        st.markdown("<div style='text-align: right;'>{}</div>".format(score[1]['file']), unsafe_allow_html=True)
        st.dataframe(score[1]['df'])

def main():
    display_intro()
    pdf_dir_path, keyword = get_user_input()

    if keyword != "":
        doc_embeds = load_pdf_files(pdf_dir_path)    

        sim = SimilarityScore()
        t2v = TextToVector()
        keyword_embed = t2v(keyword)

        scores = calculate_scores(doc_embeds, sim, keyword_embed)
        display_top_scores(scores)

if __name__ == "__main__":
    main()