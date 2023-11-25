import os
import camelot
import streamlit as st
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util

class pdf2text:
    def __init__(self):
        pass

    def __call__(self, pdf_file):
        tables = camelot.read_pdf(pdf_file, pages="all")
        streams = camelot.read_pdf(pdf_file, flavor='stream', pages='all')
        titles = list(sorted(set([s for s in sum([sum(stream.df.values.tolist(),[]) for stream in streams],[]) if str(s).startswith('ai_tables')])))
        docs = []
        for id, table in enumerate(tables):
            docs.append({
                "text":table.df.to_string().replace('\\n',''),
                "df":table.df.replace('\n','',regex=True),
                "file":os.path.basename(pdf_file)
            })
        return docs


class text2vector:
    def __init__(self):
        self.model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v2')

    def __call__(self, text):
        return self.model.encode(text)


class cosine_sim:
    def __init__(self):
        pass

    def __call__(self, vector_from_table, vector_from_keyword):
        return util.dot_score(vector_from_keyword,  vector_from_table)

@st.cache_data
def load_pdf_files(pdf_dir_path):
    pdf_parser = pdf2text()
    pdfs = []
    for pdf_path in os.listdir(pdf_dir_path):
        pdfs.append(pdf_parser(pdf_dir_path+'/'+pdf_path))

    t2v = text2vector()
    doc_embeds = []
    pg = st.progress(0, text='loading documents...')
    finished, total = 1, sum([len(doc) for doc in pdfs])
    for docs in pdfs:
        for doc in docs:
            doc_embeds.append((t2v(doc['text']),doc))
            pg.progress(finished/total)
            finished += 1

    return doc_embeds

def main():
    # modify for different pdf directory
    pdf_dir_path = 'docs/'

    st.title('Document Intelligence')
    st.write('An artificial intelligence that searches in which table in the given pdf files has the desired information.')
    st.write('By R12922040 陳威宇')
    st.write('Currently reading pdfs from {}'.format(os.path.join(os.path.dirname(os.path.realpath(__file__)),pdf_dir_path)))
    doc_embeds = load_pdf_files(pdf_dir_path)    


    sim = cosine_sim()
    t2v = text2vector()
    keyword = st.text_input("Keyword", key='keyword')
    keyword_embed = t2v(keyword)

    scores = []
    pg2 = st.progress(0, text='calculating similarities...')
    finished = 1
    for i, doc in enumerate(doc_embeds):
        scores.append((sim(doc[0], keyword_embed), doc[1]))
        pg2.progress(finished/(len(doc_embeds)))
        finished += 1

    if keyword != "":
        st.write("Top 3 related tables:")
        scores = sorted(scores, key=lambda x: x[0], reverse=True)
        for score in scores[:3]:
            s1, s2 = st.columns((1,1))
            with s1:
                st.write('score: ', score[0].item())
            with s2:
                st.markdown("<div style='text-align: right;'>{}</div>".format(score[1]['file']), unsafe_allow_html=True)
            st.table(score[1]['df'])
            

if __name__ == "__main__":
    main()