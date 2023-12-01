import pdfplumber
import camelot
import pandas as pd
from scipy.spatial.distance import cosine
import streamlit as st
from itertools import chain
import os
from sentence_transformers import SentenceTransformer

class pdf2text:
    def __init__(self):
        pass

    def __call__(self, pdf_file):
        
        with pdfplumber.open(pdf_file) as pdf:
            lines = []
            for page in pdf.pages:
                lines.extend(page.extract_text().split('\n'))
            table_titles = [line for line in lines if 'ai_tables_#' in line]

        tables = list(camelot.read_pdf(pdf_file, pages='all'))
        merged_tables = []
        i = 1
        while (i < len(tables)):
            if (tables[i].rows[0][0] > 760 and tables[i-1].rows[-1][-1] < 50):
                merged_tables = merged_tables + [tables[i-1].df.values.tolist() + tables[i].df.values.tolist()]
                i += 2
            else:
                merged_tables = merged_tables + [tables[i-1].df.values.tolist()]
                if (i == len(tables) - 1):
                    merged_tables = merged_tables + [tables[i].df.values.tolist()]
                i+=1

        merged_tables = [[[col.replace('\n', '').replace(' ', '') for col in row] for row in table if row != ['']*len(row)] for table in merged_tables]
        return table_titles, merged_tables

class text2vector:
    def __init__(self):
        self.model = SentenceTransformer('distiluse-base-multilingual-cased-v1')

    def __call__(self, text):
        sentence = [text]
        embedding = self.model.encode(sentence)
        return embedding


class cosine_sim:
    def __init__(self):
        pass

    def __call__(self, vector_from_table, vector_from_keyword):
        return 1 - cosine(vector_from_table,vector_from_keyword)

folder_path = './docs'

filenames = os.listdir(folder_path)
pdf_parser = pdf2text()


transform = text2vector()

calculator = cosine_sim()
# exit()
st.title('Keyword-Based Document Intelligence')

# Input field for keyword
keyword = st.text_input('Enter keyword')

if st.button('Search'):
    if keyword:
        
        titles = []
        tables = []
        for filename in filenames:
            _titles, _tables = pdf_parser('./docs/'+filename)
            titles = titles + _titles
            tables = tables + _tables

        sentences = ['\n'.join([titles[index]] + list(chain.from_iterable(table))) for index, table in enumerate(tables)]
        vectors = [transform(sentence) for sentence in sentences]
        vectorKeyword = transform(keyword)

        maximumValue = 0
        maximumIndex = 0
        for index, vector in enumerate(vectors):
            score=  calculator(vectorKeyword[0], vector[0])
            if score > maximumValue:
                maximumIndex = index
                maximumValue = score
        
        st.subheader(titles[maximumIndex])
        st.table(pd.DataFrame(tables[maximumIndex]))


