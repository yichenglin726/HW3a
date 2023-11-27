import camelot
from sentence_transformers import SentenceTransformer, util
import streamlit as st
import os


class pdf2text:
    def __init__(self):
        pass

    def __call__(self, pdf_file):
        tables = camelot.read_pdf(pdf_file, pages='all')
        texts = {}
        for table in tables:
            text = table.df.to_string().strip()
            texts[text] = table

        return texts


class text2vector:
    def __init__(self):
        self.model = SentenceTransformer('shibing624/text2vec-base-chinese')

    def __call__(self, text):
        return self.model.encode(text, convert_to_tensor=False)


class cosine_sim:
    def __init__(self):
        pass

    def __call__(self, vector_from_table, vector_from_keyword):
        util.dot_score(vector_from_table, vector_from_keyword)
        return cos

def main():
    st.title('BDS HW3a')
    st.header('Upload pdf file and enter keywords to search')
    submit_file = st.file_uploader('Upload PDF File', type=['pdf'], accept_multiple_files=True)
    if submit_file != []:
        pdf = st.selectbox('Choose PDF File', submit_file)
        if pdf:
            pdf_file = os.path.join('docs', pdf.name)
            keyword = st.text_input('keyword', 'Enter Keywords')
    
            if st.button('Run'):
                table = find_table(keyword, pdf_file)
                st.write('Result')
                st.write(table)


def find_table(keyword, pdf_file):
    p2t = pdf2text()
    t2v = text2vector()
    cos = cosine_sim()

    texts = p2t(pdf_file)

    # text_vector = t2v(table_text)
    keyword_vector = t2v(keyword)
    
    score, key = -1, ''
    for text in texts.keys():
        text_vector = t2v(text)
        s = cos(text_vector, keyword_vector)
        if s > score:
            score = s
            key = text
    
    table = texts[key]

    return table


if __name__ == '__main__':
    main()