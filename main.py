import camelot
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
import streamlit as st
import os


class pdf2text:
    def __init__(self):
        pass

    def __call__(self, pdf_file):
        tables = camelot.read_pdf(pdf_file, pages="all")
        texts = []
        tables_df = []
        for table in tables:
            tables_df.append(table.df)
            texts.append(table.df.to_string().replace("\\n", ""))
        return texts, tables_df


class text2vector:
    def __init__(self):
        # self.model = SentenceTransformer('GanymedeNil/text2vec-base-chinese')
        self.model = SentenceTransformer('shibing624/text2vec-base-chinese')
        # self.model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

    def __call__(self, text):
        return self.model.encode(text)


class cosine_sim:
    def __init__(self):
        pass

    def __call__(self, vector_from_table, vector_from_keyword):
        return cos_sim(vector_from_table, vector_from_keyword)

def UI():
    st.title('BDS HW3a')
    st.header('Document Intelligence')
    st.subheader('R11944008 許智凱')

    st.subheader('Upload PDF here')
    uploaded_file = st.file_uploader('Upload PDF which contains multiple tables', type=['pdf'])
    if not uploaded_file:
        return

    uploaded_file_path = os.path.join('docs', uploaded_file.name)
    with open(uploaded_file_path, 'wb') as f:
        f.write(uploaded_file.read())

    st.subheader('Enter a Keyword')
    keyword = st.text_input('Enter what you interested')

    if st.button("GO"):
        table = main(keyword, uploaded_file_path)
        st.write('Result')
        st.write(table)


def main(keyword, pdf_file):
    pdf_parser = pdf2text()
    encoder = text2vector()
    cosine_similarity = cosine_sim()

    table_texts, tables_df = pdf_parser(pdf_file)

    text_vectors = encoder(table_texts)
    keyword_vector = encoder(keyword)

    results = []
    for i in range(len(text_vectors)):
        results.append(cosine_similarity(text_vectors[i], keyword_vector))

    return tables_df[results.index(max(results))]


if __name__ == "__main__":
    UI()
