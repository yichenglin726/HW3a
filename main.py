import camelot
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import tempfile


class pdf2text:
    def __init__(self):
        pass

    def __call__(self, pdf_file):
        tables = camelot.read_pdf(pdf_file, pages='all')

        texts = []
        for table in tables:
            data = table.data
            data = [' '.join(row) for row in data]
            text = ' '.join(data)
            text = text.replace('\n', '')
            texts.append(text)
        return texts

    def get_table(self, pdf_file, index):
        tables = camelot.read_pdf(pdf_file, pages='all')
        return tables[index].df


class text2vector:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()

    def __call__(self, texts):
        return self.vectorizer.fit_transform(texts)


class cosine_sim:
    def __init__(self):
        pass

    def __call__(self, table_vector, keyword_vector):
        return cosine_similarity(table_vector, keyword_vector)


def main(keyword, pdf_file):
    pdf_parser = pdf2text()
    text_vectorizer = text2vector()
    sim_getter = cosine_sim()

    texts = pdf_parser(pdf_file)
    vectors = text_vectorizer(texts + [keyword])

    best_index = 0
    similarity = sim_getter(vectors[0], vectors[-1])
    for i in range(1, vectors.shape[0] - 1):
        cand = sim_getter(vectors[i], vectors[-1])
        if cand > similarity:
            best_index = i
            similarity = cand

    return pdf_parser.get_table(pdf_file, best_index)


if __name__ == '__main__':
    st.sidebar.title('HW3-Stage-A Document Intelligence')

    upload_file = st.sidebar.file_uploader('Plz upload your pdf', type='pdf')
    keyword = st.sidebar.text_input('Plz input your keyword', key='keyword')

    if upload_file is not None and keyword != '':
        pdf_file = tempfile.NamedTemporaryFile(dir='.', suffix='.pdf')
        pdf_file.file.write(upload_file.read())
        pdf_file.file.flush()

        table = main(keyword, pdf_file.file.name)
        st.table(table)

        pdf_file.close()
