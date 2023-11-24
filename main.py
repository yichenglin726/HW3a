import camelot
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import jieba
import streamlit as st
import tempfile


class pdf2text:
    def __init__(self):
        pass

    def __call__(self, pdf_file):
        '''
        return list of text, a text corresponds to a table
        '''
        tables = camelot.read_pdf(pdf_file, pages='all')

        texts = []
        for table in tables:
            data = table.data

            data = map(lambda l: ' '.join(l), data)
            data = ' '.join(data)

            data = data.replace('\n', '')

            texts.append(data)
        return texts

    def get_table(self, pdf_file, index):
        '''
        return target table (pandas DataFrame)
        '''
        tables = camelot.read_pdf(pdf_file, pages='all')
        return tables[index].df


class text2vector:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(tokenizer=\
            lambda text: list(jieba.cut(text)))

    def __call__(self, texts):
        return self.vectorizer.fit_transform(texts)


class cosine_sim:
    def __init__(self):
        pass

    def __call__(self, table_vector, keyword_vector):
        return cosine_similarity(table_vector, keyword_vector)


def search_table(keyword, pdf_file):
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
    st.title('b09902121 pdf searcher')

    upload_file = st.file_uploader('choose a pdf file', type='pdf')
    if upload_file is None:
        exit()

    keyword = st.text_input('enter keyword', key='keyword')
    if keyword == '':
        exit()

    temp_file = tempfile.NamedTemporaryFile(dir='.', suffix='.pdf')
    temp_file.file.write(upload_file.read())
    temp_file.file.flush()

    table = search_table(keyword, temp_file.file.name)
    st.table(table)

    temp_file.close()
