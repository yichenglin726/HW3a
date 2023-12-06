import camelot
import streamlit as st
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
import os

class pdf2text:
    def __init__(self):
        pass

    def __call__(self, pdf_file):
        tablesStream = camelot.read_pdf(pdf_file, flavor='stream', pages="all")
        texts = [table.df.to_string().replace("\\n", "").strip().split(' ') for table in tablesStream]
        titles = []
        for text in texts:
            titles = ["{w} {t}".format(w = word, t = text[text.index(word) + 1]) for word in text if "ai_tables" in word]

        tables = camelot.read_pdf(pdf_file, pages="all")
        for table in tables:
            table.df.replace('\n', '', regex=True)
        tableTexts = [{"title": title, "df": table.df, "str": table.df.to_string()} for table, title in zip(tables, titles)]
        return tableTexts


class text2vector:
    def __init__(self):
        self.model = SentenceTransformer('distiluse-base-multilingual-cased-v2')

    def __call__(self, text):
        return self.model.encode(text)


class cosine_sim:
    def __init__(self):
        pass

    def __call__(self, vector_from_table, vector_from_keyword):
        return 1 - cosine(vector_from_table, vector_from_keyword)


def main(keyword, pdf_file):
    pdf_parser = pdf2text()
    table_text = pdf_parser(pdf_file)
    transform = text2vector()
    vectors = [transform(table["str"]) for table in table_text]

    calSimilarity = cosine_sim()
    similaritys = []
    maxValue = 0
    maxIndex = 0
    for idx, vector in enumerate(vectors):
        vectorKeyword = transform(keyword)
        similarity = calSimilarity(vector, vectorKeyword)
        similaritys.append(similarity)
        if similarity > maxValue:
            maxValue = similarity
            maxIndex = idx
    resultTableTitle = table_text[maxIndex]["title"]
    resultTable = table_text[maxIndex]["df"]
    return resultTableTitle, resultTable


if __name__ == "__main__":
    folder_path = 'docs'
    filenames = os.listdir('./' + folder_path)
    fileList = []
    for filename in filenames:
        fileList.append(folder_path + '/' + filename)

    st.title("PDF analyzer")
    st.subheader("Select a PDF file: ")
    pdf_file = st.selectbox("PDF file", fileList)
    keyword = st.text_input('Enter keyword')

    if (pdf_file and keyword):
        with st.spinner('Analysing....'):
            title, table_text = main(keyword, pdf_file)
            st.subheader(title)
            st.dataframe(table_text)
