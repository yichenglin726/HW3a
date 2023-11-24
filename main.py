import camelot
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class pdf2text:
    def __init__(self):
        pass

    def __call__(self, pdf_file):
        tables = camelot.read_pdf(pdf_file, pages="all")

        table_data = []

        for table in tables:
            table.df.replace('\n', '', regex=True, inplace=True)
            table_data.append(
                {
                    "df": table.df,
                    "string": table.df.to_string(),
                }
            )

        return table_data


class text2vector:
    def __init__(self):
        pass

    def __call__(self, text):
        pass


class cosine_sim:
    def __init__(self):
        pass

    def __call__(self, vector_from_table, vector_from_keyword):
        pass


def search_best_table(keyword, pdf_files):
    pdf_parser = pdf2text()

    table_data = []
    for pdf_file in pdf_files:
        table_data.extend(pdf_parser(pdf_file))

    sentences = [entry["string"] for entry in table_data]
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(sentences + [keyword])
    similarity_scores = cosine_similarity(vectors[-1], vectors[:-1])[0]
    best_index = similarity_scores.argmax()

    return table_data[best_index]


def main():
    st.set_page_config(page_title="PDF Table Search Engine", page_icon="ℹ️")
    st.title("PDF Table Search Engine")

    pdf_info = {
        "🤖 1.pdf": "監督式學習、非監督式學習、強化學習相關的資料表格",
        "🔬 2.pdf": "動植物細胞特點、多細胞生物和單細胞生物相關的資料表格",
    }

    for pdf, intro in pdf_info.items():
        st.write(f"{pdf}: {intro}")

    pdf_files = ["docs/1.pdf", "docs/2.pdf"]
    selected_pdf = st.multiselect("Select PDF files", pdf_files)

    keyword = st.text_input("Enter a keyword")

    if selected_pdf and keyword:
        with st.spinner("🔎 Searching.."):
            table = search_best_table(keyword, selected_pdf)
        st.header("Search Result")
        st.table(table["df"])


if __name__ == "__main__":
    main()
