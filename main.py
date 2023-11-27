import camelot
from text2vec import SentenceModel
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st


class pdf2text:
    def __init__(self):
        pass

    def __call__(self, pdf_file):
        tables = camelot.read_pdf(pdf_file, pages="all", flavor="lattice")
        texts = []
        tables_df = []
        for table in tables:
            texts.append(table.df.to_string().replace("\\n", ""))
            tables_df.append(table.df)

        tables = camelot.read_pdf(pdf_file, pages="all", flavor="stream")
        titles = []
        for table in tables:
            raw_table = table.df.to_string()
            title_index = [
                idx
                for idx in range(len(raw_table))
                if raw_table.find("ai_tables_#", idx) == idx
            ]
            for idx in title_index:
                titles.append(raw_table[idx:].split("\n")[0].replace("  ", ""))

        return texts, titles, tables_df


class text2vector:
    def __init__(self):
        self.model = SentenceModel("shibing624/text2vec-base-chinese")

    def __call__(self, text):
        embeddings = self.model.encode(text)

        return embeddings


class cosine_sim:
    def __init__(self):
        pass

    def __call__(self, vector_from_table, vector_from_keyword):
        return cosine_similarity(vector_from_table, vector_from_keyword)


def main(keyword, pdf_file):
    pdf_parser = pdf2text()
    text_parser = text2vector()
    similarity = cosine_sim()

    keyword_vector = [text_parser(keyword)]

    table_text, table_title, tables_df = pdf_parser(pdf_file)
    sim_result = []
    for table in table_text:
        table_vector = []
        for text in table.split(" "):
            if text:
                table_vector.append(text_parser(text))
        sim_result.append(
            sorted(similarity(table_vector, keyword_vector).squeeze(), reverse=True)
        )

    sim_title = "\n".join([table_title[sim_result.index(max(sim_result))]])
    sim_text = tables_df[sim_result.index(max(sim_result))]

    return sim_text, sim_title


if __name__ == "__main__":
    st.set_page_config(
        page_title="Document Intelligence",
        page_icon="random",
        layout="centered",
        initial_sidebar_state="auto",
    )
    st.header("BDS HW3A - Document Intelligence")
    st.write("Name: 高榮浩")
    st.write("ID: R12922127")
    st.subheader("Select a PDF File")
    pdf = st.selectbox("PDF File", ["監督式、非監督式與強化學習", "動物和植物細胞、多細胞和單細胞生物、多細胞生物和植物細胞膜"])
    st.subheader("Input a Searching Keyword")
    keyword = st.text_input("Searching Keyword", value="非監督式學習的應用")
    clicked = st.button("Submit")

    if clicked:
        with st.spinner("Wait for it..."):
            if pdf == "監督式、非監督式與強化學習":
                text, title = main(keyword, "docs/1.pdf")
            elif pdf == "動物和植物細胞、多細胞和單細胞生物、多細胞生物和植物細胞膜":
                text, title = main(keyword, "docs/2.pdf")

            st.subheader(title)
            st.write(text)
