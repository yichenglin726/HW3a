import camelot
from text2vec import SentenceModel
from sentence_transformers import util
import streamlit as st


class pdf2text:
    def __init__(self):
        pass

    def __call__(self, pdf_file):
        tables = camelot.read_pdf(pdf_file, pages="all")
        self.get_titles(pdf_file)

        tables_info = []
        for table, title in zip(tables, self.titles):
            tables_info.append({"title": title, "table": table.df, "table_string": table.df.to_string().replace("\\n", "")})
        
        return tables_info
    
    def get_titles(self, pdf_file):
        tables = camelot.read_pdf(pdf_file, flavor='stream', pages="all")
        texts = []
        for table in tables:
            texts.append(table.df.to_string().replace("\\n", "").strip().split(' '))

        self.titles = []
        for text in texts:
            for word in text:
                if "ai_tables" in word:
                    title = word + " " + text[text.index(word) + 1]
                    title = title.replace("ai_tables_", "")
                    self.titles.append(title)
        

class text2vector:
    def __init__(self):
        self.model = SentenceModel('shibing624/text2vec-base-chinese')

    def __call__(self, text):
        vector = self.model.encode(text)
        return vector


class cosine_sim:
    def __init__(self):
        pass

    def __call__(self, vector_from_table, vector_from_keyword):
        return util.pytorch_cos_sim(vector_from_table, vector_from_keyword)


def main(keyword, pdf_file):
    pdf_parser = pdf2text()
    tables_info = pdf_parser(pdf_file)
    text2vector_f = text2vector()

    for table_info in tables_info:
        table_string = table_info["table_string"]
        table_vector = text2vector_f(table_string)
        keyword_vector = text2vector_f(keyword)
        similarity = cosine_sim()(table_vector, keyword_vector)
        table_info["similarity"] = similarity
    
    # find the most similar table by table_info["similarity"]
    max_similarity = 0
    max_similarity_table = None
    max_similarity_title = None
    for table_info in tables_info:
        if table_info["similarity"] > max_similarity:
            max_similarity = table_info["similarity"]
            max_similarity_table = table_info["table"]
            max_similarity_title = table_info["title"]
    
    return max_similarity_table, max_similarity_title



if __name__ == "__main__":
    st.set_page_config(
        page_title="AI Table Search Ingine",
        page_icon="random",
        layout="centered",
        initial_sidebar_state="expanded",
        )
    
    st.title("AI Table Search Ingine")
    st.write("💡PDF 1: 監督式學習、非監督式學習、強化學習")
    st.write("💡PDF 2: 動物細胞和植物細胞、多細胞生物和單細胞生物、多細胞生物細胞膜和植物細胞膜")
    st.subheader("Select a PDF file")
    pdf_file = st.selectbox("PDF file", ["docs/1.pdf", "docs/2.pdf"])
    st.subheader("Input a keyword")
    keyword = st.text_input("Keyword")
    submit_button = st.button("Search")

    if submit_button or (keyword and pdf_file):
        with st.spinner('Just a moment! The magic is happening...✨✨✨'):
            max_similarity_table, max_similarity_title = main(keyword, pdf_file)
            # show the most similar table on streamlit
            st.subheader(max_similarity_title)
            st.write(max_similarity_table)
