import camelot
import streamlit as st
import tempfile
from text2vec import SentenceModel
from sentence_transformers import util


class pdf2text:
    def __init__(self):
        pass

    def __call__(self, pdf_file):
        n_tables = camelot.read_pdf(pdf_file, pages="all")
        tables = camelot.read_pdf(pdf_file, flavor='stream', pages="all")
        all_table_texts = []
        for table in tables:
            table_texts = table.df.to_string()
            table_texts_clean = table_texts.replace("\\n", "").strip().split(' ')
            all_table_texts.append(table_texts_clean)

        self.titles = []
        for table_texts in all_table_texts:
            for i in range(len(table_texts)):
                text = table_texts[i]
                if "ai_tables" in text:
                    title = text + table_texts[i+1]
                    self.titles.append(title)

        tables_data = []
        for table, title in zip(n_tables, self.titles):
            tables_data.append({"title": title, "table_df": table.df, "table_df_toString": table.df.to_string().replace("\\n", "")})
        
        return tables_data
    


class text2vector:
    def __init__(self):
        self.model = SentenceModel('shibing624/text2vec-base-chinese')
    def __call__(self, text):
        return self.model.encode(text)


class cosine_sim:
    def __init__(self):
        pass

    def __call__(self, vector_from_table, vector_from_keyword):
        return util.pytorch_cos_sim(vector_from_table, vector_from_keyword)


def main(keyword, pdf_file):
    pdf_parser = pdf2text()
    table_text = pdf_parser(pdf_file)
    Text2Vector = text2vector()
    Cosine_Sim = cosine_sim()

    for table_data in table_text:
        table_df_string = table_data["table_df_toString"]
        table_df_string_vector = Text2Vector(table_df_string)
        keyword_vector = Text2Vector(keyword)
        similarity = Cosine_Sim(table_df_string_vector, keyword_vector)
        table_data["similarity"] = similarity
    
    Max = -1
    max_title = None
    max_table = None
    for table_data in table_text:
        if table_data["similarity"] > Max:
            Max = table_data["similarity"]
            max_title = table_data["title"]
            max_table = table_data["table_df"]
    
    return max_title, max_table



if __name__ == '__main__':
    st.set_page_config(
        page_title="HW3",
        layout="centered"
        )
    st.title("R12946003 HW3")
    st.subheader("Choose a PDF file")
    pdf_file = st.file_uploader('', type='pdf')
    st.subheader("Input the searching keywords")
    keyword = st.text_input("")
    button = st.button("OK")

    if button :
        with st.spinner('Running...'):
            new_file = tempfile.NamedTemporaryFile(dir='.', suffix='.pdf')
            new_file.file.write(pdf_file.read())
            new_file.file.flush()

            title, table = main(keyword, new_file.file.name)
            st.subheader(title)
            st.table(table)

            new_file.close()