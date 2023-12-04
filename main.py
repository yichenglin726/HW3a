import camelot
from text2vec import SentenceModel, cos_sim, semantic_search # https://pypi.org/project/text2vec/
import streamlit as st

class pdf2text:
    def __init__(self):
        pass

    def __call__(self, pdf_file):
        with open("docs/tmp.pdf","wb") as f:
            f.write(pdf_file.getvalue())
        tables = camelot.read_pdf("docs/tmp.pdf", pages="all")
        print(tables)
        texts = []
        dfs = []
        for table in tables:
            print(table)
            text = table.df.to_string().replace("\\n", "")
            texts.append(text)
            dfs.append(table.df)
        # text = "\n".join(texts)
        # text = text.replace("\\n", "")
        return texts,dfs


class text2vector:
    def __init__(self):
        self.embedder = SentenceModel()

    def __call__(self, text):
        return self.embedder.encode(text)


class search:
    def __init__(self):
        pass

    def __call__(self, vector_from_table, vector_from_keyword):
        hits = semantic_search(vector_from_keyword, vector_from_table, top_k=1)
        hits = hits[0]
        return hits[0]['corpus_id']

def main(keyword, pdf_file):
    pdf_parser = pdf2text()
    table_text,tables = pdf_parser(pdf_file)
    t2v = text2vector()
    search_engine = search()
    result_id = search_engine(t2v(table_text),t2v(keyword))
    print(table_text[result_id])
    return tables[result_id]


if __name__ == "__main__":
    # main("keyword", "docs/1.pdf")
    # main("keyword", "docs/2.pdf")
    st.title("BDS HW3a R11922072 江宗翰")
    f = st.file_uploader("選擇僅含有表格們的pdf", type=".pdf")
    q = st.text_input("要查詢的關鍵字")
    if(st.button('點我開始搜尋')):
        if f is None:
            st.write("請選擇一個PDF")
        elif (q == ""):
            st.write("請輸入要查詢的關鍵字")
        else:
            with st.spinner("Searching"):
                st.write(main(q,f))