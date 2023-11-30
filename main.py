import streamlit as st
# import camelot
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import threading
import pandas as pd
import pdfplumber
import os

class pdf2table:
    def __init__(self):
        pass

    def __call__(self, pdf_files):
        tables = []
        docs = "docs/"
        for pdf in pdf_files:
            text_all_table = self.extract_tables(docs + pdf.name)
            prefix = pdf.name.split(".pdf")[0]
            for page_num, page in enumerate(text_all_table):
                for table_num, table in enumerate(page):
                    print("處理表格頁面{0}/表格{1}".format(page_num, table_num))
                    if table:
                        table = [row for row in table if not all(cell == '' for cell in row)]
                        table_df = pd.DataFrame(table[1:], columns=table[0])
                        tables.append(table_df)
                        final_filename = docs + "{0}_page{1}_table{2}.csv".format(prefix, page_num, table_num)
                        if not os.path.exists(final_filename):
                            table_df.to_csv(final_filename, index=False, encoding="utf_8_sig")
                            print("生成文件：", final_filename)
        return tables
    
    # page_chars最尾部的非空字符
    def tail_not_space_char(self, page_chars):
        i = -1
        while page_chars[i].get('text').isspace():
            i = i - 1
            # 如果字符是空格，繼續向前尋找
        return page_chars[i]

    # 返回列表最頭部的非空字符
    def head_not_space_char(self, page_chars):
        i = 0
        while page_chars[i].get('text').isspace():
            i += 1
            # 如果字符是空格，繼續向後尋找
        return page_chars[i]


    # 將pdf表格數據抽取到文件中
    def extract_tables(self, input_file_path):
        print("========================================表格抽取開始========================================")
        # 讀取pdf文件，保存為pdf實例
        pdf = pdfplumber.open(input_file_path)
    
        # 存儲每個頁面最底部字符的y0坐標
        y0_bottom_char = []
        # 存儲每個頁面最底部表格中最底部字符的y0坐標
        y0_bottom_table = []
        # 存儲每個頁面最頂部字符的y1坐標
        y1_top_char = []
        # 存儲每個頁面最頂部表格中最頂部字符的y1坐標
        y1_top_table = []
        # 存儲所有頁面內的表格文本
        text_all_table = []
        # 訪問每一頁
        print("===========開始抽取每頁頂部和底部字符坐標及表格文本===========")
        for page in pdf.pages:
            # table對象，可以訪問其row屬性的bbox對象獲取坐標
            table_objects = page.find_tables()
            text_table_current_page = page.extract_tables()
            if text_table_current_page:
                text_all_table.append(text_table_current_page)
                # 獲取頁面最底部非空字符的y0
                y0_bottom_char.append(self.tail_not_space_char(page.chars).get('y0'))
                # 獲取頁面最底部表格中最底部字符的y0，table對象的bbox以左上角為原點，而page的char的坐標以左下角為原點，可以用page的高度減去table對象的y來統一
                y0_bottom_table.append(page.bbox[3] - table_objects[-1].bbox[3])
                # 獲取頁面最頂部字符的y1
                y1_top_char.append(self.head_not_space_char(page.chars).get('y1'))
                # 獲取頁面最頂部表格中最底部字符的y1
                y1_top_table.append(page.bbox[3] - table_objects[0].bbox[1])
        print("===========抽取每頁頂部和底部字符坐標及表格文本結束===========")

        # 處理跨頁面表格，將跨頁面表格合併，i是當前頁碼，對於連跨數頁的表，應跳過中間頁面，防止重複處理
        print("===========開始處理跨頁面表格===========")
        i = 0
        while i < len(text_all_table):
            print("處理頁面{0}/{1}".format(i+1, len(text_all_table)))
            # 判斷當前頁面是否以表格結尾且下一頁面是否以表格開頭，若都是則說明表格跨行，進行表格合併
            # j是要處理的頁碼，一般情況是當前頁的下一頁，對於連跨數頁情況，也可以是下下一頁,跨頁數為k
            # 若當前頁是最後一頁就不用進行處理
            if i + 1 >= len(text_all_table):
                break
            j = i + 1
            k = 1
            # 要處理的頁為空時退出
            while text_all_table[j]:
                if y0_bottom_table[i] <= y0_bottom_char[i] and y1_top_table[j] >= y1_top_table[j]:
                    # 當前頁面最後一個表與待處理頁面第一個表合併
                    text_all_table[i][-1] = text_all_table[i][-1] + text_all_table[j][0]
                    text_all_table[j].pop(0)
                    # 如果待處理頁面只有一個表，就要考慮下下一頁的表是否也與之相連
                    if not text_all_table[j] and j + 1 < len(text_all_table) and text_all_table[j + 1]:
                        k += 1
                        j += 1
                    else:
                        i += k
                        break
                else:
                    i += k
                    break
        print("===========處理跨頁面表格結束===========")
        return text_all_table
    
class text2vector:
    def __init__(self):
        self.model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")

    def __call__(self, text):
        return self.model.encode(text)


class cosine_sim:
    def __init__(self):
        pass

    def __call__(self, vector_from_table, vector_from_keyword):
        return cosine_similarity(vector_from_keyword, vector_from_table)

def search(kw_embed, table_embed):
    table_embed = np.vstack(table_embed)
    sim_calculator = cosine_sim()
    similarity = sim_calculator(table_embed, kw_embed)
    print(similarity)
    return np.argmax(similarity)

def get_tb_embed(tables, model):
    table_embed = []
    for table in tables:
        # print(table)
        table_embed.append(model(table.to_string()))
    return table_embed

result_container = []

def search_and_return_table(keyword, pdf_files):
    model = text2vector()
    pdf_parser = pdf2table()
    tables = pdf_parser(pdf_files)
    tb_embed = get_tb_embed(tables, model)
    kw_embed = [model(keyword)]
    best_tb = tables[search(kw_embed, tb_embed)]
    print(best_tb)
    result_container.append(best_tb)

def main():
    st.title("PDF Search App")

    keyword = st.text_input("Enter Keyword:")
    pdf_files = st.file_uploader("Please upload some pdf files", accept_multiple_files=True)  # Add more PDF files if needed
    if st.button("Search", ):

        # Run the search in a separate thread to avoid blocking the UI
        search_thread = threading.Thread(target=search_and_return_table, args=(keyword, pdf_files))
        search_thread.start()

        st.text("Searching for '{}'. Please wait...".format(keyword))

        # Wait for the search thread to finish before continuing
        search_thread.join()

    if result_container:
        st.text("Best Table:")
        st.table(result_container[0])
 
if __name__ == "__main__":
    main()
