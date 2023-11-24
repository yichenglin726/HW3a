import camelot
import numpy as np
import pandas as pd
from numpy.linalg import norm
from sentence_transformers import SentenceTransformer
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QFileDialog, QLineEdit, QTableWidget, QTableWidgetItem, QLabel
from PyQt5.QtCore import Qt
import os

class pdf2text:
    def __init__(self):
        pass

    def __call__(self, pdf_file):
        tables = camelot.read_pdf(pdf_file, pages="all")
        texts = []
        table_df = []
        for table in tables:
            texts.append(table.df.to_string())
            table_df.append(table.df)
        text = "\n".join(texts)
        text = text.replace("\\n", "")
        return text , table_df


class text2vector:
    def __init__(self):
        self.model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        

    def __call__(self, text):
        return self.model.encode(text)


class cosine_sim:
    def __init__(self):
        pass

    def __call__(self, vector_from_table, vector_from_keyword):
        return np.dot(vector_from_table, vector_from_keyword)/(norm(vector_from_table)*norm(vector_from_keyword))


def main(keyword, pdf_file):
    pdf_parser = pdf2text()
    text_parser = text2vector()
    cosine_sim_do = cosine_sim()
    embedding = []
    similarity = []
    table_text , table_df = pdf_parser(pdf_file)
    for i in table_df:
        input_text = i.to_string()
        embedding.append(text_parser(input_text))
    
    #query = input("你要找哪張表")
    query_embedding = text_parser(keyword)
    for i in embedding:
        similarity.append(cosine_sim_do(i , query_embedding))
    
    #print(table_df[np.argmax(similarity)])
    #print(np.argmax(similarity))
    return table_df[np.argmax(similarity)]

class PDFTableFinderApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.layout = QVBoxLayout(self)

        # 上傳PDF按鈕
        self.upload_btn = QPushButton('Upload PDF', self)
        self.upload_btn.clicked.connect(self.openFileNameDialog)
        self.layout.addWidget(self.upload_btn)

        # 文件名標籤
        self.filename_label = QLabel('No PDF selected', self)
        self.layout.addWidget(self.filename_label)

        # 輸入關鍵詞
        self.keyword_entry = QLineEdit(self)
        self.layout.addWidget(self.keyword_entry)

        # 查找表格按鈕
        self.find_table_btn = QPushButton('Find Table', self)
        self.find_table_btn.clicked.connect(self.find_table)
        self.layout.addWidget(self.find_table_btn)

        # 顯示表格的TableWidget
        self.table_widget = QTableWidget(self)
        self.layout.addWidget(self.table_widget)

        self.setGeometry(300, 300, 600, 400)
        self.setWindowTitle('PDF Table Finder')
        self.show()

    def openFileNameDialog(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(self, "Open PDF", "", "PDF Files (*.pdf)", options=options)
        if fileName:
            self.pdf_file = fileName
            self.filename_label.setText(f"Selected File: {os.path.basename(fileName)}")  # 更新標籤以顯示文件名

    def find_table(self):
        keyword = self.keyword_entry.text()
        if hasattr(self, 'pdf_file') and keyword:
            dataframe = main(keyword ,self.pdf_file)
            self.show_dataframe(dataframe)

    def show_dataframe(self, df):
        self.table_widget.setColumnCount(len(df.columns))
        self.table_widget.setRowCount(len(df.index))

        

        for i in range(len(df.index)):
            for j in range(len(df.columns)):
                item = QTableWidgetItem(str(df.iloc[i, j]))
                item.setTextAlignment(Qt.AlignLeft | Qt.AlignTop)
                item.setFlags(item.flags() | Qt.TextWrapAnywhere)
                self.table_widget.setItem(i, j, item)

        self.table_widget.resizeRowsToContents()  # 調整行高以適應內容
        self.table_widget.resizeColumnsToContents()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = PDFTableFinderApp()
    sys.exit(app.exec_())
    #main("keyword", "docs/1.pdf")
    #main("keyword", "docs/2.pdf")
