import tkinter as tk
from tkinter import filedialog, messagebox, ttk, scrolledtext
import pandas as pd
import camelot
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class Pdf2Text:
    def __call__(self, pdf_file):
        try:
            tables = camelot.read_pdf(pdf_file, pages='all')
            if tables.n == 0:
                raise ValueError("PDF檔中找不到表格。")
            texts = [table.df.to_string(index=False) for table in tables]
            return texts  # 返回包含所有表格文本的列表
        except Exception as e:
            raise e

class Text2Vector:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()

    def fit_transform(self, texts):
        return self.vectorizer.fit_transform(texts)  # 需要一個字符串列表

    def transform(self, text):
        return self.vectorizer.transform([text])  # 單個字符串需要放在列表中

class CosineSim:
    def __call__(self, vectors_from_tables, vector_from_keyword):
        return cosine_similarity(vectors_from_tables, vector_from_keyword)


class PdfSearchApp:
    def __init__(self, root):
        self.root = root
        self.root.title("PDF搜尋工具")
        self.create_widgets()
        self.pdf_file_path = None
        self.keyword = None
        self.search_results = None  # To store the search results

    def create_widgets(self):
        self.upload_button = tk.Button(self.root, text="上傳PDF文件", command=self.upload_file)
        self.upload_button.grid(row=0, column=0, padx=10, pady=10)

        self.keyword_label = tk.Label(self.root, text="輸入關鍵字:")
        self.keyword_label.grid(row=1, column=0, padx=10, pady=10)

        self.keyword_entry = tk.Entry(self.root)
        self.keyword_entry.grid(row=1, column=1, padx=10, pady=10)

        self.search_button = tk.Button(self.root, text="搜尋", command=self.search_in_pdf)
        self.search_button.grid(row=2, column=0, columnspan=2, padx=10, pady=10)

        self.tree = ttk.Treeview(self.root)
        self.tree.grid(row=3, column=0, columnspan=2, padx=10, pady=10)
        
        self.save_button = tk.Button(self.root, text="儲存結果", command=self.save_results)
        self.save_button.grid(row=4, column=0, columnspan=2, padx=10, pady=10)

        self.result_text = scrolledtext.ScrolledText(self.root, height=4, width=50)
        self.result_text.grid(row=5, column=0, columnspan=2, padx=10, pady=10)

    def upload_file(self):
        self.pdf_file_path = filedialog.askopenfilename(
            filetypes=[("PDF files", "*.pdf")]
        )
        if self.pdf_file_path:
            messagebox.showinfo("文件上傳", f"成功上傳文件：{self.pdf_file_path}")

    def search_in_pdf(self):
        self.keyword = self.keyword_entry.get()
        if not self.pdf_file_path or not self.keyword:
            messagebox.showwarning("警告", "請先上傳文件並輸入關鍵字。")
            return

        pdf_parser = Pdf2Text()
        text_vectorizer = Text2Vector()
        cosine_similarity_calc = CosineSim()

        try:
            tables = camelot.read_pdf(self.pdf_file_path, pages='all')
            texts = pdf_parser(self.pdf_file_path)
            vectors_from_tables = text_vectorizer.fit_transform(texts)
            vector_from_keyword = text_vectorizer.transform(self.keyword)
            similarity = cosine_similarity_calc(vectors_from_tables, vector_from_keyword)

            best_match_index = similarity.argmax()
            best_similarity_score = similarity[best_match_index]
            best_table_df = tables[best_match_index].df

            # Display the table in the Treeview widget
            self.display_table(best_table_df)

            # Save the search results for potential saving
            self.search_results = best_table_df
            
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, f"最佳匹配表格餘弦相似度：{best_similarity_score}\n\n")
        except ValueError as ve:
            messagebox.showinfo("結果", str(ve))
        except Exception as e:
            messagebox.showerror("錯誤", f"搜索過程中出錯：{e}")
            self.result_text.insert(tk.END, "搜索失敗。\n")

    def display_table(self, table):
        for i in self.tree.get_children():
            self.tree.delete(i)
        self.tree["column"] = list(table.columns)
        self.tree["show"] = "headings"
        for column in self.tree["column"]:
            self.tree.heading(column, text=column)

        for row in table.to_numpy().tolist():
            self.tree.insert("", "end", values=row)

    def save_results(self):
        with filedialog.asksaveasfile(mode='w', defaultextension=".csv") as file:
            if file:
                pd.DataFrame(self.search_results).to_csv(file, index=False)
                messagebox.showinfo("儲存成功", "表格結果已儲存。")


if __name__ == "__main__":
    root = tk.Tk()
    app = PdfSearchApp(root)
    root.mainloop()