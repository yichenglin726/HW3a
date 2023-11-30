import camelot
from text2vec import SentenceModel
from numpy import dot
from numpy.linalg import norm
import tkinter as tk
import sys
from pandastable import Table

class pdf2text:
    def __init__(self):
        pass

    def __call__(self, pdf_file):
        tables = camelot.read_pdf(pdf_file, pages="all")
        texts = []
        table_dfs = []
        for table in tables:
            # print(table.df)
            table_dfs.append(table.df)
            text = ''
            shape = table.df.shape
            for i in range(0, shape[0]):
                for j in range(0, shape[1]):
                    table.df.iloc[i, j] = table.df.iloc[i, j].replace("\n", "")
            for i in range(1, shape[0]):
                for j in range(1, shape[1]):
                    text += f"{table.df.iloc[i, 0]}，{table.df.iloc[0, j]}： {table.df.iloc[i, j]}。"
                text += "\n"
            texts.append(text)
        # text = "\n".join(texts)
        # text = text.replace("\\n", "")
        return texts, table_dfs


class text2vector:
    def __init__(self):
        self.model = SentenceModel('shibing624/text2vec-base-chinese')

    def __call__(self, text):
        return self.model.encode(text)


class cosine_sim:
    def __init__(self):
        pass

    def __call__(self, vector_from_table, vector_from_keyword):
        return dot(vector_from_table, vector_from_keyword) / (norm(vector_from_table) * norm(vector_from_keyword))


def main(keyword, pdf_file):
    pdf_parser = pdf2text()
    encoder = text2vector()
    cos_sim_fn = cosine_sim()
    texts, tables = pdf_parser(pdf_file)
    vectors = encoder(texts)

    keyword_vector = encoder(keyword)
    max_idx = -1
    max_cos_sim = -1
    for i, vector in enumerate(vectors):
        sim = cos_sim_fn(vector, keyword_vector)
        if sim > max_cos_sim:
            max_cos_sim = sim
            max_idx = i
    # print(tables[max_idx])
    return tables[max_idx]

class TestApp(tk.Frame):
    """Basic test frame for the table"""
    def __init__(self, parent, df):
        self.parent = parent
        tk.Frame.__init__(self)
        self.main = self.master
        self.main.title('Table app')
        f = tk.Frame(self.main)
        f.grid(row=4)
        self.table = pt = Table(f, dataframe=df,
                                showtoolbar=True, showstatusbar=True)
        pt.show()
        return

if __name__ == "__main__":
    root=tk.Tk()
    
    root.geometry("600x400")
    keyword_var = tk.StringVar()

    def submit():
        keyword = keyword_var.get()
        table = main(keyword, sys.argv[1])
        print(table.to_string())
        TestApp(root, table)
        
    name_label = tk.Label(root, text='Keyword')
    name_entry = tk.Entry(root, textvariable=keyword_var)
    sub_btn=tk.Button(root, text='Query', command=submit)

    name_label.grid(row=0,column=0)
    name_entry.grid(row=0,column=1)
    sub_btn.grid(row=2,column=1)
    root.mainloop()
