import camelot
from sentence_transformers import SentenceTransformer
import numpy as np
import jieba
import tkinter as tk
from tkinter import filedialog

class cosine_sim:
    def __init__(self):
        pass

    def __call__(self, vector_from_table, vector_from_keyword):
        vec1 = np.array(vector_from_table)
        vec2 = np.array(vector_from_keyword)

        cos_sim = vec1.dot(vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        return cos_sim

check = False
tables = None

def main():

    def search(keyword):
        global tables
        model = SentenceTransformer('sentence-transformers/LaBSE')
        cos_s = cosine_sim()

        # 精確模式分詞
        seg_list = jieba.cut(keyword, cut_all=False)
        emb_key = model.encode(list(seg_list))
        # l = len(emb_key)
        tables_output = [0] * len(tables)
        for p, table in enumerate(tables):
            for i in range(table.df.shape[0]):
                for j in range(table.df.shape[1]):
                    emb = model.encode(table.df.iat[i, j])
                    for k in emb_key:
                        tmp = cos_s(emb, k)
                        if tmp > 0.6:
                            tables_output[p] += tmp
                    
        # output_text.insert(tk.END, f"{tables_output}\n")
        # print(tables_output)
        # 找分數最高的表格
        maxidx = 0
        maxval = 0
        for i, output in enumerate(tables_output):
            if output > maxval:
                maxval = output
                maxidx = i

        label.config(text=f"搜尋完成！")
        output_text.insert(tk.END, f"{tables[maxidx].df}\n")
        # print(tables[maxidx].df)

    def on_button_click():
        global check  # 使用全局变量 check
        input_text = entry.get()
        if input_text:
            if check:
                output_text.delete(1.0, tk.END)  # 清空输出区块
                # label.config(text=f"正在搜尋與<{input_text}>相關的表格...")
                search(input_text)
            else:
                label.config(text="請先選擇PDF文件！")
        else:
            label.config(text="請輸入keyword！")

    def open_pdf():
        global check  # 使用全局变量 check
        global tables
        file_path = filedialog.askopenfilename(title="選擇PDF文件", filetypes=[("PDF Files", "*.pdf")])

        if file_path:
            tables = camelot.read_pdf(file_path, pages="all")
            label.config(text=f"已選擇PDF文件：{file_path}")
            check = True

    # 创建主窗口
    root = tk.Tk()
    root.title("BDS HW3 R11922048")

    # 设置主窗口宽度和高度
    root.geometry("640x480")  # 设置为更大的宽度和高度

    # 创建一个标签
    label = tk.Label(root, text="歡迎使用GUI介面", font=("Arial", 16))  # 指定字体和大小
    label.pack(pady=10)

    # 创建一个输入字段
    default_text = "keyword"
    entry = tk.Entry(root, font=("Arial", 16))  # 指定字体和大小
    entry.insert(0, default_text)
    entry.bind("<FocusIn>", lambda event: entry.delete(0, tk.END))
    entry.bind("<FocusOut>", lambda event: entry.insert(0, default_text) if entry.get() == "" else None)
    entry.pack(pady=10)

    # 创建两个按钮
    button1 = tk.Button(root, text="選擇PDF文件", command=open_pdf, font=("Arial", 16))
    button1.pack()

    button2 = tk.Button(root, text="搜尋", command=on_button_click, font=("Arial", 16))
    button2.pack(pady=5)

    # 提示文本
    hint_label = tk.Label(root, text="最相關的表格", font=("Arial", 16), fg="red")
    hint_label.pack(pady=5)

    # 创建一个输出区块
    output_text = tk.Text(root, height=12, width=60, font=("Arial", 16))
    output_text.pack()

    # 开始事件循环
    root.mainloop()


if __name__ == "__main__":
    main()
    # main("keyword", "docs/2.pdf")
