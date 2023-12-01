import camelot
import os
from difflib import SequenceMatcher


class pdf2text:
    def __init__(self):
        pass

    def __call__(self, pdf_file):
        tables = camelot.read_pdf(pdf_file, pages="all")
        texts = []
        for table in tables:
            data = table.df
            for i in range (data.shape[1]):
                for j in range (data.shape[0]):
                    data[i][j] = data[i][j].replace("\n", "").replace(" ","")
            texts.append(data)
        return texts


class text2vector:
    def __init__(self):
        pass

    def __call__(self, text):
        pass
        
class cosine_sim:
    def __init__(self):
        pass

    def __call__(self, vector_from_table, vector_from_keyword):
        return SequenceMatcher(None, vector_from_table, vector_from_keyword).find_longest_match()
        

def find_table(keyword, pdf_file):
    pdf_parser = pdf2text()
    table = pdf_parser(pdf_file)
    
    cos = cosine_sim()
    sim, id = [-1,-1], -1
    for i in range (len(table)):
        s = [-1,-1]
        tempword = ""
        for j in range (table[i].shape[0]):
            word = table[i][0][j]
            for k in range (len(keyword)):
                if (s[0] <= cos(word, keyword).size):
                    tempword = keyword[cos(word, keyword).b : cos(word, keyword).b+cos(word, keyword).size]
                s[0] = max(s[0], cos(word, keyword).size)
        leftword = keyword.replace(tempword, "")
        for j in range (table[i].shape[1]):
            word = table[i][j][0]
            for k in range (len(leftword)):
                s[1] = max(s[1], cos(word, leftword).size)
        if (s[0]+s[1] >= sim[0]+sim[1]):
            sim, id = s, i
    return table[id]
    # return table


if __name__ == "__main__":
    while (True):
        filename = input("\n請輸入檔案路徑(如，docs/1.pdf)： ")
        if (os.path.exists(filename) == False):
            print("檔案不存在")
            continue
        if (".pdf" not in filename):
            print("請輸入 .pdf 檔")
            continue
        keyword = input("\n請輸入要搜尋的關鍵字：")
        table = find_table(keyword, filename)
        table.to_csv('output/'+keyword+'.csv', index=False, header=False)
        
        ans = 0
        while (ans != "n" and ans != "y"):
            ans = input("\n對應的表格已輸出至 output 資料夾。是否繼續？ (y/n) ")
        if (ans == "n"):
            break
        print("\n\n\n\n\n")
