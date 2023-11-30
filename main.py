import camelot
import re    
from Levenshtein import distance, hamming, median
import sys
import streamlit as st

class pdf2text:
    def __init__(self):
        pass

    def __call__(self, pdf_file):
        tables = camelot.read_pdf(pdf_file, pages="all")
        texts = []
        for table in tables:
            texts.append(table.df.to_string())
        text = "\n".join(texts)
        text = text.replace("\\n", "")
        return text


class text2vector:
    def __init__(self):
        pass

    def __call__(self, text):
        pass


class cosine_sim:
    def __init__(self):
        pass

    def __call__(self, vector_from_table, vector_from_keyword):
        pass
# data processing for input
def table_query (table_text):
    rows = table_text.split('\n')
    init = True
    first_row = True
    row_topics = []
    column_topics = []
    tables_text = []
    tables_text.append('')
    row_topics.append([])
    column_topics.append([])
    table = 0
    word_min_len = [10000]
    for row in rows:
        if init == True:
            tables_text[table] += (row + '  \n')
            init = False
        else:
            columns = re.split(r'\s{2,}', row)
            init2 = 0 
            for column in columns:
                column = column.replace(' ', '')
                if init2 == 0:
                    init2 = 1
                    if column == '':
                        init2 = 0
                        first_row = True
                        row_topics.append([])
                        column_topics.append([])
                        tables_text.append('')
                        word_min_len.append(10000)
                        table += 1
                        break
                    tables_text[table] += (row + '  \n')
                else:
                    if first_row == True:
                        row_topics[table].append(column)
                        if len(column) < word_min_len[table] and len(column) != 0:
                            word_min_len[table] = len(column)
                    elif init2 == 1:
                        column_topics[table].append(column)
                        if len(column) < word_min_len[table] and len(column) != 0:
                            word_min_len[table] = len(column)
                        init2 = 2
            if first_row == True and init2 != 0:
                first_row = False
    return row_topics, column_topics, word_min_len, tables_text

def word_in_query(query, text, word_min_len):
    for i in range(len(query) - (word_min_len-1)):
        q_word = query[i:i+word_min_len]
        for j in range(len(text) - (word_min_len-1)):
            t_word = text[j:j+word_min_len]
            if q_word == t_word:
                return True
    return False

def threshold(word_min_len, query):
    threshold = word_min_len
    threshold -= len(query)
    return abs(threshold)

def find_table(row_topics, column_topics, word_min_len, query):
    
    table_count = len(row_topics)
    query = query.strip()
    table_index = []
    for i in range(table_count):
        # find word in row
        min = 10000
        min_word = 0
        for j in range(len(row_topics[i])):
            word = row_topics[i][j]
            d = distance(query, word)
            # match
            query2 = query
            if d <= min and word_in_query(query, word, word_min_len[i]):
                min_word = word
                min = d
        if min < threshold(word_min_len[i], query):
            min = 10000
            # find word in column
            for ch in min_word:
                query2 = query2.replace(ch, '', 1)
            for k in range(len(column_topics[i])):
                word = column_topics[i][k]
                d = distance(query2, word)
                if d <= min and word_in_query(query2, word, word_min_len[i]):
                    min_word = word
                    min = d
            if min < threshold(word_min_len[i], query):
                table_index.append(i)
    return table_index

def main(keyword, pdf_file):
    pdf_parser = pdf2text()
    table_text = pdf_parser(pdf_file)

    row_topics, column_topics, word_min_len, tables = table_query(table_text)
    table_index = find_table(row_topics, column_topics, word_min_len, keyword)
    
    for i in table_index:
        st.write(tables[i])
    
    st.write('End!')
    # return table


if __name__ == "__main__":
    st.write('Input the PDF file name (1.pdf or 2.pdf) and keyword')
    pdf = st.text_input('PDF', '1.pdf')
    keyword = st.text_input('keyword', '非監督式學習的應用')
    st.write('serach ', keyword, ' in ', pdf)
    pdf = 'docs/' + pdf
    main(keyword, pdf)
    # main("keyword", "docs/2.pdf")
