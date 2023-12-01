import pdfplumber
import camelot

# Extract text and coordinates using pdfplumber
# text_with_coordinates = []
# with pdfplumber.open('docs/1.pdf') as pdf:
#     for page_num in range(len(pdf.pages)):
#         page = pdf.pages[page_num]
#         text_instances = page.extract_text(x_tolerance=4, y_tolerance=2)
#         for text in text_instances:
#             print(text)
#             # text_with_coordinates.append({
#             #     'text': text['text'],
#             #     'x0': text['x0'],
#             #     'x1': text['x1'],
#             #     'top': text['top'],
#             #     'bottom': text['bottom']
#             # })

# Extract tables and their coordinates using Camelot
tables = list(camelot.read_pdf('docs/2.pdf', pages='all'))
merged_tables = []
i = 1
while (i < len(tables)):
    if (tables[i].rows[0][0] > 760):
        merged_tables = merged_tables + tables[i-1].df.values.tolist() + tables[i].df.values.tolist()
        i += 2
    else:
        merged_tables = merged_tables + tables[i-1].df.values.tolist()
        i+=1

# merged_tables = [[[col.replace('\\n', '') for col in row] for row in table] for table in merged_tables]
print(merged_tables)

# Match text and tables by coordinates
# for table in tables:
#     table_bbox = table.df.bbox  # Accessing the table bounding box information
#     # Match text with tables based on coordinates
#     for text in text_with_coordinates:
#         # You can define your logic here to associate text with tables based on coordinates
#         if (text['top'] < table_bbox.bottom and text['bottom'] > table_bbox.top) or \
#            (text['bottom'] > table_bbox.top and text['top'] < table_bbox.bottom):
#             # Match found, process or store the text and table data as needed
#             print("Text:", text['text'])
#             print("Table Data:", table.df)
