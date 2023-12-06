import camelot
import numpy as np
from sentence_transformers import SentenceTransformer
import streamlit as st

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class pdf2text:
	def __init__(self):
		pass

	def __call__(self, pdf_file):
		tables = camelot.read_pdf("docs/{}".format(pdf_file), pages="all")
		streams = camelot.read_pdf("docs/{}".format(pdf_file), pages="all", flavor="stream")

		table_title = []

		for stream in streams:
			data = stream.data
			data = [''.join(row) for row in data]
		for text in data:
			if "ai_tables" in text:
				table_title.append(text)

		texts = []
		table_shape = []

		for table in tables:
			data = table.data
			# append same row data to a string
			data = [' '.join(row) for row in data]
			# append every row to a string
			text = ' '.join(data)
			text = text.replace('\n', '')
			table_shape.append(table.shape)
			texts.append(text)
		return texts, table_shape, table_title


class text2vector:
	def __init__(self):
		self.Transformer = SentenceTransformer("distiluse-base-multilingual-cased-v2")

	def __call__(self, text):
		return self.Transformer.encode(text)


class cosine_sim:
	import numpy as np
	def __init__(self):
		pass

	def __call__(self, keyword_vector, table_vector):
		return np.dot(table_vector, keyword_vector) / (np.linalg.norm(table_vector) * np.linalg.norm(keyword_vector))


def execute(keyword, pdf_file):
	pdf_parser = pdf2text()
	text_vector = text2vector()
	similarity = cosine_sim()

	table_text, table_shape, table_title = pdf_parser(pdf_file)
	table_vectors = text_vector(table_text)
	keyword_vectors = text_vector(keyword)

	result = []
	for table_vector in table_vectors:
		result.append(similarity(keyword_vectors, table_vector))

	maxIdx, _ = max(enumerate(result), key=lambda x: x[1])

	outupt_table = table_text[maxIdx].split(' ')

	st.write(table_title[maxIdx])
	st.table(np.reshape(outupt_table, table_shape[maxIdx]))
	


if __name__ == "__main__":
	st.title("BDS - HW3a")
	st.divider()

	st.subheader("Please choose PDF file")
	pdf_file = st.selectbox("choose file", ("1.pdf", "2.pdf"), label_visibility="hidden")

	st.subheader("Please input keyword")
	keyword = st.text_input("input keyword", label_visibility="hidden")

	st.subheader("")
	if st.button("Execute"):
		execute(keyword, pdf_file)
