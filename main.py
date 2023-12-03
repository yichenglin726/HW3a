import camelot
from text2vec import SentenceModel
from sentence_transformers import util
import streamlit as st


class pdf2text:
    def __init__(self):
        pass

    def __call__(self, pdf_file):
        # Extract tables from a PDF and retrieve titles
        tables = camelot.read_pdf(pdf_file, pages="all")
        self.get_titles(pdf_file)

        tables_text = []
        for table, title in zip(tables, self.titles):
            tables_text.append({"title": title, "table": table.df, "table_string": table.df.to_string().replace("\\n", "")})
        
        return tables_text
    
    def get_titles(self, pdf_file):
        # Extract titles from a PDF
        tables = camelot.read_pdf(pdf_file, flavor='stream', pages="all")
        texts = []
        for table in tables:
            texts.append(table.df.to_string().replace("\\n", "").strip().split(' '))

        self.titles = []
        for text in texts:
            for word in text:
                if "ai_tables" in word:
                    ##print(text)
                    title = word + " " + text[text.index(word) + 1]
                    ##print(text[text.index(word) + 1])
                    title = title.replace("ai_tables_", "")
                    self.titles.append(title)
        

class text2vector:
    def __init__(self):
        self.model = SentenceModel('shibing624/text2vec-base-chinese')

    def __call__(self, text):
        vector = self.model.encode(text)
        return vector


class cosine_sim:
    def __init__(self):
        pass

    def __call__(self, vector_from_table, vector_from_keyword):
        # Calculate cosine similarity between two vectors
        cos_sim_val = util.pytorch_cos_sim(vector_from_table, vector_from_keyword)
        # return util.pytorch_cos_sim(vector_from_table, vector_from_keyword)
        return cos_sim_val


def main(keyword, pdf_files):
    pdf_parser = pdf2text()
    max_sim = 0
    max_sim_table = None
    max_sim_title = None
    for pdf_file in pdf_files:
        tables_text = pdf_parser(pdf_file)
        text2vector_fn = text2vector()

        for table_text in tables_text:
            table_string = table_text["table_string"]
            table_vector = text2vector_fn(table_string)
            keyword_vector = text2vector_fn(keyword)
            similarity = cosine_sim()(table_vector, keyword_vector)
            table_text["similarity"] = similarity
    
        # Find the most similar table by table_text["similarity"]
        for table_text in tables_text:
        #print(table_text["title"])
        #print(table_text["table"])
        #print(table_text["similarity"])
            if table_text["similarity"] > max_sim:
                max_sim = table_text["similarity"]
                max_sim_table = table_text["table"]
                max_sim_title = table_text["title"]
    
    return max_sim_table, max_sim_title



if __name__ == "__main__":
    # Configure Streamlit page settings
    st.set_page_config(
        page_title="AI Table Search Agent",
        page_icon="random",
        layout="centered",
        initial_sidebar_state="expanded",
        )
    max_sim_table = None
    max_sim_title = None
    # Display the application title and examples
    st.title("AI Table Search Agent")
    st.write("Example 1: 監督式學習、非監督式學習、強化學習")
    st.write("Example 2: 動物細胞、植物細胞、多細胞生物、單細胞生物、多細胞生物細胞膜、植物細胞膜")

    # Define the list of PDF files
    pdf_files = ["docs/1.pdf", "docs/2.pdf"]
    #print(pdf_files)

    # Display the input keyword field
    st.subheader("Input a keyword")
    print('submit status:', st.session_state)
    keyword=''
    keyword = st.text_input("Key in the fill-in-the-blank box with Keyword")
    submit_button = st.button("Search")
    #if "submit_button_state" not in st.session_state:
     #   st.session_state.submit_button_state = False

    #if submit_button or st.session_state.submit_button_state:
        #st.session_state.submit_button_state = True

    #if submit_button or keyword:

    if "submit_button_state" not in st.session_state:
        st.session_state.submit_button_state = False
    if submit_button or st.session_state.submit_button_state:
        st.session_state.submit_button_state = True
        #if submit_button or (keyword and pdf_files):
        print(keyword)
        print(pdf_files)
        print(submit_button)
        #submit_button = False
        with st.spinner('Searching the most suitable table'):
        #table = main(keyword, pdf_file)
        #st.write("Output:", table)
            max_sim_table, max_sim_title = main(keyword, pdf_files)
            # show the most similar table 
            st.subheader(max_sim_title)
            st.write(max_sim_table)
        #if "submit_button_state" not in st.session_state:
            st.session_state.submit_button_state = False


