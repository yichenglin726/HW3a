import streamlit as st
import spacy
import camelot
import re
import PyPDF2


# Create a Streamlit application


class pdf2text:
    def __init__(self):
        pass

    def __call__(self, pdf_file):
        return camelot.read_pdf(pdf_file, pages='all')
       


class text2vector:
    def __init__(self):
        pass

    def __call__(self, nlp, text):
        return nlp(text)


class cosine_sim:
    def __init__(self):
        pass

    def __call__(self, vector_from_table, vector_from_keyword):
        return vector_from_keyword.similarity(vector_from_table)
    
class SearchEngine:
    def __init__(self):
       
        self.fileList = []
        self.t2v = text2vector()
        self.cs = cosine_sim()
        self.p2t = pdf2text()
        

    def clean_text(self, text):
        # Replace newlines and strip whitespace
        text = text.replace('\n', ' ').strip()
        text = text.replace('\r', ' ').strip()
        text = text.replace('\\n', ' ').strip()
        text = text.replace('\r\n', ' ').strip()
        pattern = r'[^\u4e00-\u9fffA-Za-z]'

    # Replace non-English and non-Chinese characters with an empty string
        text = re.sub(pattern, '', text)
        # Remove punctuation (optional, add more if needed)
        punctuations = '、，。！？；：「」『』（）《》&#8203;``【oaicite:0】``&#8203;'+u'\u200b'+u'\u2002'+u'\u2009'+u'\u2003'+u'\u3000'+u'\u2028'+u'\u2029'+u'\u0085'
        return ''.join(char for char in text if char not in punctuations)

    

    def add_pdf(self, fileList):
        self.fileList = fileList
        # print(self.fileList)
    
    def calculate_table_sim(self, nlp,search_vector,sentences):
        max_sim = -1
        for s in sentences:
            if s != "":
                target_vector = self.t2v(nlp, s)
                sim = self.cs(search_vector, target_vector)
                max_sim = max(max_sim,sim)
        return max_sim

    def find_title(self, table, name):
        x1, y1, x2, y2 = table._bbox

        # Read the PDF with PyPDF2
        reader = PyPDF2.PdfReader(name)
        page = reader.pages[0]

        
        # Define the area to extract text (this is an estimate)
        # Adjust the offset values as needed (e.g., 100, 50)
        title_area = (x1 , y1,x2, y2 )  # Left, top, right, bottom

        # Extract text in this area
        title = page.extract_text(title_area)

        print("Possible table title:", title)

    def pdf_to_text(self):

        tableDict = {}
        tableIndex = 0
        for uploaded_file in self.fileList:
            # Convert the uploaded file to bytes
            with open(uploaded_file.name, "wb") as f:
                f.write(uploaded_file.getbuffer())

            try:
                # Extract tables from the PDF
                tables = self.p2t(uploaded_file.name)
               

                # Display each table
                for table in tables:
                            
                    # table_texts = table.df.to_string()
                    # print(table_texts)
                   

                    sentences = [self.clean_text(cell) for row in table.df.itertuples(index=False) for cell in row]
                    tableDict[tableIndex] =  {"sentences":sentences, "table":table, "file_name":uploaded_file.name}
                   
                    tableIndex += 1

            except Exception as e:
                st.error(f"An error occurred while processing the file {uploaded_file.name}: {e}")

        
        return tableDict

    def search(self, search_str):
        self.nlp = spacy.load("zh_core_web_lg")
        tableDict = self.pdf_to_text()
        search_vector = self.t2v(self.nlp, search_str)
        best_similarity = -1
        best_key = None
        for key, value in tableDict.items():
            
            temp_similarity = self.calculate_table_sim(self.nlp, search_vector, value["sentences"])
            if (temp_similarity >= best_similarity):
                best_similarity = temp_similarity
                best_key = key

        self.find_title(tableDict[best_key]["table"], tableDict[best_key]["file_name"])
        return tableDict[best_key]["table"].df

        





        








def webUI( searchEngine):
    st.title("PDF Upload App")

    # Initialize a list to store the files

    
    # Create a file uploader widget
    uploaded_files = st.file_uploader("Choose a PDF file", type="pdf", accept_multiple_files=True)

    if len(uploaded_files) != 0:
        # You can do something with the file here
        # For now, let's just add it to our list
        # uploaded_files_list.append(uploaded_file)

        st.success("File uploaded successfully!")

        # Display the list of uploaded files (optional)
        st.write("Uploaded files:")
        for file in uploaded_files:
            st.write(file.name)

    user_input = st.text_input("Enter your query here")

# Button to read text
    if st.button('search'):
        st.write("You entered: ", user_input)
        
        searchEngine.add_pdf(uploaded_files)
        df = searchEngine.search(user_input).copy()
        new_header = df.iloc[0]
        df = df[1:]
        df.columns = new_header
        st.dataframe(df, hide_index=True)
        


def main():
    searchEngine = SearchEngine()
    webUI(searchEngine)

if __name__ == "__main__":
    main()