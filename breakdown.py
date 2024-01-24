from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
# or langchain_community.vectorstores
from langchain.vectorstores.faiss import FAISS

# gets goes through each pdf and concatenates the text in each pdf 
def get_pdf_text(pdf_documents):
    text = ""
    for pdf in pdf_documents:
        pdf_reader = PdfReader(pdf)  # initialize reader for each pdf
        for page in pdf_reader.pages:  # loop through each page in one pdf
            text += page.extract_text()  # extract text from that page and concatenate to text
    return text

# turns the concatenated text into smaller chunks
def get_text_chunks(raw_text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,  # help stop issues from stopping in middle of word or sentence
        length_function=len
    )
    chunks = text_splitter.split_text(raw_text)
    return chunks

# turns the text chunks into vector stores
def get_vectorstore(text_chunks):
    # embeddings = OpenAIEmbeddings()
    embeddings = HuggingFaceInstructEmbeddings(
        model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore