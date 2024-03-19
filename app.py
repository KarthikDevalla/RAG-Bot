# Intially read the documents. 
from PyPDF2 import PdfReader
import streamlit as st
st.title('RAG-Bot')
st.set_page_config(page_title="RAG")
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain_community.document_loaders import PyPDFLoader



def get_data_chunks(documents):
    docs = PyPDFLoader(file_path=documents).load()
    splitter=RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    data=splitter.split_documents(docs)
    chunks = filter_complex_metadata(data)
    return chunks

# Create embeddings and store them in the FIASS database
def get_embeddings_and_retrieve(data):
    embeddings=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-l6-v2',model_kwargs={'device':'cpu'})
    vector_db=Chroma.from_documents(data, embeddings)
    retriever=vector_db.as_retriever(search_kwargs={'k':5})
    return retriever


custom_prompt=""" As a helpful assistant please provide an answer for the question, based on the given information. Don't invent anything new. 
If you can't provide an answer based on the data, say you don't know the answer.
question:{question}
Use this context for answering the questions
context:{context} """

prompt=PromptTemplate(template=custom_prompt,input_variables=['question','context'])

# Build a retrieval chain to fetch the data and give the answer.
llm=Ollama(model="gemma:2b")

def get_answer(retriever,question):
    chain=RetrievalQA.from_chain_type(llm=llm,retriever=retriever,chain_type='stuff',chain_type_kwargs={'prompt':prompt})
    answer=chain.invoke({'query':question})
    return answer

files=st.file_uploader('Upload your documents here (PDFs only)',accept_multiple_files=True)
text_inp=st.text_input('What do you want to know about your document(s)?')

if st.button('Generate'):
    with st.spinner('Processing Information'):
        data=get_data_chunks('Apps_of_DS_Assignment_4.pdf')
        retriever=get_embeddings_and_retrieve(data)
        answer=get_answer(retriever, text_inp)
    st.write(answer['result'])



