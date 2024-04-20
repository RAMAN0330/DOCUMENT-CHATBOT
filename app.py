import os
import io
import re
import streamlit as st
from PyPDF2 import PdfReader
from docx import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
os.getenv("AIzaSyBzIZFqiM75Jx3bKNDM_eK4WL6XVlt9XeY")
genai.configure(api_key=os.getenv("AIzaSyBzIZFqiM75Jx3bKNDM_eK4WL6XVlt9XeY"))

# read all pdf and docx files and return text

def get_text(files):
    text = ""
    for file in files:
        if file.type == "application/pdf":
            pdf_reader = PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text()
        elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            doc = Document(file)
            text += " ".join([paragraph.text for paragraph in doc.paragraphs])
    return text

# split text into chunks

def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000, chunk_overlap=1000)
    chunks = splitter.split_text(text)
    return chunks  # list of strings

# get embeddings for each chunk

def get_vector_store(chunks):
    if not chunks:
        st.error("No text chunks found. Please make sure to upload some files and process them.")
        return
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key="AIzaSyBzIZFqiM75Jx3bKNDM_eK4WL6XVlt9XeY")  # type: ignore
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_file_type(file):
    if file.type == "application/pdf":
        return "PDF"
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        return "DOCX"
    else:
        return "TEXT"

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro",
                                   client=genai,
                                   temperature=0.3,
                                   google_api_key="AIzaSyBzIZFqiM75Jx3bKNDM_eK4WL6XVlt9XeY"
                                   )
    prompt = PromptTemplate(template=prompt_template,
                            input_variables=["context", "question"])
    chain = load_qa_chain(llm=model, chain_type="stuff", prompt=prompt)
    return chain

def clear_chat_history():
    st.session_state.messages = [
        {"role": "assistant", "content": "upload some files and ask me a question"}]

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key="AIzaSyBzIZFqiM75Jx3bKNDM_eK4WL6XVlt9XeY")  # type: ignore

    new_db = FAISS.load_local("faiss_index", embeddings )
    #allow_dangerous_deserialization=True
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    response = chain(
        {"input_documents": docs, "question": user_question}, return_only_outputs=True, )

    print(response)
    return response

def main():
    st.set_page_config(
        page_title="Graph based Chatbot",
        page_icon="🤖"
    )

    # Sidebar for uploading files
    with st.sidebar:
        st.title("Menu:")
        file_types = ["PDF", "DOCX", "TEXT"]
        file_type = st.selectbox("File Type", file_types)
        if file_type == "PDF":
            file = st.file_uploader(
                "Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=False)
        elif file_type == "DOCX":
            file = st.file_uploader(
                "Upload your DOCX Files and Click on the Submit & Process Button", accept_multiple_files=False)
        else:
            file = st.text_area("Paste your TEXT Files here")

        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                if file:
                    raw_text = get_text([file])
                else:
                    raw_text = file
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")

    # Main content area for displaying chat messages
    st.title("Chat with ")
    st.write("Welcome to the chat!")
    st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

    # Chat input
    # Placeholder for chat messages

    if "messages" not in st.session_state.keys():
        st.session_state.messages = [
            {"role": "assistant", "content": "upload some files and ask me a question"}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

    # Display chat messages and bot response
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = user_input(prompt)
                placeholder = st.empty()
                full_response = ''
                for item in response['output_text']:
                    full_response += item
                    placeholder.markdown(full_response)
                placeholder.markdown(full_response)
        if response is not None:
            message = {"role": "assistant", "content": full_response}
            st.session_state.messages.append(message)


if __name__ == "__main__":
    main()
