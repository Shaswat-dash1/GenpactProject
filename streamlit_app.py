import os
import streamlit as st
from langchain.callbacks.tracers import ConsoleCallbackHandler
from langchain.globals import set_debug, set_verbose
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter

set_debug(True)
set_verbose(True)

def build_chain(pdf_file, key):
    os.environ["OPENAI_API_KEY"] = key
    loader = UnstructuredFileLoader(pdf_file)
    docs = loader.load()
    chunks = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20).split_documents(docs)
    retriever = Chroma.from_documents(chunks, OpenAIEmbeddings()).as_retriever()
    memory = ConversationBufferMemory(memory_key="chat_history", k=3, return_messages=True)
    llm = ChatOpenAI(model_name="gpt-4o", callbacks=[ConsoleCallbackHandler()])
    return ConversationalRetrievalChain.from_llm(llm, retriever, memory=memory, chain_type="stuff", verbose=True)

def main():
    st.title("📄 PDF Q&A Chatbot")
    api_key = st.sidebar.text_input("OpenAI API Key", type="password")
    pdf = st.sidebar.file_uploader("Upload PDF", type="pdf")

    if api_key and pdf:
        qa = build_chain(pdf, api_key)
        if "history" not in st.session_state:
            st.session_state.history = {"queries": [], "answers": []}
        query = st.text_input("Your question:")
        if query:
            resp = qa.run({"question": query})
            st.session_state.history["queries"].append(query)
            st.session_state.history["answers"].append(resp)
        for q, a in zip(st.session_state.history["queries"], st.session_state.history["answers"]):
            st.markdown(f"**You:** {q}")
            st.markdown(f"**Bot:** {a}")

if __name__ == "__main__":
    main()
