#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
from langchain.callbacks.tracers import ConsoleCallbackHandler
from langchain.globals import set_debug, set_verbose
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Enable logging
set_debug(True)
set_verbose(True)

os.environ["OPENAI_API_KEY"] = "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"

# Load & split documents
loader = UnstructuredFileLoader("Guide-to-Navigating-Tariff-Uncertainty.pdf.coredownload.inline.pdf")
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
chunks = text_splitter.split_documents(docs)
print(f"Indexed {len(chunks)} chunks.")

# Test retriever before building chain
retriever = Chroma.from_documents(chunks, OpenAIEmbeddings(), persist_directory="db").as_retriever()
vectordb = retriever.vectorstore
vectordb.persist()
sample_docs = retriever.get_relevant_documents("example")
print("Sample retrieved docs:", sample_docs[:1])

# Setup chain and memory
memory = ConversationBufferMemory(memory_key="chat_history", k=3, return_messages=True)
llm = ChatOpenAI(model_name="gpt-4o", callbacks=[ConsoleCallbackHandler()])
qa = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    chain_type="stuff",
    verbose=True
)

# Run interactive session
print("Ask questions (or type 'exit'):")
while True:
    query = input("You: ")
    if query.lower().strip() == "exit":
        break
    result = qa.invoke(
        {"question": query},
        config={"callbacks": [ConsoleCallbackHandler()]}
    )
    print("Bot:", result["answer"])


# In[ ]:




