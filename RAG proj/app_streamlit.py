import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.title("Converted Streamlit App")

import streamlit as st
from genpactprojectRAG import load_genpactprojectRAG

st.title("📄 Chat with Your Document")
openai_key = "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"
uploaded = st.sidebar.file_uploader("Upload PDF or TXT", type=["pdf","txt"])

if openai_key and uploaded:
    qa = load_pipeline(uploaded, openai_key)
    question = st.chat_input("Ask something:")
    if question:
        result = qa.invoke({"question": question})
        st.chat_message("assistant").write(result["answer"])


