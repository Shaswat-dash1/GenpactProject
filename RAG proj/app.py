{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ff081b20",
   "metadata": {},
   "outputs": [],
   "source": [
    "import streamlit as st\n",
    "from pipeline import load_pipeline\n",
    "\n",
    "st.title(\"📄 Chat with Your Document\")\n",
    "openai_key = st.sidebar.text_input(\"OpenAI API Key\", type=\"password\")\n",
    "uploaded = st.sidebar.file_uploader(\"Upload PDF or TXT\", type=[\"pdf\",\"txt\"])\n",
    "\n",
    "if openai_key and uploaded:\n",
    "    qa = load_pipeline(uploaded, openai_key)\n",
    "    question = st.chat_input(\"Ask something:\")\n",
    "    if question:\n",
    "        result = qa.invoke({\"question\": question})\n",
    "        st.chat_message(\"assistant\").write(result[\"answer\"])\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.13"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
