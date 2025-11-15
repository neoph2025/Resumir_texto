import streamlit as st
from langchain_openai import OpenAI
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


prompt_esp = ChatPromptTemplate.from_template(
    "Summarize the following text, the result has to be translated into Spanish"
)


def generate_response(txt):
    llm = OpenAI(
        temperature=0,
        openai_api_key=openai_api_key
    )
    llm_esp = prompt_esp | llm | StrOutputParser()
    text_splitter = CharacterTextSplitter()
    texts = text_splitter.split_text(txt)
    docs = [Document(page_content=t) for t in texts]
    chain = load_summarize_chain(
        llm_esp,
        chain_type="map_reduce"
    )

    return chain.run(docs)

st.set_page_config(
    page_title = "Resumir texto"
)
st.title("Resumir texto")

txt_input = st.text_area(
    "Introduce tu texto",
    "",
    height=200
)

result = []
with st.form("summarize_form", clear_on_submit=True):
    openai_api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        disabled=not txt_input
    )
    submitted = st.form_submit_button("Resumir")
    if submitted and openai_api_key.startswith("sk-"):
        response = generate_response(txt_input)
        result.append(response)
        del openai_api_key

if len(result):
    st.info(response)
