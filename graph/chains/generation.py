from langchain_classic import hub
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser

llm=ChatGroq(model="llama-3.1-8b-instant")

prompt=hub.pull("rlm/rag-prompt")

generation_chain=prompt|llm|StrOutputParser()