from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq

llm = ChatGroq(model="llama-3.1-8b-instant")

class GradeDocuments(BaseModel):
    """Binary score relevance check on retrieved documents"""
    binary_score: str = Field(description="Documents are relevant to question, 'yes' or 'no'")


structured_llm_grader = llm.with_structured_output(GradeDocuments, method="json_mode")


system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
    If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
    Provide the binary score as a JSON with a single key 'binary_score' and no preamble or explanation."""

grade_prompt = ChatPromptTemplate.from_messages([
    ("system", system),
    ("human", "Retrieved document:\n\n {document}\n\n User question:{question}")
])

retrieval_grader = grade_prompt | structured_llm_grader