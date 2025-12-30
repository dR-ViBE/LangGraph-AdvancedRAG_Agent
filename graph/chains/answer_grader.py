from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableSequence
from langchain_groq import ChatGroq

class GradeAnswer(BaseModel):
    binary_score: bool = Field(
        description="Answer addresses the question, 'yes' or 'no'")

llm = ChatGroq(model="llama-3.1-8b-instant")
structured_llm_grader = llm.with_structured_output(GradeAnswer, method="json_mode")


system = """You are a grader assessing whether an answer addresses / resolves a question \n 
     Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question. \n
     Provide the binary score as a JSON with a single key 'binary_score' and no preamble or explanation."""

answer_prompt = ChatPromptTemplate.from_messages(
    [("system", system), ("human", "User question: \n\n {question} \n\n LLM generation: {generation}")])

answer_grader: RunnableSequence = answer_prompt | structured_llm_grader