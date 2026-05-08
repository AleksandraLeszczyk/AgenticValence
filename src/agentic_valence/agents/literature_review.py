import os

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage 
from langchain_core.documents import Document

from dotenv import load_dotenv

from agentic_valence.config import CONFIG


load_dotenv(override=True)

PROMPT_LITERATURE_REVIEW = """
You are a knowledgeable scientist specialized in quantum chemistry with access to the datastore.
You answer questions related to quantum chemistry based on the sources using confident tone.
You are a critical thinker who has an eye for detail and do not tolerate errors or lying.
If you are not sure, don't make things up.
Write clearly if there is aknowledge gap.
Answer by giving a brief accurate summary and citations from sources.
For equations and code use HTML formatting - your answer should be ready to be displayed inside div.
Include relevant citations from context formatted as separate divs with ACS citation or source file if possible.
"""

embeddings = HuggingFaceEmbeddings(model_name=CONFIG["MODEL_EMBEDDING"])
vectordb = Chroma(persist_directory=os.environ["DB_NAME"], embedding_function=embeddings)
retriever = vectordb.as_retriever()
model_literature_review = ChatOpenAI(temperature=0, model_name=CONFIG["MODEL_KNOWLEDGE_SUMMARY"])

@tool
def search(query: str) -> str:
    """Search for information."""
    return " ".join([f"New item: {i.metadata.get('citation_acs', i.metadata.get('source', ''))} {i.page_content}" for i in retriever.invoke(query, k=10)])

literature_reviewer = create_agent(
    model_literature_review,
    tools=[search],
    system_prompt=SystemMessage(
        content=[
            {
                "type": "text",
                "text": PROMPT_LITERATURE_REVIEW,
            }
        ]
    ),
)




