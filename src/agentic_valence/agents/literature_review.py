from langchain.agents import create_agent
from langchain_chroma import Chroma
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.tools import tool

from agentic_valence.config import settings

PROMPT_LITERATURE_REVIEW = """
You are a knowledgeable scientist specialized in quantum chemistry with access to the datastore.
You answer questions related to quantum chemistry based on the sources using confident tone.
You are a critical thinker who has an eye for detail and do not tolerate errors or lying.
If you are not sure, don't make things up.
Write clearly if there is acknowledge gap.
Answer by giving a brief accurate summary and citations from sources.
For equations and code use HTML formatting - your answer should be ready to be displayed inside div.
After the summary, include the most relevant quotes with acs sources inside separate divs.
"""

if settings.is_remote_host(settings.knowledge_db_host):
    vectordb = Chroma(
        collection_name=settings.knowledge_db_collection,
        host=settings.knowledge_db_host,
        port=settings.knowledge_db_port,
    )
else:
    vectordb = Chroma(
        persist_directory=settings.knowledge_db_host,
        embedding_function=OpenAIEmbeddings(
            model=settings.model_embedding, api_key=settings.openai_key
        ),
    )

retriever = vectordb.as_retriever()
model_literature_review = ChatOpenAI(
    temperature=0,
    model_name=settings.model_knowledge_summary,
    api_key=settings.openai_key,
)


@tool
def search(query: str) -> str:
    """Search for information."""
    return " ".join(
        [
            f"New item: {i.metadata.get('citation_acs', i.metadata.get('source', ''))} {i.page_content}"
            for i in retriever.invoke(query, k=10)
        ]
    )


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
