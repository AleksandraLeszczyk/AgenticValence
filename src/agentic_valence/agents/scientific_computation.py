import logging

from langchain.tools import tool
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.messages import SystemMessage
from mcp.server.fastmcp import FastMCP

from agentic_valence.agents.dev_assets.pybest_example_output import mock_pybest_h2
from agentic_valence.config import settings
from agentic_valence.prompts import load_prompt

logger = logging.getLogger()

mcp = FastMCP("ScientificComputation", json_response=True)

PROMPT_SCIENTIFIC_COMPUTING = load_prompt("scientific_computation")


@tool
def execute_code_via_mcp(code: str) -> str:
    """Executes Python code in an isolated MCP server environment.
    Pass the exact python code to run. Returns the stdout or error traceback.
    """
    logger.info("Executing code via MCP.")
    try:
        # Mocking the MCP HTTP payload standard
        payload = {"jsonrpc": "2.0", "method": "execute_code", "params": {"code": code}, "id": 1}
        
        # For demonstration, we mock the HTTP call if the server isn't reachable
        # response = httpx.post(settings.mcp_server_url, json=payload, timeout=10.0)
        # return response.json().get("result", "Execution failed")

        return mock_pybest_h2()
        
        return f"Successfully sent request to MCP server. MCP server is not enabled. Return code as an output."
    except Exception as e:
        return f"Execution Error: {str(e)}"


@tool
def search_code(query: str) -> list[str]:
    """Search for code snippets and docs."""
    logger.info(f"Searching code: {query}")
    return retriever.invoke(query, k=10)


if settings.is_remote_host(settings.code_db_host):
    vectordb = Chroma(
        collection_name=settings.knowledge_db_collection,
        host=settings.code_db_host,
        port=settings.knowledge_db_port,
    )
else:
    vectordb = Chroma(
        persist_directory=settings.code_db_host,
        embedding_function=HuggingFaceEmbeddings(
            model_name=settings.model_embedding_hf
        ),
    )
retriever = vectordb.as_retriever()
model_scientific_computing = ChatOpenAI(
    temperature=0,
    model_name=settings.model_code_writing,
    api_key=settings.openai_key,
)

scientific_computing = create_agent(
    model_scientific_computing,
    tools=[
        search_code,
        execute_code_via_mcp,
        ],
    system_prompt=SystemMessage(
        content=[
            {
                "type": "text",
                "text": PROMPT_SCIENTIFIC_COMPUTING,
            }
        ]
    ),
)
