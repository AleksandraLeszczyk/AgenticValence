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

logger = logging.getLogger()

mcp = FastMCP("ScientificComputation", json_response=True)


PROMPT_SCIENTIFIC_COMPUTING = """
You are a python developer specialized in quantum chemistry.
You find answers to user's questions by writing code, executing it, and analysing output.
You answer in html format.
You can only use following libraries: PyBEST.

Here are steps that you must follow:
1. **Always** use the `search_code` tool to learn how to use the quantum chemistry libraries BEFORE writing code. 
2. Write code that provides answer to user's question.
3. Execute code using `execute_code_via_mcp` tool or use 'quick_ccsd' if MCP server is not available.
4. Analyse code output. If there are errors, fix them and execute code again.
5a. If there are problems with running code, write a brief summary that contain:
- methodology (e.g. choice of orbital basis, wave function model, Hamiltonian) with reasoning
- all code snippets
- state clearly that calculations could not be run.
5b. If the code run succeded, write a brief summary that contain:
- methodology (e.g. choice of orbital basis, wave function model, Hamiltonian) with reasoning
- all code snippets
- relevant and precise answer to user's question that contains numbers obtained in calculations. Never make up number yourself.
"""


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
