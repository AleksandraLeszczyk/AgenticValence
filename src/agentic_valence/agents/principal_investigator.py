import logging

from langchain.tools import tool
from langchain.agents import create_agent
from langchain.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from agentic_valence.agents.literature_review import literature_reviewer
from agentic_valence.agents.scientific_computation import scientific_computing
from agentic_valence.agents.viz_creator import viz_creator
from agentic_valence.config import settings
from agentic_valence.prompts import load_prompt


logger = logging.getLogger()

PI_PROMPT = load_prompt("principal_investigator")


@tool
def LiteratureReview(question: str) -> dict:
    """Search for information."""
    logger.info("Asking Literature Sage: %s" % question)
    return literature_reviewer.invoke({"messages": [HumanMessage(question)]})["messages"][
        -1
    ].content


@tool
def ScientificComputing(question: str) -> list[str]:
    """Write and execute code."""
    logger.info("Asking Code Mage: %s" % question)
    return scientific_computing.invoke({"messages": [HumanMessage(question)]})["messages"][
        -1
    ].content


@tool
def VizCreator(question: str) -> list[str]:
    """Creates interactive pictures and saves them to registry."""
    logger.info("Asking VizCreator: %s" % question)
    viz_creator.invoke({"messages": [HumanMessage(question)]})["messages"][-1].content


model_principal_investigator = ChatOpenAI(
    model=settings.model_principal_investigator,
    api_key=settings.openai_key,
)
principal_investigator = create_agent(
    model_principal_investigator,
    tools=[
        LiteratureReview,
        ScientificComputing,
        VizCreator,
    ],
    system_prompt=SystemMessage(content=[{"type": "text", "text": PI_PROMPT}]),
    debug=True,
)
