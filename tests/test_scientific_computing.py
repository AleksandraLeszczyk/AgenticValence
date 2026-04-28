import logging
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from agentic_valence.agents.scientific_computation import scientific_computing

logger = logging.getLogger()


def test_scientific_computing_search_code():
    messages = scientific_computing.invoke(
        {"messages": [HumanMessage("Write a code snippet that computes ground state energy of water.")]}
    )["messages"]
    logger.info("Calculation Mage answered: %s" % messages)
    answer = messages[-1]
    assert isinstance(answer, AIMessage)
    assert len(answer.content) > 1
    assert "RHF" in answer.content
    tools_called = [i.name for i in messages if isinstance(i, ToolMessage)]
    assert "search_code" in tools_called


def test_scientific_computing_compute():
    messages = scientific_computing.invoke(
        {"messages": [HumanMessage("Compute PES of H2 using CCSD and cc-pvdz basis.")]}
    )
    logger.info("Calculation Mage answered: %s" % messages)
    answer = messages["messages"][-1]
    assert isinstance(answer, AIMessage)
    assert len(answer.content) > 1
    assert "RCCSD" in answer.content
    tools_called = [i.name for i in messages if isinstance(i, ToolMessage)]
    assert "execute_code_via_mcp" in tools_called
    assert "search_code" in tools_called
