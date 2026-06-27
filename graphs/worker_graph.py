from __future__ import annotations

from typing_extensions import Literal
from langgraph.graph import END, START, StateGraph

from schemas import WorkerState
from agents.service_selection import service_selection_agent
from agents.crawler          import crawler_agent
from agents.td_generation   import td_generation_agent
from agents.service_ranker  import service_ranker_agent
from agents.request_agent   import request_agent

def route_decision(
    state: WorkerState,
) -> Literal["crawler_agent", "service_ranker_agent"]:

    if state.get("no_service_found", False):
        print("[Router] no_service_found=True  → crawler_agent")
        return "crawler_agent"
    print("[Router] no_service_found=False → service_ranker_agent")
    return "service_ranker_agent"

def build_worker_graph():
    """Assembles and compiles the Worker sub-graph (Router pattern)."""

    builder = StateGraph(WorkerState)

    builder.add_node("service_selection_agent", service_selection_agent)
    builder.add_node("crawler_agent",           crawler_agent)
    builder.add_node("td_generation_agent",     td_generation_agent)
    builder.add_node("service_ranker_agent",    service_ranker_agent)
    builder.add_node("request_agent",           request_agent)

    builder.add_edge(START, "service_selection_agent")

    builder.add_conditional_edges(
        "service_selection_agent",
        route_decision,
        {
            "crawler_agent":        "crawler_agent",
            "service_ranker_agent": "service_ranker_agent",
        },
    )

    builder.add_edge("crawler_agent",       "td_generation_agent")
    builder.add_edge("td_generation_agent", "request_agent")

    builder.add_edge("service_ranker_agent", "request_agent")

    builder.add_edge("request_agent", END)

    return builder.compile()

worker_graph = build_worker_graph()