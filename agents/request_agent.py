from __future__ import annotations
import json, pymongo, requests

from langchain.tools import tool
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI

from config import mongo_uri, LLM_STUDIO_HOST, LLM_STUDIO_API_KEY
from schemas import WorkerState
@tool
def get_full_td(service_id: str) -> str:
    """
    Retrieve full Thing Description from local MongoDB using service_id.
    """
    client = pymongo.MongoClient(mongo_uri)
    db = client["SOLIS_db"]
    collection = db["TDs"]

    td = collection.find_one({"_id": service_id})

    if not td:
        return json.dumps({"error": "service not found"})

    td["_id"] = str(td["_id"])

    return json.dumps(td, default=str)

def request_agent(state: WorkerState) -> dict:
    """
    Input:
        state["ranked_services"]

    Output:
        state["code_snippet"]
    
    Behavior:
    1. Call WoTT framework to generate a test suite from the selected service's Thing Description
    2. Generate executable Python code that calls APIs from the selected service
    """

    task = state["current_task"]
    service = state.get("ranked_services") or {}
    full_td = get_full_td(service.get("service_id"))
    response = requests.post(
        "http://localhost:8080/api/td/generate",
        json=full_td
    )
    response.raise_for_status()
    test_suite = response.json()
    service_id = service.get("service_id") or service.get("_id")
    svc_name = service.get("title") or service.get("name", "Unknown Service")

    print(f"\n[RequestAgent] Generating code for service {svc_name}")
    llm = ChatOpenAI(
        base_url=LLM_STUDIO_HOST,
        api_key=LLM_STUDIO_API_KEY,
        model="openai/gpt-oss-20b",
        temperature=0.25,
    )
    prompt = f"""
You are a Request Generation Agent.

You generate executable Python code that calls APIs from Web of Things services.

You are given:
- a task
- a selected service (lightweight description)
- a test suite generated from the service's Thing Description

You MAY optionally call a tool to retrieve the full Thing Description
if you need more details about:
- endpoints
- HTTP methods
- parameters
- schemas

IMPORTANT RULES:
- Do NOT invent endpoints
- Do NOT invent parameters
- Use only information from TD or tool output
- If unsure, call get_full_td(service_id)

TASK:
{json.dumps(task, indent=2, default=str)}

SELECTED SERVICE:
{json.dumps(service, indent=2, default=str)}

TEST SUITE:
{json.dumps(test_suite, indent=2, default=str)}

SERVICE ID:
{service_id}

OUTPUT:
Return ONLY valid Python code.
"""
    agent = create_agent(
        llm,
        tools=[get_full_td]
    )

    result = agent.invoke({
        "messages": [{"role": "user", "content": prompt}]
})
    print(result["messages"][-1].content)
    code_snippet = result["output"]

    print("  ✓ Code generated.")

    return {
        "code_snippet": code_snippet
    }