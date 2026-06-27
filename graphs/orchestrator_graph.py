from __future__ import annotations

from typing import List
from langgraph.graph import END, START, StateGraph

from schemas import OrchestratorState, WorkerState
from graphs.worker_graph import worker_graph

def dispatch_to_worker(state: OrchestratorState) -> dict:
    tasks: List[dict] = state["tasks"]
    idx: int          = state["current_task_index"]
    task: dict        = tasks[idx]

    print(
        f"\n{'═' * 62}\n"
        f"[Orchestrator] Dispatching task {idx + 1}/{len(tasks)}\n"
        f"  id  : {task.get('id', 'N/A')}\n"
        f"  desc: {task.get('description', 'N/A')}\n"
        f"{'═' * 62}"
    )

    # ── Initialise a clean WorkerState for this task ──────────
    worker_initial: WorkerState = {
        "current_task":        task,
        "query_result":        None,
        "no_service_found":    False,
        "service_description": None,
        "selected_service":    None,
        "code_snippet":        None,
    }

    worker_result: WorkerState = worker_graph.invoke(worker_initial)

    svc = worker_result.get("selected_service") or {}
    path = (
        "crawler → td_generation"
        if worker_result.get("no_service_found")
        else "service_ranker"
    )

    task_result: dict = {
        "task_id":          task.get("id", f"task_{idx}"),
        "description":      task.get("description", ""),
        "selected_service": svc,
        "code_snippet":     worker_result.get("code_snippet", ""),
        "path_taken":       path,
    }

    print(f"[Orchestrator] ✓ Task {idx + 1} complete (path: {path})")

    return {
        "task_results":       state.get("task_results", []) + [task_result],
        "current_task_index": idx + 1,
    }


def aggregate_results(state: OrchestratorState) -> dict:
    results: List[dict] = state.get("task_results", [])

    print(
        f"\n{'═' * 62}\n"
        f"[Orchestrator] All {len(results)} task(s) completed.\n"
        f"{'═' * 62}"
    )
    for r in results:
        svc = r.get("selected_service", {})
        svc_name = svc.get("title") or svc.get("name") or "N/A"
        print(
            f"  ✓ [{r['task_id']}]  "
            f"service = {svc_name:<30}  "
            f"path = {r['path_taken']}"
        )

    return {"all_tasks_completed": True}

def should_continue(state: OrchestratorState) -> str:
    if state["current_task_index"] < len(state["tasks"]):
        return "dispatch_to_worker"
    return "aggregate_results"

def build_orchestrator_graph():
    builder = StateGraph(OrchestratorState)

    builder.add_node("dispatch_to_worker", dispatch_to_worker)
    builder.add_node("aggregate_results",  aggregate_results)

    builder.add_edge(START, "dispatch_to_worker")

    builder.add_conditional_edges(
        "dispatch_to_worker",  # source node
        should_continue,       # routing function
        {                      # name → node mapping
            "dispatch_to_worker": "dispatch_to_worker",   # loop
            "aggregate_results":  "aggregate_results",    # exit
        },
    )

    builder.add_edge("aggregate_results", END)

    return builder.compile()

orchestrator_graph = build_orchestrator_graph()