from __future__ import annotations

import json
import sys

from schemas import OrchestratorState
from graphs.orchestrator_graph import orchestrator_graph


TASKS_JSON: str = """
[
    {
        "id": "task_1",
        "description": "Given these coordinates, I would like to get the correct address: (45.4845, 9.2040).",
        "coverage": null,
        "constraints": null,
        "inputs": "Coordinates in format (latitude, longitude).",
        "expected_output": "The correct address corresponding to the provided coordinates."
    }
]
"""

def _banner(text: str, width: int = 62) -> None:
    print("\n╔" + "═" * width + "╗")
    print("║  " + text.ljust(width - 2) + "║")
    print("╚" + "═" * width + "╝")


def main(tasks_json: str = TASKS_JSON) -> dict:
    tasks: list[dict] = json.loads(tasks_json)

    _banner(f"SOLIS         │  Orchestrator-Workers  │  tasks={len(tasks)}")

    initial_state: OrchestratorState = {
        "tasks":               tasks,
        "current_task_index":  0,
        "task_results":        [],
        "all_tasks_completed": False,
    }

    final_state: dict = orchestrator_graph.invoke(initial_state)

    _banner("GENERATED CODE SNIPPETS")
    for result in final_state.get("task_results", []):
        svc = result.get("selected_service", {})
        svc_name = svc.get("title") or svc.get("name") or "N/A"
        print(f"\n{'─' * 62}")
        print(f"  Task   : {result['task_id']} – {result['description']}")
        print(f"  Service: {svc_name}")
        print(f"  Path   : {result['path_taken']}")
        print(f"{'─' * 62}")
        print(result.get("code_snippet", "(no snippet generated)"))

    return final_state


if __name__ == "__main__":
    main()