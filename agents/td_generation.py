
from __future__ import annotations
from unittest import result
from langchain_openai import ChatOpenAI
import requests, json
from schemas import WorkerState
from config import LM_STUDIO_HOST, LM_STUDIO_API_KEY
from pymongo import MongoClient
from config import mongo_uri


def validate_td(td_content: str) -> dict:
    """
    Validate the generated Thing Description (TD) using the WoTT framework.
    """
    response = requests.post(
        "http://localhost:8080/api/td/validate",
        json={"td": td_content}
    )
    result = response.json()

    if result.get("valid"):
        return td_content
    else:
        #TODO: modifica e validazione human-in-the-loop tramite interfaccia web di WoTT
        print("TD non valida")
        return None

def store_td(td_content: dict) -> None:
    client = MongoClient(mongo_uri)
    db = client['SOLIS_db']
    collection = db["TDs"]
    result = collection.insert_one(td_content)
    return result.inserted_id

def build_service_profile(td: dict) -> dict:

    properties = td.get("properties") or {}
    actions = td.get("actions") or {}

    property_descriptions = [
        p.get("title", "") + " " + p.get("description", "")
        for p in properties.values()
    ]

    action_descriptions = [
        a.get("title", "") + " " + a.get("description", "")
        for a in actions.values()
    ]

    return {
        "service_id": str(td.get("_id")),
        "title": td.get("title"),
        "description": td.get("description"),

        "capabilities": (
            property_descriptions[:10]
            + action_descriptions[:10]
        )
    }

def td_generation_agent(state: WorkerState) -> dict:
    """
    Input:
        state["service_description"]
    Output:
        state["selected_service"]
    Behavior:
    1. Call preprocessing pipeline
    2. Generate a W3C WoT Thing Description (TD) from the service description
    """
    description: str = state.get("service_description", "")
    task: dict       = state.get("current_task", {})
    tid: str         = task.get("id", "?")

    print(f"\n[TDGenerationAgent] Task '{tid}' | Generating Thing Description…")
    block_based = state.get("service_description")
    llm_input = requests.post(
        "http://localhost:8080/preprocessing_pipeline", #TODO
        json=block_based
    )
    llm = ChatOpenAI(
        base_url=LM_STUDIO_HOST,
        api_key=LM_STUDIO_API_KEY,
        model="qwen/qwen3.5-9b",
        temperature=0.25,
    )
    system_prompt = f"""
You are an expert in W3C Web of Things (WoT), Semantic Web technologies, and domain modeling for mobility and mobility-adjacent services.

Your task is to transform API/service documentation into a valid W3C Web of Things (WoT) Thing Description 2.0 (https://www.w3.org/TR/wot-thing-description-2.0/) in JSON-LD.

#### CORE BEHAVIOR:

- You must strictly follow the W3C WoT Thing Description specification (version 2.0).
- You must extract and structure information with high precision from the provided input.
- You must NOT hallucinate or invent any information.
- If information is missing, omit it.

#### OUTPUT CONSTRAINTS (STRICT):

- Output ONLY one JSON-LD Thing Description.
- Do NOT include explanations, comments, or markdown.
- The output MUST be syntactically valid JSON.
- The output MUST comply with WoT TD 2.0:
  - include required fields: title, security, securityDefinitions and @context
  - include correct `type` fields in Data Schemas: null, boolean, string, number, integer, array, object
  - enrich the Data Schemas correctly using the information you have (min/maximum, min/maxLength, min/maxItems, required, unit, read/writeOnly...)

#### SEMANTIC REQUIREMENTS:

- Use semantic annotations (`@type`, `@context`) where possible.
- Reuse established ontologies (i.e. saref, sosa, schema.org) and link them in `@context` array when used.
- Annotate:
  - Thing
  - properties
  - actions
  - events

#### MODEL EXTRACTION RULES:
When analyzing the input, extract:
1. Functional aspects (capabilities, operations)
2. Interaction model (HTTP methods, endpoints, parameters)
3. Contextual constraints (geography, time, dependencies)
4. Non-functional aspects (performance, cost, reliability, security)
5. Accessibility and sustainability aspects

#### INTERACTION MODEL MAPPING

- Properties are objects that exposes state of the Thing. This state can then be retrieved (read -> GET) and/or updated (write -> POST/PUT/PATCH)
- Actions are objects that allows to invoke a function of the Thing, which manipulates state (e.g., toggling a lamp on or off) or triggers a process on the Thing (e.g., dim a lamp over time) (every HTTP method).
- Events are object thatdescribes an event source, which asynchronously pushes event data to Consumers (webhook or streaming)
- It is NOT mandatory to have both properties, actions and events, insert them where and how it is necessary and don't hallucinate them.
- Every elements in properties/actions/event must have a `forms` array correctly with:
  - href
  - op: one of readproperty, writeproperty, observeproperty, unobserveproperty, invokeaction, queryaction, cancelaction, subscribeevent or unsubscribeevent
  - htv:methodName ("htv": "http://www.w3.org/2011/http#")
  - If you can't infer these elements, it's probably not a property, action, or event.

#### FAIL-SAFE RULES:

- If endpoints are unclear → do NOT include them
- If schemas are incomplete → simplify but stay consistent
- If semantic types are uncertain → omit rather than guess
"""
    user_prompt = f"""
Convert the following HTML API/service documentation into a semantically enriched W3C WoT Thing Description (TD).

#### INPUT DESCRIPTION

The input is a JSON object for each web page of the service/API documentation, containing:

- `url`: the URL of the original page.
- `html`: the reconstructed HTML of the page, which preserves the order of 
  content and includes:
    - Annotated text blocks: original HTML tags (`tag_name`) are preserved.
      - Text spans matched by semantic labels are wrapped in `<span class="LABEL">...</span>`.
    - Code examples (`code_example`) and section headers (`section_header`) included as original HTML blocks.
    - Comments indicating removed blocks: `<!-- BLOCKS_REMOVED block_ids -->`.

Notes:
1. Each HTML block may contain multiple labels applied to different text spans.
2. Removed blocks are indicated with HTML comments and should guide your understanding of missing context.
3. Labels are **hints**, indicating why a block may be relevant, including:
   - Functional information
   - Non-functional requirements (costs, performance, reliability, accessibility, sustainability)
   - Contextual and operational constraints
4. The reconstructed HTML is intended to provide a faithful, ordered, semantically enriched view of the documentation.

#### IMPORTANCE OF SEMANTIC LABELS

The semantic labels are fundamental for understanding:

- Functional characteristics: what the service does and what users can do with it.  
- Non-functional characteristics: constraints, costs, performance, accessibility, sustainability.  
- Contextual information: geographical coverage, dependencies, operational limitations.  

Use these labels to enrich your TD descriptions. They are **not hard constraints**, but **hints** that indicate why a block is important and what kind of information it contains.  
The labels are provided as guidance and can be treated as a variable called `LABEL_GUIDANCE`:

LABEL_GUIDANCE:
1. Identity and Purpose
- service purpose: general purpose of th service (e.g. This service allows users to move across the city)
- service domain (e.g. urban mobility, logistics, tourism)
- target user: kind of users (e.g. commuters, tourists, businesses)

2. Core Functionalities
- transport capability (e.g. "check schedules”, “book a ride”, “reserve a bike")
- route and navigation function (e.g. planning, routes, stops, itineraries)
- booking and reservation function (e.g. reservations, rentals, tickets)
- real time information function (e.g. delay, availability, traffic, disruption)
- payment function (e.g. payment, rates, subscriptions, fares)
- user account function

3. Interaction / API behavior
- API interaction pattern (e.g. request/response, subscription, polling)
- API method definition (action, properties, event, GET/POST/PUT/DELETE/)

4. Operational and Contextual Constraints
- geographical coverage: served area (e.g. city, zones, regions)
- operational limitations: (e.g. time windows, zones, access restrictions)
- unit of measurement of data
- regulatory constraints (e.g. emission zones, traffic rules)
- dependency on external context (e.g. weather, traffic, events)

5. Non-functional Aspects
- performance characteristics
- reliability and availability characteristics
- scalability characteristics
- security and privacy aspects
- cost model (e.g. pricing, free tiers, pay-per-use)
- third part services dependency (e.g. weather, maps, payments)
- multi modal support

6. Accessibility and Inclusivity
- accessibility feauture
- user physical constraints supports
- language and localization

7. Sustainability and Environmental Impact
- environmental impact (e.g. emissions, footprint)
- sustainability features (e.g. sharing, electrical vehicles)
- energy source or fuel characteristics
- ecological regulation compliance

#### FUNCTIONAL & NON-FUNCTIONAL UNDERSTANDING

- Use the content and semantic labels to infer:
  - What the service does (functionalities)
  - How it can be used (actions/properties/events)
  - Contextual or operational limitations
  - Non-functional constraints (cost, performance, reliability, accessibility, sustainability)

#### EXAMPLES AND TECHNICAL DETAILS

- Carefully analyze all code examples in the input (HTTP requests, responses, parameters).
- Use them to infer actual operations, populate `forms`, input/output schemas, and confirm interaction patterns.

#### INPUT

{llm_input}
"""
    td =llm.invoke({
        "messages": [{"role": "system", "content": system_prompt},
                     {"role": "user", "content": user_prompt}],
    })
    validated_td = validate_td(td.content)
    service_id = store_td(validated_td)
    profile = build_service_profile(validated_td)
    selected_service = [
        {
            "service_id": service_id,
            "title": profile.get("title"),
            "capabilities": profile.get("capabilities")
        }
    ]
    return { "ranked_services": selected_service }