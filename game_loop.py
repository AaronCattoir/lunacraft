import logging
import os
import asyncio
import json
import re
from typing import Dict, List, Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError
from openai import AsyncAzureOpenAI
import instructor
from collections import deque
from azure.identity import DefaultAzureCredential  
from azure.keyvault.secrets import SecretClient  

# ----- CONFIG -------
'''
load_dotenv()
azure_key = os.getenv('AZURE_EAST')
if not azure_key:
    raise EnvironmentError("AZURE_EAST not set")
'''

#------ Azure Init

KEY_VAULT_URL = "https://cattoirkeyvault.vault.azure.net/"  
SECRET_NAME = "AZUREEAST"  
  
# Replace with your user-assigned managed identity client ID  
USER_ASSIGNED_MSI_CLIENT_ID = ""  
  
credential = DefaultAzureCredential(managed_identity_client_id=USER_ASSIGNED_MSI_CLIENT_ID)  
  
client = SecretClient(vault_url=KEY_VAULT_URL, credential=credential)  
  
# Now you can get the secret  
secret = client.get_secret(SECRET_NAME)  
print(f"Secret value: {secret.value}")  

# Get the secret value  
try:  
    secret_bundle = client.get_secret(SECRET_NAME)  
    azure_key = secret_bundle.value  
except Exception as e:  
    raise EnvironmentError(f"Failed to retrieve secret '{SECRET_NAME}': {e}")  

if not azure_key:  
    raise EnvironmentError(f"Secret '{SECRET_NAME}' is empty or not found")  

#------- Model

model_deployment = "gpt-4o-mini"
logging.basicConfig(level=logging.INFO, format='[%(asctime)s][%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger("game_agent")

# ----- DATA CLASSES -------

class HistoryBuffer:
    def __init__(self, max_events=10):
        # Holds last N events/descriptions as strings
        self.max_events = max_events
        self.events = deque(maxlen=max_events)

    def add_event(self, event_str: str):
        """
        Add a new event to the history buffer.
        Shortens or skips empty events if needed before appending.
        """
        event_str = event_str.strip()
        if not event_str:
            return
        # Optional: truncate long events
        max_len = 500  # max chars per event to keep
        if len(event_str) > max_len:
            event_str = event_str[:max_len] + "…"
        self.events.append(event_str)

    def get_summary(self) -> str:
        """
        Return a concatenated, formatted summary of all stored events suitable for LLM context.
        """
        if not self.events:
            return "No previous events."
        summary = "Recent events:\n"
        for i, event in enumerate(self.events, 1):
            summary += f"{i}. {event}\n"
        return summary

    def reset(self):
        self.events.clear()


class LLMObjectUpdate(BaseModel):
    object: str = Field(..., description="Name of the object being updated or added")
    state: Optional[str] = Field(None, description="The state or description of the object")
    category: Optional[str] = Field(
        None,
        description="Category of the object, e.g., 'room_object' or 'interactable'"
    )

class LLMNPCUpdate(BaseModel):
    npc: str = Field(..., description="Name of the NPC")
    updated_goals: Optional[List[str]] = Field(default=None, description="Updated goals for the NPC")
    score: Optional[float] = Field(default=None, description="Updated score for the NPC")

class LLMGoalUpdates(BaseModel):
    player: Optional[List[str]] = Field(default=None, description="Updated player goals")
    npc_goals: Optional[Dict[str, List[str]]] = Field(default=None, description="Updated goals for each NPC")

class LLMSimulationOutcome(BaseModel):
    narration: str = Field(..., description="Narrative description of the simulation outcome")
    dialogue: Optional[List[Dict[str, str]]] = Field(
        default=None,
        description="Dialogue lines as a list of speaker-line pairs"
    )
    npc_updates: Optional[List[LLMNPCUpdate]] = Field(
        default=None,
        description="List of NPC state updates"
    )
    object_updates: Optional[List[LLMObjectUpdate]] = Field(
        default=None,
        description="List of object updates including category"
    )
    goal_updates: Optional[LLMGoalUpdates] = Field(
        default=None,
        description="Updated goals for player and NPCs"
    )

class LLMDialogue(BaseModel):
    speaker: str
    line: str

class LLMContestOutcome(BaseModel):
    narration: str
    dialogue: Optional[List[LLMDialogue]] = None
    npc_updates: Optional[List[LLMNPCUpdate]] = None
    object_updates: Optional[List[LLMObjectUpdate]] = None
    goal_updates: Optional[LLMGoalUpdates] = None

class PlayerAgent:
    def __init__(self, player_state):
        self.name_id = "player"
        self.name_long = "Player"
        self.state = player_state 

class Interactable(BaseModel):
    id: str = Field(..., description="A unique identifier for the iteractable")
    name_long: str = Field(..., description="A simple name for an iteractable")
    description: Optional[str] = Field(None, description="A detailed description of the iteractable")
    effect: Optional[str] = Field(None, description="The effect or ability granted by the iteractable")
    wearable: Optional[bool] = Field(None, description="Indicates if the iteractable can be worn")
    holdable: Optional[bool] = Field(None, description="Indicates if the iteractable can be held")
    held: Optional[bool] = Field(None, description="Indicates if the iteractable is currently being held")
    worn: Optional[bool] = Field(None, description="Indicates if the iteractable is currently being worn")
    location: Optional[str] = Field(None, description="The current location of the iteractable")
    value: Optional[int] = Field(None, description="The value of the iteractable")
    weight: Optional[float] = Field(None, description="The weight of the iteractable")
    type: Optional[str] = Field(None, description="The category or type of interactable, such a door, person, iteractable, weapon")
    
class InteractableList(BaseModel):
    interactables: List[Interactable]

class PlayerState(BaseModel):
    current_room_id: str
    last_room_id: Optional[str] = None
    inventory: List[str] = Field(default_factory=list)
    goals: List[str] = Field(default_factory=list)  # player’s active goals/tasks

class IntentObjectMapping(BaseModel):
    intent: str
    associated_characters: List[str] = Field(default_factory=list)
    associated_interactables: List[str] = Field(default_factory=list)

class QueryAnalysis(BaseModel):
    user_query: str
    classification: List[str]
    explanation: str
    characters_mentioned: List[str] = Field(default_factory=list)
    characters_matched: List[str] = Field(default_factory=list)
    interactables_mentioned: List[str] = Field(default_factory=list)
    interactables_matched: List[str] = Field(default_factory=list)
    intent_objects: List[IntentObjectMapping] = Field(default_factory=list)
    competition: Optional[bool] = False
    competitors: Optional[List[str]] = []
    move_text: Optional[str] = None 

class CompQueryAnalysis(BaseModel):
    user_query: str
    classification: List[str]
    explanation: str
    strategy: str = Field(..., description="The methods in which this player would approach the event.")
    competition: Optional[bool] = False
    competitors: Optional[List[str]]
    move_text: Optional[str] = None 

class NPCState(BaseModel):
    name_id: str
    goals: List[str] = Field(default_factory=list)
    dialogue_prompt: str
    backstory: str
    inventory: List[str] = Field(default_factory=list)
    autonomous_goals: List[str] = Field(default_factory=list)
    intent_history: List[str] = Field(default_factory=list)
    reaction_history: List[str] = Field(default_factory=list)
    score: float = 0.0
    physical_description: Optional[str] = None 

class NPCParent(BaseModel):
    name_id: str
    name_long: str
    state: NPCState
    def apply_outcome(self, outcome: "NPCOutcome"):
        assert outcome.name_id == self.name_id
        self.state.score = outcome.updated_score
        self.state.intent_history.append(outcome.last_intent)
        self.state.reaction_history.append("Success" if outcome.success else "Failure")
        if outcome.updated_autonomous_goals:
            self.state.autonomous_goals = outcome.updated_autonomous_goals
        logger.info(f"Updated {self.name_long}: Score={self.state.score}, Goals: {self.state.autonomous_goals}")

class NPCParentList(BaseModel):
    npcs: List[NPCParent]

class NPCIntent(BaseModel):
    name_id: str
    intented_reaction: str

class NPCOutcome(BaseModel):
    name_id: str
    last_intent: str
    success: bool
    updated_score: float
    updated_autonomous_goals: List[str]
    notes: str = ""

class ScenarioOutcome(BaseModel):
    outcomes: List[NPCOutcome]
    narration: str

class RoomState(BaseModel):
    description: str
    objects: List[str] = Field(default_factory=list)
    hazards: List[str] = Field(default_factory=list)
    ambient_noise: str = ""

class Room(BaseModel):
    name_id: str
    name_long: str
    npcs: List[NPCParent] = Field(default_factory=list)
    state: RoomState
    exits: Dict[str, str] = Field(default_factory=dict)  # e.g., {"north": "gamma_square"}

class NPCDialogueResponse(BaseModel):
    name_id: str
    dialogue: str
    updated_autonomous_goals: List[str]

class Action:
    def __init__(self, intent: str, target: Optional[str] = None, description: Optional[str] = None):
        self.intent = intent
        self.target = target
        self.description = description
    def __repr__(self):
        return f"Action(intent={self.intent}, target={self.target})"

class Plan:
    def __init__(self, actions: List[Action]):
        self.actions = actions
        self.current_step = 0
    def next_action(self) -> Optional[Action]:
        if self.current_step < len(self.actions):
            action = self.actions[self.current_step]
            self.current_step += 1
            return action
        return None

class LLMAction(BaseModel):
    intent: str
    target: Optional[str]
    description: Optional[str]
class LLMPlan(BaseModel):
    steps: List[LLMAction]

class GameWorld(BaseModel):
    player_state: PlayerState
    rooms: Dict[str, Room]




# ---- OpenAI/Instructor client -----
client_raw = AsyncAzureOpenAI(
    api_version="2024-12-01-preview",
    azure_endpoint="https://cattoiragenteast.openai.azure.com/",
    api_key=azure_key,
)
client = instructor.from_openai(client_raw)

#------ Globals 
DIRECTIONS = {"north", "south", "east", "west", "up", "down"}

COMPETITIVE_VERBS = {"attack","fight","challenge","compete","duel","race","contest","battle","steal","sabotage"}

INTENT_SYNONYMS = {
    'explore': 'Look',
    'investigate': 'Look',
    'search': 'Look',
    'scout': 'Look',
    'survey': 'Look',
    'examine': 'Look',
    'inspect': 'Look',
    'observe': 'Look',
    'view': 'Look',
    'look': 'Look',
    'see': 'Look',

    'talk': 'Talk',
    'ask': 'Talk',
    'speak': 'Talk',
    'question': 'Talk',
    'greet': 'Talk',
    'converse': 'Talk',

    'interact': 'Interact',
    'manipulate': 'Interact',
    'press': 'Interact',
    'engage': 'Interact',
    'engage with': 'Interact',

    'use': 'Use',
    'operate': 'Use',
    'activate': 'Use',

    'take': 'Take',
    'grab': 'Take',
    'pick up': 'Take',
    'collect': 'Take',
    'loot': 'Take',

    'equip': 'Equip',
    'wear': 'Equip',
    'put on': 'Equip',

    'give': 'Give',
    'offer': 'Give',
    'hand': 'Give',

    'go': 'Move',
    'go to': 'Move',
    'move': 'Move',
    'move to': 'Move',
    'walk to': 'Move',
    'travel': 'Move',
    'travel to': 'Move',
    'head': 'Move',
    'head to': 'Move',
    'exit': 'Move',
    'leave': 'Move',
}


# ---- SAVE/LOAD -----
def save_game_state(file_path: str, world: GameWorld):
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(world.model_dump(), f, indent=2)
    logger.info(f"Game saved to {file_path}")

def load_game_state(file_path: str) -> GameWorld:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Save file not found: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    world = GameWorld.model_validate(data)
    logger.info(f"Game loaded from {file_path}")
    return world

# ---- IMAGES -----
async def generate_dalle_image_for_scene(room: Room, history_buffer=None) -> str:
    """
    Extracts key info from the room and its NPCs to generate a DALL·E prompt,
    calls Azure OpenAI's DALL·E model, and returns the image URL.
    """
    # Build prompt describing room and NPCs
    history_text = ""

    if history_buffer and len(history_buffer) > 0:  
        # Access only the most recent item  
        recent_entry = history_buffer[-1]  # This gets the last item in the deque  
        history_text = f"Recent narrative history:\n{recent_entry}\n"   

    npc_descriptions = []
    for npc in room.npcs:
        desc = npc.state.physical_description or "Unknown appearance"
        npc_descriptions.append(f"{npc.name_long}: {desc}")

    prompt = (
        "Important: NO TEXT or LABELS"
        f"Create trippy pixelated 8-bit style art including the location and characters."
        f"Recent narrative history:\n{history_text}\n\n"
        "The image should be detailed and glitchy"
        "Focus on the characters and the action."
        f"Location: {room.name_long}, described as: {room.state.description}. "
        f"Characters present:\n" + "\n".join(npc_descriptions) + "\n"
        "No text, no words, no letters visible in the image."
    )

    messages = [{"role": "user", "content": prompt}]

    # Call Azure DALL·E openAI image generation
    response = await client_raw.images.generate(
        prompt=prompt,
        model="dall-e-3",   # Use your deployed Azure DALL·E model name
        size="1024x1024",
        n=1
    )
    image_url = response.data[0].url
    return image_url

# ---- PLANNING -----

def build_plan_prompt(player_intent, room, player_action):
    allowed_intents = player_intent.classification

    # Canonical matching for NPCs
    matched_npcs = set()
    for c in player_intent.characters_matched:
        for npc in room.npcs:
            if canon_id(c) == canon_id(npc.name_long):
                matched_npcs.add(npc.name_long)
    allowed_npcs = matched_npcs
    # Optionally fallback to all present if no match:
    # if not allowed_npcs: allowed_npcs = {npc.name_long for npc in room.npcs}

    # Canonical objects
    allowed_objs = set()
    for o in (player_intent.interactables_matched or []):
        for obj in room.state.objects:
            if canon_id(o) == canon_id(obj):
                allowed_objs.add(obj)
    allowed_objs |= set(room.state.objects)

    npc_summaries = "\n".join(
        f"{npc.name_long}: score={npc.state.score}, speaking_style={npc.state.dialogue_prompt}, goals={npc.state.goals}" for npc in room.npcs
    )
    interactables = ", ".join(room.state.objects) if room.state.objects else "none"
    system_content = (
        "You generate minimal, strictly relevant, step-by-step action plans for a text RPG game. "
        "Use exactly these canonical intents: Look, Talk, Interact, Take, Use, Equip, Give. No synonyms."
    )
    user_content = (
        f"Given the player action: \"{player_action}\"\n"
        f"Allowed intents: {allowed_intents}\n"
        f"Allowed NPC targets: {list(allowed_npcs)}\n"
        f"Allowed object targets: {list(allowed_objs)}\n"
        f"Room objects: {interactables}\n"
        f"NPC summaries:\n{npc_summaries}\n\n"
        "Instructions:\n"
        "- Only generate plan steps for the NPCs and objects in the allowed lists above.\n"
        "- Do NOT invent, pluralize, or generalize.\n"
        "- Never use targets not in these lists or invent new ones.\n"
        "- Plan should be concise; avoid redundancy and off-topic actions."
    )
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content}
    ]

async def extract_npcs_from_narration(narration: str) -> List[NPCParent]:
    prompt = (
        f"From the following narration, extract ALL newly introduced NPCs. "
        f"Return a JSON object: {{ \"npcs\": [ <list of NPCs with fields name_id, name_long, dialogue_prompt (if any), backstory (if any)> ] }}. "
        f"If none, return {{\"npcs\":[]}}.\n\n"
        f"NARRATION:\n{narration}\n"
    )
    messages = [{"role": "user", "content": prompt}]
    try:
        response = await client.chat.completions.create(
            model=model_deployment,
            messages=messages,
            response_model=NPCParentList,
            temperature=0.0,
            max_tokens=800,
        )
        return response.npcs  # List[NPCParent]
    except Exception as e:
        logger.warning(f"No new NPCs found or invalid: {e}")
        return []
    
async def extract_interactables_from_narration(narration: str) -> List[Interactable]:
    prompt = (
        f"From the following narration, extract new interactable objects that would be available to a local reader in that space."
        f"If none, return {{\"interactables\":[]}}.\n\n"
        f"NARRATION:\n{narration}\n"
    )
    messages = [{"role": "user", "content": prompt}]
    try:
        response = await client.chat.completions.create(
            model=model_deployment,
            messages=messages,
            response_model=InteractableList,
            temperature=0.0,
            max_tokens=800,
        )
        return response.interactables  # List[Interactable]
    except Exception as e:
        logger.warning(f"No new interactables found or invalid: {e}")
        return []

async def process_plan_actions(plan, room, player_query, save_file, history_buffer):
    # Get player goals (from world or room)
    world_ref = getattr(room, 'world', None)
    if world_ref:
        player_goals = ', '.join(world_ref.player_state.goals)
    elif hasattr(room, 'player_state'):
        player_goals = ', '.join(room.player_state.goals)
    else:
        player_goals = "none"

    npc_goal_block = "\n".join(
        f"{npc.name_long} | Goals: {', '.join(npc.state.goals)} | Description: {npc.state.backstory} | Speaking style: {npc.state.dialogue_prompt or 'N/A'}"
        for npc in room.npcs
    )

    plan_desc = "\n".join(
        f"- {action.intent} {action.target or ''}: {action.description or ''}" for action in plan.actions
    )

    room_desc = (
        f"Room: {room.name_long}\n"
        f"Description: {room.state.description}\n"
        f"Objects: {', '.join(room.state.objects)}\n"
        f"NPCs: {', '.join(n.name_long for n in room.npcs)}\n"
    )

    # Compose recent events string from history_buffer or "No previous events"
    if history_buffer and len(history_buffer) > 0:
        recent_events_summary = "Recent events:\n" + "\n".join(f"{i+1}. {ev}" for i, ev in enumerate(history_buffer))
    else:
        recent_events_summary = "No previous events."

    prompt = (
        f"You are the world simulation for a text RPG.\n"
        f"{room_desc}\n"
        f"{recent_events_summary}\n"
        f"Player's active goals: {player_goals or 'none'}\n"
        f"NPC details:\n{npc_goal_block}\n"
        f"Here are the plan steps for the player's next action:\n{plan_desc}\n\n"
        "Instructions:\n"
        "- Whenever the player acts toward a goal, narrate and update goals as needed.\n"
        "- Narrate outcomes including dialogue.\n"
        "- Output goal changes under 'goal_updates'\n"
        "- Do NOT invent or pluralize objects/NPCs except as necessary.\n"
        "- Do NOT invent new room names or locations.\n"
        "- Return JSON ONLY:\n"
        "{\n"
        "  \"narration\": <string>,\n"
        "  \"dialogue\": [ { \"speaker\": ..., \"line\": ... } ],\n"
        "  \"npc_updates\": [ { \"npc\": ..., \"updated_goals\": [...], \"score\": ... } ],\n"
        "  \"object_updates\": [ { \"object\": ..., \"state\": ..., \"category\": ... } ],\n"
        "  \"goal_updates\": {\"player\": [...], \"npc_goals\": {npc_name: [...]}}\n"
        "}\n"
    )

    messages = [{"role": "user", "content": prompt}]
    print("[DEBUG][LLM SIM EXECUTE] Prompt:\n", prompt)

    response = await client.chat.completions.create(
        model=model_deployment,
        messages=messages,
        response_model=LLMSimulationOutcome,
        temperature=0.4,
    )

    narration = response.narration or ""
    goal_updates = response.goal_updates or LLMGoalUpdates()
    npc_updates = response.npc_updates or []
    object_updates = response.object_updates or []

    # Append narration to history for future context
    if narration:
        history_buffer.append(narration)

    # Update player goals
    if world_ref and goal_updates.player:
        world_ref.player_state.goals = goal_updates.player
    elif hasattr(room, "player_state") and goal_updates.player:
        room.player_state.goals = goal_updates.player

    # Update NPC goals
    if goal_updates.npc_goals:
        for npc_name, goals in goal_updates.npc_goals.items():
            for npc in room.npcs:
                if canon_id(npc.name_long) == canon_id(npc_name):
                    npc.state.goals = goals

    # Update NPC score and state
    for npc_update in npc_updates:
        updated_name = npc_update.npc
        for npc in room.npcs:
            if canon_id(npc.name_long) == canon_id(updated_name):
                if npc_update.updated_goals:
                    npc.state.goals = npc_update.updated_goals
                if npc_update.score is not None:
                    npc.state.score = npc_update.score

    # Update objects (validate and deduplicate)
    for obj_update in object_updates:
        obj_name = obj_update.object
        if obj_name and not is_invalid_object(room, obj_name) and not object_already_present(room, obj_name):
            room.state.objects.append(obj_name)
            print(f"[DEBUG] Added object ({obj_update.category or 'interactable'}): {obj_name}")

    class DummyOutcome:
        pass

    outcome = DummyOutcome()
    outcome.narration = narration
    save_game_state(save_file, room)
    return outcome, []


# ---- INTENT PARSING: LLM Normalized -----
def make_get_intent_prompt(query, npc_list, object_list):
    npc_str = ", ".join(npc_list) + ", Player"
    obj_str = ", ".join(object_list)
    prompt = f"""
You are an assistant for a text RPG.
Given a player command, classify it into canonical intents: Look, Interact, Talk, Take, Use, Equip, Give.
When player uses a nickname, shortened, or partial name (e.g., "Milo"), map it to the full name from the Known Characters list (e.g., "Milo Gearhart").
Only return characters matched from the list, even if referred to indirectly.
If no clear match, return empty.
Known Characters (NPCs): {npc_str}
Known Objects: {obj_str}

Return output as strict JSON with fields:
- user_query
- classification
- explanation
- characters_mentioned
- characters_matched
- interactables_mentioned
- interactables_matched
- intent_objects (empty list if none)
- competition: true if command involves contest, duel, challenge, race, or other opposition; otherwise false.
- competitors: list of involved NPC or 'Player' names exactly as in the NPC list above (includes 'Player' if player is involved).

Example:
{{
  "user_query": "challenge Axiom and Nova to a race",
  ...
  "competition": true,
  "competitors": ["Axiom Vex", "Captain Nova Kiran", "Player"]
}}

Player Input: {query}
"""
    return [{"role": "system", "content": prompt}]


async def get_intent(query: str, npc_list: List[str], interactables: List[str]) -> QueryAnalysis:
    npc_list_str = ", ".join(npc_list)
    interactables_str = ", ".join(interactables)
    prompt = f"""
You are an assistant for a text RPG game.
Your job:  
- Given a player command, classify it into one or more of these canonical intents: Look, Interact, Talk, Take, Use, Equip, Give.
- Map synonyms to the correct intent (examples below).
Examples:
- "Explore", "investigate", "search", "scout", "survey", "examine" → Look
- "Talk to", "ask", "speak", "greet", "question" → Talk
- "Use", "operate", "activate" → Use
- "Take", "grab", "pick up", "collect", "loot" → Take
- "Interact", "engage with", "manipulate", "press" → Interact
- "Equip", "put on", "wear" → Equip
- "Give", "offer", "hand" → Give
- "go", "go to", "exit", "leave", "walk to", "move", "move to", "travel to", "head to" → Move

Only use the canonical intent names above.  
Identify named NPCs and objects referenced (from the provided lists), even when referenced indirectly (e.g. "the deck"→"Data cache terminal" if that's a listed object).

Return a JSON object:
- user_query
- classification
- explanation- characters_mentioned
- characters_matched
- interactables_mentioned
- interactables_matched
- intent_objects

Player Input: {query}
Known Characters: {npc_list_str}
Known Objects: {interactables_str}
Output JSON ONLY:
    """
    messages = [{"role": "system", "content": prompt}]
    for attempt in range(3):
        try:
            response = await client.chat.completions.create(
                model=model_deployment,
                messages=messages,
                response_model=QueryAnalysis,
            )
            return response
        except Exception as exc:
            logger.warning(f"Intent parse error: {exc}")
        await asyncio.sleep(2)
    raise RuntimeError("Failed to get intent after retries")


def normalize_intents(intents: List[str]) -> List[str]:
    normalized = set()
    for intent in intents:
        lower = intent.lower().strip()
        canonical = INTENT_SYNONYMS.get(lower)
        if canonical:
            normalized.add(canonical)
        elif lower.capitalize() in {"Look", "Talk", "Interact", "Take", "Use", "Equip", "Give"}:
            normalized.add(lower.capitalize())
    return list(normalized)


# ---- PLAN: LLM and EXECUTION -----
async def create_plan_with_llm(player_intent: QueryAnalysis, room: Room, player_action: str) -> Plan:
    messages = build_plan_prompt(player_intent, room, player_action)
    for attempt in range(3):
        try:
            response = await client.chat.completions.create(
                model=model_deployment,
                messages=messages,
                response_model=LLMPlan,
                temperature=0.2,
            )
            llm_plan = response
            actions = [
                Action(intent=step.intent, target=step.target, description=step.description)
                for step in llm_plan.steps
            ]
            # Filtering no longer needed—LLM is already only giving allowed names!
            if not actions:
                raise ValueError("LLM returned no plan steps")
            return Plan(actions)
        except (ValidationError, ValueError) as e:
            logger.error(f"Plan creation validation error: {e}")
        except Exception as e:
            logger.error(f"Plan creation error: {e}")
        await asyncio.sleep(2)
    logger.warning("Falling back to default plan: Look around")
    return Plan([Action("Look", None, "Look around the area.")])




# ---- GAME WORLD -----
''''
This should really be a default json file to load in, but it's here for now.
'''
def initialize_game_world() -> GameWorld:
    chase = NPCParent(
        name_id="chase",
        name_long="Chase Ketchup",
        state=NPCState(
            name_id="chase",
            goals=["Find experimental treatment to treat his fried brain."],
            dialogue_prompt="Speaks with nervous jitters and half-formed sentances. Loses the point.",
            backstory="A burned-out netrunner with a fried nervous system, Chase would do or say anything to get repair altered brain. A hotshot and fast-talker, he bluffed the wrong crime syndicate and ended up unable to connect to the net, even with this the newest decks due to the nano-toxins they pumped into his system as revenge. He is short, with a thin build and a light, patchy beard developed from days without a good shower. He fidgets and stutters, will steal and is not trustworthy, despite his efforts to appear as a morale person. He will only help if he can get paid.",
            inventory=["Broken cyberdeck", "10-credit chip"],
            autonomous_goals=["Find experimental treatment."],
        ),
    )
    froggy = NPCParent(
        name_id="froggy",
        name_long="Froggy Owlbear",
        state=NPCState(
            name_id="froggy",
            goals=["Get a job as an animatronic band leader."],
            dialogue_prompt="Speaks like a TV dinosaur.",
            backstory="Froggy is the mascot to a closed chain of dungeon themed pizzerias. Technically sentient, when the Froggy’s Foggy Pizza went under, hundreds of models just like him were released into the world, protected as sentient life, but without rights or means. This Froggy has had an easier time, taking photos with tourists for tips in major city squares. He’s massive, over two meters tall and over three hundred kilograms. He’s always in a good mood, and always willing to lend a hand, even if he ends up damaged. His robotic outer layers tell the story of battles hard fought in the streets of the sprawl.",
            inventory=["Useless pizza coupons"],
            autonomous_goals=["Find a tourist hotspot."],
        ),
    )
    ginzo = NPCParent(
        name_id="ginzo",
        name_long="Hamna Ginzo",
        state=NPCState(
            name_id="ginzo",
            goals=["Draw a silly picture of an important person."],
            dialogue_prompt="Speaks with cold logic.",
            backstory="Once kept cryogenically frozen for a powerful megacorporation's use, Ryu now operates as a freelance caricature artist. His body is enhanced with cutting-edge cybernetic implants, making him a formidable illustrator. He is exceptionally intense in aura but gentle in nature. Haunted by his past, he seeks redemption by taking on jobs that humiliate the corrupt corporate overlords. He does not want to help with tasks that don't involve revenge against corporations.",
            inventory=["Small paint brush", "Large paint brush", "Canvas"],
            autonomous_goals=["Clean his brushes of old ink."],
        ),
    )
    will = NPCParent(
        name_id="will",
        name_long="Will Egress",
        state=NPCState(
            name_id="will",
            goals=["Find the escape samurai and convince them to return."],
            dialogue_prompt="Speaks with smooth arrogant words that make you feel both impressed and talked-down to.",
            backstory="Will dropped out of college to pursue his dream of developing sentient computer beings. As a pioneer in the field, he’s now massively rich and enigmatic. Only his closest friends and family have seen his face since its total reconstruction 100 years ago. Living in eternal youth due to his ability to pay for new lab-grown bodies, he searches for his greatest achievement, the Hamna line of synthetic humans. He is wearing the body of a beautiful and massive Asian man, but he himself is over three hundred years old and from Detroit before the third American civil war.",
            inventory=["1 million credit chip", "Phaser Gun"],
            autonomous_goals=["Find a decent bite to eat."],
        ),
    )
    lucy = NPCParent(
        name_id="lucy",
        name_long="Lucy Brown",
        state=NPCState(
            name_id="lucy",
            goals=["Buy a prized football back from the pawn shop."],
            dialogue_prompt="Funny and sarcastic, Lucy is also insightful and smart.",
            backstory="Lucy is a computer psychologist, specializing in former corporate assets now protected under the Silicon-Kennedy Act. She has been collecting Froggy Owlbears together and has formed a collective of them in the radioactive portion of old California. She has a short black bob haircut and wears a simple blue jumper.",
            inventory=["Secret map to Froggy Conclave", "Ham Sandwich"],
            autonomous_goals=["Find Froggy Owlbear."],
        ),
    )
    linux = NPCParent(
        name_id="linux",
        name_long="Linux Brown",
        state=NPCState(
            name_id="linux",
            goals=["Steal a prized football back from the pawn shop."],
            dialogue_prompt="Linux speaks with a slur due to constant web detox.",
            backstory="Linux is the bumbling younger brother of Lucy. Jobless, he often pawns family heirlooms for cash and attention. He is short and a bit overweight. He’s balding, but handsome otherwise. He's addicted to the web, spending the bit of cash he can get on bootleg AR experiences and VR trips that often result in slight brain damage.",
            inventory=["Half-eaten Ham Sandwich"],
            autonomous_goals=["Ditch his sister and find a VR cafe."],
        ),
    )

    pawn_owner = NPCParent(
        name_id="pawn_owner",
        name_long="Milo Gearhart",
        state=NPCState(
            name_id="pawn_owner",
            goals=["Acquire rare and valuable items to sell at a high profit."],
            dialogue_prompt="Speaks with a smooth, persuasive tone, always trying to make a deal.",
            backstory="Milo Gearhart is a shrewd and cunning pawn shop owner who has been in the business for over two decades. He started his shop in the heart of the city, where he quickly learned the art of negotiation and the value of rare items. Milo has a knack for finding hidden treasures and is always on the lookout for the next big score. He is tall, with a lean build and a sharp, calculating gaze. Despite his charming demeanor, he is not above using underhanded tactics to get what he wants.",
            inventory=["Rare antique watch", "High-tech lockpicking kit"],
            autonomous_goals=["Expand his pawn shop empire."],
        ),
    )

    security_guard = NPCParent(
        name_id="security_guard",
        name_long="Jaxon Steele",
        state=NPCState(
            name_id="security_guard",
            goals=["Ensure the safety and security of the premises."],
            dialogue_prompt="Speaks with a firm, authoritative tone, always vigilant.",
            backstory="Jaxon Steele is a former military officer who now works as a security guard for various establishments in the city. After serving in the armed forces, he decided to use his skills to protect civilian properties. Jaxon is highly disciplined and takes his job very seriously. He is muscular, with a stern expression and a no-nonsense attitude. His experience in combat and security makes him a formidable presence, and he is always ready to handle any threat that comes his way.",
            inventory=["Stun baton", "Security access card"],
            autonomous_goals=["Maintain order and prevent theft."],
        ),
    )

    panhandler = NPCParent(
        name_id="panhandler",
        name_long="Sammy Scraps",
        state=NPCState(
            name_id="panhandler",
            goals=["Collect enough money to survive another day."],
            dialogue_prompt="Speaks with a desperate, pleading tone, often sharing stories of hardship.",
            backstory="Sammy Scraps is a down-on-his-luck individual who roams the streets, begging for spare change. Once a skilled mechanic, Sammy lost everything due to a series of unfortunate events, including the collapse of his business and the loss of his family. Now, he survives by relying on the kindness of strangers. He is disheveled, with a scruffy beard and worn-out clothes. Despite his circumstances, Sammy remains hopeful and tries to maintain a positive outlook, often sharing his mechanical knowledge with anyone willing to listen.",
            inventory=["Rusty wrench", "Old photograph"],
            autonomous_goals=["Find a safe place to sleep."],
        ),
    )

    ghost = NPCParent(
        name_id="ghost",
        name_long="Evelyn Shade",
        state=NPCState(
            name_id="ghost",
            goals=["Uncover the truth behind her untimely death."],
            dialogue_prompt="Speaks with an eerie, whispering tone, often revealing cryptic clues.",
            backstory="Evelyn Shade was a brilliant scientist who met a mysterious and untimely end. Her spirit now haunts the city, seeking answers and justice for her death. Evelyn's ghostly form is ethereal and translucent, with a faint glow that illuminates her surroundings. She is driven by a deep sense of injustice and a desire to uncover the truth. Evelyn can interact with the physical world to a limited extent, often leaving clues and messages for those who are willing to help her. She is both haunting and captivating, with a presence that evokes both fear and sympathy.",
            inventory=["Ancient locket", "Cryptic journal"],
            autonomous_goals=["Find someone who can help solve her murder."],
        ),
    )
    neon_nexus = Room(
        name_id="neon_nexus",
        name_long="The Neon Nexus",
        npcs=[panhandler],
        state=RoomState(
            description="The Neon Nexus is a bustling digital bazaar located in just another strip mall of the sprawl. The area is bathed in fluttering, broken neon lights, casting a kaleidoscope of colors across the pitted metallic surfaces. Holographic advertisements flicker and change, promoting everything from cybernetic enhancements to exotic alien delicacies. The air is thick with the hum of electronic devices and the distant chatter of merchants and customers haggling over prices. The ground is a mosaic of glowing tiles, leading visitors to various stalls and shops that offer a wide array of goods and services, legal and blackmarket.",
            objects=["Data cache terminal", "Map of the Sprawl"],
            hazards=["Police patrol bots"],
            ambient_noise="Electronic hum and distant chatter.",
        ),
        exits={"north": "gamma_square", "east": "the_spire"},
    )
    pawn_shop = Room(
        name_id="pawn_shop",
        name_long="Quickie Pawn Emporium",
        npcs=[pawn_owner, lucy, linux],
        state=RoomState(
            description="The pawnshop squats in the shadows of the megacity, a relic of forgotten dreams and broken promises. Its neon sign flickers erratically, casting a sickly green glow. Shelves sag under the weight of discarded tech, each piece a testament to someone's shattered ambition. The walls are lined with mismatched monitors displaying a chaotic dance of static. A rusty fan creaks overhead, struggling to circulate the stale air. The counter, a battered slab of metal, is cluttered with an assortment of items: a tarnished cyberdeck, a pile of outdated credit chips, and a collection of bizarre trinkets that defy categorization.",
            objects=["cyberdeck", "football", "gold chain"],
            hazards=["Security drones", "Booby-trapped items"],
            ambient_noise="Soft hum of machinery and occasional beeps from scanning devices.",
        ),
        exits={"east": "gamma_square", "north": "cyber_alley"}, 
    )
    gamma_square = Room(
        name_id="gamma_square",
        name_long="Gamma Square",
        npcs=[froggy],
        state=RoomState(
            description="The strip mall stands like a monument to the city's decay. Its once-bright facade is now a canvas of peeling paint and graffiti, each tag a testament to the passage of time. The parking lot is a wasteland of cracked asphalt and weeds, with faded lines barely visible under layers of grime. In the corner, a lone vending machine stands as a sentinel, its lights flickering sporadically. The shelves are empty, save for a few expired snacks and drinks, a sad reminder of the mall's former life.",
            objects=["Old terminal", "Rusty supply crate"],
            hazards=["Unstable power grid"],
            ambient_noise="Buzz of failing electronics.",
        ),
        exits={"south": "neon_nexus", "up": "shadowed_underpass", "north": "cyber_alley", "west":"pawn_shop", "east":"noodle_shop"},
    )
    noodle_shop = Room(
        name_id="noodle_shop",
        name_long="Starlight Noodles",
        npcs=[ginzo],
        state=RoomState(
            description="Starlight Noodles is the kind of place where you can get a bowl of noodles and a side of robot grease. The automated noodle dispensers have a mind of their own, sometimes serving noodles, sometimes serving questions about the bleak meaning of their autonomous noodle filled lives. The holographic menu flickers between delicious dishes and random cat videos, leaving customers both hungry and confused. The robotic waitstaff are programmed with little knowledge of human customs, and often insult the customers while attempting to compliment them. The walls are covered in massive manga panels, depicting a theme of Gundams crushing New York City, a controversial subject among humans and robots given the recent past.",
            objects=["Automated noodle dispensers", "Holographic menu", "Robotic waitstaff"],
            hazards=["Grease spills", "Malfunctioning robots"],
            ambient_noise="Sizzling sounds from the kitchen and soft chatter of patrons.",
        ),
        exits={"west": "cyber_alley", "south":"the_spire", "north":"arcade"}, 
    )
    cyber_alley = Room(
        name_id="cyber_alley",
        name_long="Cyber Alley",
        npcs=[ghost],
        state=RoomState(
            description="A narrow, dimly-lit alleyway that winds through the heart of the megacity's underbelly. The walls are covered in layers of paint, some of it glowing with neon. The ground is slick with rain and oil, reflecting the flickering lights of malfunctioning street lamps. The air is thick with the smell of garbage and the distant hum of machinery.",
            objects=["Discarded cybernetic parts", "Hacked vending machine", "Broken surveillance camera"],
            hazards=["Rogue drones", "Hidden tripwires", "Toxic waste puddles"],
            ambient_noise="Distant sirens, the buzz of neon signs, and the occasional scurry of rats.",
        ),
        exits={"south": "gamma_square", "east": "arcade"},
    )
    the_spire = Room(
        name_id="the_spire",
        name_long="The Spire",
        npcs=[will],
        state=RoomState(
            description="A glass-walled skyscraper that pierces the clouds.",
            objects=["Encrypted archives"],
            hazards=["Security drones"],
            ambient_noise="Distant whir of servos.",
        ),
        exits={"west": "neon_nexus", "north": "noodle_shop"}
    )
    shadowed_underpass = Room(
        name_id="shadowed_underpass",
        name_long="Shadowed Underpass",
        npcs=[],
        state=RoomState(
            description="A dark, damp passage echoing with distant sounds.",
            objects=["Oil barrel", "Abandoned backpack"],
            hazards=["Rat infestation"],
            ambient_noise="Dripping water, scurrying rats.",
        ),
        exits={"down": "gamma_square"}
    )
    arcade = Room(
        name_id="arcade",
        name_long="Pixel Paradise",
        npcs=[security_guard, chase],
        state=RoomState(
            description="Pixel Paradise is where gamers go to lose their dignity. The VR gaming pods are so immersive that players often forget they're still in the real world, leading to hilarious moments of people trying to engage inappropriately with holograms. The classic arcade machines are haunted by the ghosts of high scores past, and the prize counter offers rewards like 'World's Best Button Masher' trophies and 'Free Hug from the Security Guard' coupons. The atmosphere is electric, filled with the excited shouts of gamers and the occasional sound of someone rage-quitting.",
            objects=["VR gaming pods", "Classic arcade machines", "Prize counter"],
            hazards=["Overloaded circuits", "Rowdy gamers"],
            ambient_noise="Excited shouts and electronic game sounds.",
        ),
        exits={"south": "noodle_shop", "west": "cyber_alley"}, 
    )
    player_state = PlayerState(
        current_room_id="neon_nexus",
        inventory=[],
    )
    world = GameWorld(
        player_state=player_state,
        rooms={
            neon_nexus.name_id: neon_nexus,
            gamma_square.name_id: gamma_square,
            the_spire.name_id: the_spire,
            shadowed_underpass.name_id: shadowed_underpass,
            cyber_alley.name_id: cyber_alley,
            arcade.name_id: arcade,
            noodle_shop.name_id: noodle_shop,
            pawn_shop.name_id: pawn_shop,
        }
    )
    return world

def show_exits(room: Room, world: GameWorld) -> str:
    if not room.exits:
        return "There are no visible exits."
    directions = []
    for direction, dest_room_id in room.exits.items():
        dest_room = world.rooms.get(dest_room_id)
        if dest_room:
            directions.append(f"{direction.title()}: {dest_room.name_long}")
        else:
            directions.append(f"{direction.title()}: (unknown)")
    return "Exits:\n" + "\n".join(directions)

def look_direction(room: Room, direction: str, world: GameWorld) -> str:
    dest_room_id = room.exits.get(direction.lower())
    if not dest_room_id or dest_room_id not in world.rooms:
        return f"You see nothing of interest to the {direction}."
    dest_room = world.rooms[dest_room_id]
    return f"To the {direction}, you notice: {dest_room.state.description}"

def object_already_present(room, obj_name):
    canon = canon_id(obj_name)
    for o in room.state.objects:
        o_canon = canon_id(o)
        if canon == o_canon or canon in o_canon or o_canon in canon:
            return True
    return False

def is_invalid_object(room, obj_name):
    c_obj = canon_id(obj_name)
    if c_obj == canon_id(room.name_long):
        return True
    for npc in room.npcs:
        if c_obj == canon_id(npc.name_long) or c_obj == canon_id(npc.name_id):
            return True
    return False

def npc_already_present(room, npc):
    """
    Check if an NPC with a similar canonical name ("iron_golem", "golem") already exists.
    Also look for subset/superset relationships (e.g. if either is contained in the other).
    """
    new_canon = canon_id(getattr(npc, "name_long", getattr(npc, "name_id", "")))
    for existing in room.npcs:
        existing_canon = canon_id(getattr(existing, "name_long", getattr(existing, "name_id", "")))
        # Direct match or substring match (order-insensitive)
        if new_canon == existing_canon:
            return True
        if new_canon in existing_canon or existing_canon in new_canon:
            return True
    return False


async def gameplay_loop(world: GameWorld, save_file: str = "save.json"):
    print("Welcome to the AI-powered terminal RPG!")
    print("Type commands or 'exit' to quit.\n")
    current_room_id = world.player_state.current_room_id
    current_room = world.rooms.get(current_room_id)
    if current_room:
        print(f"\n=== {current_room.name_long} ===\n{current_room.state.description}")
        print(show_exits(current_room, world))
    while True:
        player_query = input(f"[{world.player_state.current_room_id}] > ").strip().lower()
        if player_query in {"exit", "quit"}:
            print("Exiting game. Saving state...")
            save_game_state(save_file, world)
            print("Game saved. Bye!")
            break
        current_room = world.rooms.get(world.player_state.current_room_id)
        # -------- Directional and parser navigation --------
        if player_query in {"exits", "show exits", "where can i go"}:
            print(show_exits(current_room, world))
            continue
        moved = False
        for dir in DIRECTIONS:
            target_patterns = [f"go {dir}", f"move {dir}", f"walk {dir}", f"run {dir}"]
            if player_query == dir or any(player_query.startswith(pat) for pat in target_patterns):
                dest_room_id = current_room.exits.get(dir)
                if dest_room_id and dest_room_id in world.rooms:
                    world.player_state.current_room_id = dest_room_id
                    current_room = world.rooms[dest_room_id]
                    print(f"\nYou move {dir}.\n")
                    print(f"=== {current_room.name_long} ===")
                    print(current_room.state.description)
                    print(show_exits(current_room, world))
                    save_game_state(save_file, world)
                else:
                    print(f"You can't go {dir} from here.")
                moved = True
                break
        if moved:
            continue
        looked = False
        for dir in DIRECTIONS:
            if player_query == f"look {dir}":
                print(look_direction(current_room, dir, world))
                looked = True
                break
        if looked:
            continue
        # Otherwise, LLM-based intent/planning fallback
        npc_names = [npc.name_long for npc in current_room.npcs]
        object_names = list(current_room.state.objects)
        try:
            messages = make_get_intent_prompt(player_query, npc_names, object_names)
            player_intent = await client.chat.completions.create(
                model=model_deployment,
                messages=messages,
                response_model=QueryAnalysis,
            )
        except Exception as e:
            print(f"Intent parsing error: {e}")
            continue
        player_intent.classification = normalize_intents(player_intent.classification)
        if "Move" in player_intent.classification:
            print("Use classic parser commands (e.g., 'go north', 'move west') to travel between rooms.")
            print(show_exits(current_room, world))
            continue
        # ---- PLAN GENERATION ----
        plan = await create_plan_with_llm(player_intent, current_room, player_query)
        print(f"Plan steps generated:")
        for a in plan.actions:
            print(f"- {a.intent} on {a.target} : {a.description}")
        # ---- Plan EXECUTION: One LLM-driven step!
        outcome, _ = await process_plan_actions(plan, current_room, save_file=save_file, player_query=player_query)
        # ---- Narration/feedback ----
        print(f"\n[Plan Summary]\n{getattr(outcome, 'narration', '')}\n")
        save_game_state(save_file, world)
        print(f"You are at {current_room.name_long}: {current_room.state.description}")
        print(show_exits(current_room, world))
        print(f"NPCs here: {', '.join(n.name_long for n in current_room.npcs) if current_room.npcs else 'None'}")


def canon_id(s):
    s = (s or '').lower().strip()
    s = re.sub(r'[^\w\s]', '', s)
    s = re.sub(r'\s+', '_', s)
    return s

async def competitive_plan_resolution(plan, room, player_query, save_file, competitors=None, player_name="Player"):
    # 1. Identify participants (NPCs + player)
    relevant_npcs = []
    include_player = False
    if competitors:
        for name in competitors:
            if name.lower() == player_name.lower():
                include_player = True
                continue
            name_canon = canon_id(name)
            for npc in room.npcs:
                if name_canon == canon_id(npc.name_long) or name_canon == canon_id(npc.name_id):
                    if npc not in relevant_npcs:
                        relevant_npcs.append(npc)
    else:
        for action in plan.actions:
            for npc in room.npcs:
                if canon_id(action.target) == canon_id(npc.name_long) or canon_id(action.target) == canon_id(npc.name_id):
                    if npc not in relevant_npcs:
                        relevant_npcs.append(npc)
        if not relevant_npcs:
            relevant_npcs = list(room.npcs)

    # 2. Wrap player as agent so we can treat as NPC
    if include_player:
        # You may want to provide inventory/goals for the player here if relevant
        player_state = room.world.player_state if hasattr(room, "world") else PlayerState(current_room_id="", inventory=[], goals=[])
        player_agent = PlayerAgent(player_state)
        relevant_npcs.append(player_agent)

    # 3. Prompt for in-character, direct, actionable move_text
    npc_intents = []
    for npc in relevant_npcs:
        npc_name_long = npc.name_long
        backstory = getattr(npc.state, "backstory", "No backstory available.") if hasattr(npc, "state") else "Player character."
        dialogue_prompt = getattr(npc.state, "dialogue_prompt", "N/A") if hasattr(npc, "state") else "Player’s voice."
        goals = getattr(npc.state, "goals", []) if hasattr(npc, "state") else []
        strategy_prompt = (
            f"You are {npc_name_long}, known for: {backstory}\n"
            f"Your speaking style: {dialogue_prompt}\n"
            f"Current goals: {', '.join(goals) if goals else 'none'}.\n"
            "THIS IS A CONTEST or COMPETITION in the scene. "
            f"Contest: '{player_query}'.\n"
            "- As this character, write only what you would actually say (as taunt, taunting move, diss rap verse, attack, challenge, or boast) or do as your first move. Use your own style. Do not summarize, do not say 'The player...'."
            "- Provide ONLY a JSON with the key 'move_text': <your opening move/line>."
        )
        result = await client.chat.completions.create(
            model=model_deployment,
            messages=[
                {"role": "system", "content": "You are an RPG contestant NPC (or player)."},
                {"role": "user", "content": strategy_prompt}
            ],
            response_model=CompQueryAnalysis,
            temperature=0.3,
        )
        print("[DEBUG][COMPETITIVE] API Response:", result)  
        
        move_text = getattr(result, "move_text", "No move text provided.")  
        explanation = getattr(result, "explanation", "No explanation provided.")  
        strategy = getattr(result, "strategy", "No strategy provided.")  
    
        npc_intents.append({  
            "npc": npc_name_long,  
            "move_text": move_text,
            "strategy": strategy,
            "explanation": explanation
        })  
    
        print(f"[DEBUG][COMPETITIVE] {npc_name_long} move: {move_text}, strategy: {strategy}, explanation: {explanation}")  

    # 4. GM prompt with direct agent actions
    npc_lines = "\n".join(f"{ni['npc']}: move: {ni['move_text']} and strategy: {ni['strategy']}" for ni in npc_intents)
    gm_prompt = (
        f"This is a competitive event in a text RPG.\n"
        f"Room: {room.name_long}\nDescription: {room.state.description}\n"
        f"Each participant's actual first move and their strategy :\n{npc_lines}\n\n"
        f""
        "Instructions:\n"
        "- Narrate the outcome vividly and use the actual lines/attacks provided as dialogue and action in the narration."
        "Return a score of -3 to 3 based on each NPC performance vs the group."
        "Update NPC goals to be more aggressive and varied when the NPC scores are under zero."
        "Do not return the score to the narrarion, only the basemodel."
        "Do not create Player NPCs"
    )
    print("[DEBUG][COMPETITIVE] GM prompt:\n", gm_prompt)

    gm_result = await client.chat.completions.create(
        model=model_deployment,
        messages=[{"role": "user", "content": gm_prompt}],
        response_model=LLMContestOutcome,
        temperature=0.2,
        max_tokens=1200
    )
    resp = gm_result
    print("[DEBUG][COMPETITIVE] API Response:", resp)
    print("[DEBUG][GOAL_UPDATES] API Response goal updates:", resp.goal_updates)    

    # --- State updates as before ---
    world_ref = getattr(room, "world", None)
# Update player goals if available in the response  
    if resp.goal_updates:  
        # Update player goals if they exist  
        if resp.goal_updates.player:  
            if world_ref:  
                print("[DEBUG] Updating player goals...", resp.goal_updates.player)  # Debug statement  
                world_ref.player_state.goals = resp.goal_updates.player  
            elif hasattr(room, "player_state"):  
                print("[DEBUG] Updating player goals...", resp.goal_updates.player)  # Debug statement  
                room.player_state.goals = resp.goal_updates.player  
    
        # Update NPC goals if they exist  
        if resp.goal_updates.npc_goals:  
            print("[DEBUG] NPC Goal Updates Detected:", resp.goal_updates.npc_goals)  # Debug statement  
            for npc_name, goals in resp.goal_updates.npc_goals.items():  
                for npc in room.npcs:  
                    print(f"[DEBUG] Comparing {canon_id(npc.name_long)} to {canon_id(npc_name)}")  # Debug statement  
                    if canon_id(npc.name_long) == canon_id(npc_name):  
                        print(f"[DEBUG] Updating goals for {npc_name} to: {goals}")  # Debug statement  
                        npc.state.goals = goals  # Update NPC goals  
                        break  # Exit the inner loop once the NPC is found  
    
    # Handle updates for NPCs  
    if resp.npc_updates:  
        print("[DEBUG] NPC Updates Detected:", resp.npc_updates)  # Debug statement  
        for npc_update in resp.npc_updates:  
            updated_name = npc_update.npc  
            for npc in room.npcs:  
                print(f"[DEBUG] Comparing {canon_id(npc.name_long)} to {canon_id(updated_name)}")  # Debug statement  
                if canon_id(npc.name_long) == canon_id(updated_name):  
                    if npc_update.updated_goals:  
                        print(f"[DEBUG] Updating goals for {updated_name} to: {npc_update.updated_goals}")  # Debug statement  
                        npc.state.goals = npc_update.updated_goals  
                    if npc_update.score is not None:  
                        print(f"[DEBUG] Updating score for {updated_name} to: {npc_update.score}")  # Debug statement  
                        npc.state.score = npc_update.score  
                    break  # Exit loop once the NPC is found  
    
    # Handle object updates  
    if resp.object_updates:  
        print("[DEBUG] Object Updates Detected:", resp.object_updates)  # Debug statement  
        for obj_update in resp.object_updates:  
            obj_name = obj_update.object  
            if obj_name and not is_invalid_object(room, obj_name) and not object_already_present(room, obj_name):  
                print(f"[DEBUG] Adding object: {obj_name}")  # Debug statement  
                room.state.objects.append(obj_name)  
    
    # Create a DummyOutcome to manage the response narration  
    class DummyOutcome: pass  
    outcome = DummyOutcome()  
    outcome.narration = resp.narration  
    
    # Save the game state  
    save_game_state(save_file, room)  
    
    # Return the outcome and relevant NPCs  
    return outcome, relevant_npcs  


# ---- MAIN -----
if __name__ == "__main__":
    import sys
    save_path = sys.argv[1] if len(sys.argv) > 1 else "save.json"
    try:
        world_instance = load_game_state(save_path)
        print(f"Game loaded from {save_path}")
    except FileNotFoundError:
        world_instance = initialize_game_world()
        print("New game world initialized.")
    
    # Generate and show DALL·E image for starting room
    async def startup_image():
        current_room_id = world_instance.player_state.current_room_id
        current_room = world_instance.rooms.get(current_room_id)
        if current_room:
            image_url = await generate_dalle_image_for_scene(current_room)
            print(f"Scene image at startup: {image_url}")

    asyncio.run(startup_image())
    
    # Now run your main game loop
    asyncio.run(gameplay_loop(world_instance, save_path))