from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from contextlib import asynccontextmanager
from game_loop import *
import sys
from collections import deque
import re

save_file = "save.json"
history_buffer = deque(maxlen=10)

html = """
<!DOCTYPE html>
<html>
<head>
  <title>Lunacraft: Forge Terminal</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    html, body {
      margin: 0; padding: 0; height: 100%; width: 100vw; box-sizing: border-box;
      background: #111;
      color: #eee;
      font-family: monospace;
    }
    body {
      /* Flex to center container vertically & horizontally */
      min-height: 100vh;
      display: flex;
      justify-content: center;
      align-items: center;
    }
    #maincontainer {
      width: 80vw; max-width: 1120px; min-width: 340px;
      height: 90vh; max-height: 720px; min-height: 400px;
      background: #16161a;
      box-shadow: 0 4px 24px #0008; border-radius: 10px;
      display: flex;
      flex-direction: column;
      justify-content: flex-start;
      align-items: center;
      overflow: hidden;
      position: relative;
    }
    #title {
      width: 100%;
      text-align: center;
      margin: 1.2em 0 0.7em 0;
      font-size: 1.5em;
      letter-spacing: 0.04em;
      font-weight: bold;
    }
    #mainrow {
      width: 97%;
      height: 80%;
      flex: 1;
      margin: 0 auto;
      display: flex;
      flex-direction: row;
      gap: 0.5em;
    }
    #wrapper {
      flex:3;
      min-width: 0; /* ensures flex children shrink if needed */
      display: flex;
      flex-direction: column;
      height: 100%;
    }
    #log {
      flex: 1 1 0;
      background: #090d12; color: #0f0;
      border: 1.5px solid #222; 
      border-radius: 5px;
      font-size: 1em;
      overflow-y: auto; min-height:100px;
      padding: 12px; margin-bottom: 1.2em;
      word-break: break-word;
      height: 0; /* Ensure flex-grow only */
    }
    #inputrow {
      display: flex; width: 100%;
      font-family: monospace;
      padding-bottom: 1em;
    }
    #input {
      flex: 1; font-size: 1em; padding: 0.5em;
      border: none; outline:none; border-radius: 4px 0 0 4px;
      background: #222; color: #fff;
      font-family: monospace; /* Ensure monospace font */
      padding-bottom: 0.5em; /* Added padding */
    }
    #sendBtn {
      font-size:1em; padding:0.5em 1em; border:none;
      font-family: monospace;
      border-radius:0 4px 4px 0; background:#49f; color: #fff; cursor:pointer;
      padding: 0.5em 1em; /* Added padding */
      font-family: monospace; /* Ensure monospace font */
      background: #c0c0c0; /* Vintage Mac OS color */
      border: 2px solid #808080; /* Vintage border */
      box-shadow: inset -2px -2px 0 #fff, inset 2px 2px 0 #000; /* 3D effect */
    }
    #sendBtn:hover { 
      background: #a0a0a0; /* Slightly darker on hover */
    }
    #thinking { color: #ff0; font-style: italic; margin-left: 10px; align-self: center;}

    #sideimg {
      flex: 1; display: flex; flex-direction: column; align-items: center; 
      height: 100%;
      min-width: 120px; max-width:340px;
      justify-content: center; position: relative; overflow: hidden;
    }
    #sceneimgbg {
      width: 100%; height: 100%;
      background: #222 center center/cover no-repeat;
      border-radius: 10px;
      margin-bottom: 0.6em;
      box-shadow: 0 2px 12px #0009;
      aspect-ratio: 3/4;
      position: relative;
      overflow: hidden;
      display: flex; 
      align-items: center; justify-content: center;
    }
    #imgbtn {
      position: absolute; bottom: 12px; left: 50%; transform: translateX(-50%);
      font-size: 1em; padding: 0.4em 1em;
      background: #191d35; color:#fff; border-radius:8px; border:1px solid #333;
      opacity: 0.81;
      cursor: pointer;
      font-family: monospace; /* Ensure monospace font */
      background: #c0c0c0; /* Vintage Mac OS color */
      border: 2px solid #808080; /* Vintage border */
      box-shadow: inset -2px -2px 0 #fff, inset 2px 2px 0 #000; /* 3D effect */
    }
    #imgbtn:hover { 
      opacity:1; 
      background: #a0a0a0; /* Slightly darker on hover */
    }
    @media (max-width:900px) {
      #maincontainer { width: 99vw; }
      #mainrow { flex-direction: column; gap: 0; height: 85%; }
      #wrapper { min-width:0; }
      #sideimg { width: 100%; max-width: none; min-width: 0; height: 260px; }
      #sceneimgbg { max-width: 260px; aspect-ratio: 1/1; margin: 0 auto;}
      #imgbtn { position: static; margin-top: 0.8em; transform: none;}
    }


  </style>
</head>
<body>
  <div id="maincontainer">
    <div id="title">Lunacraft: Forge Terminal</div>
    <div id="mainrow">
      <div id="wrapper">
        <div id="log"></div>
        <div id="inputrow">
          <input id="input" type="text" autocomplete="off" placeholder="Type command..." />
          <button id="sendBtn">Send</button>
          <span id="thinking" style="display:none;"><br>Thinking...</span>
        </div>
      </div>
      <div id="sideimg">
        <div id="sceneimgbg"></div>
        <button id="imgbtn">Assemble Visual</button>
      </div>
    </div>
  </div>
  <script>
    const ws = new WebSocket(`ws://${location.host}/ws`);
    const log = document.getElementById('log');
    const input = document.getElementById('input');
    const btn = document.getElementById('sendBtn');
    const thinking = document.getElementById('thinking');
    const imgbtn = document.getElementById('imgbtn');
    const sceneimgbg = document.getElementById('sceneimgbg');

    ws.onmessage = function(event) {
      thinking.style.display = 'none';
      let data = event.data;
      if (data.startsWith("SCENE_IMAGE_URL:")) {
        let imageUrl = data.replace("SCENE_IMAGE_URL:", "");
        sceneimgbg.style.backgroundImage = `url('${imageUrl}')`;
      } else {
        log.innerHTML += data.replace(/\\n/g, "<br>");
        log.scrollTop = log.scrollHeight;
      }
    };
    ws.onopen = () => { log.innerHTML += "<b>Connected to server.</b><br>"; };
    ws.onclose = () => { log.innerHTML += "<b>Disconnected.</b><br>"; };

    function send() {
      if (input.value.trim()) {
        thinking.style.display = 'inline';
        ws.send(input.value.trim());
        input.value = '';
      }
    }
    btn.onclick = send;
    input.addEventListener('keyup', function(event) { if(event.key === 'Enter') send(); });
    imgbtn.onclick = function() {
      thinking.style.display = 'inline';
      ws.send("__IMAGE__");
    };
  </script>
</body>
</html>
"""

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Game world initialization at startup
    try:
        world_instance = load_game_state(save_file)
        print(f"Game loaded from {save_file}")
    except FileNotFoundError:
        world_instance = initialize_game_world()
        print("New game world initialized.")

    app.state.world_instance = world_instance
    app.state.save_file = save_file

    # Optional: Do your backend-side room description/image generation here too!
    current_room_id = world_instance.player_state.current_room_id
    current_room = world_instance.rooms.get(current_room_id)
    if current_room:
        print(f"Room description at startup:\n{current_room.state.description}\n")
        try:
            image_url = await generate_dalle_image_for_scene(current_room)
            print(f"Scene image at startup: {image_url}\n")
        except Exception as e:
            print(f"Scene image generation failed at startup: {e}", file=sys.stderr)

    yield

    # (Any shutdown/cleanup can go here if needed)

app = FastAPI(lifespan=lifespan)

@app.get("/")
async def get():
    return HTMLResponse(html)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Game world initialization at startup
    try:
        world_instance = load_game_state(save_file)
        print(f"Game loaded from {save_file}")
    except FileNotFoundError:
        world_instance = initialize_game_world()
        print("New game world initialized.")

    app.state.world_instance = world_instance
    app.state.save_file = save_file

    # No image generation here!
    current_room_id = world_instance.player_state.current_room_id
    current_room = world_instance.rooms.get(current_room_id)
    if current_room:
        print(f"Room description at startup:\n{current_room.state.description}\n")
        # (Removed: Don't generate image here!)

    yield

app = FastAPI(lifespan=lifespan)

@app.get("/")
async def get():
    return HTMLResponse(html)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    history_buffer = deque(maxlen=10)

    await websocket.send_text("<b>Please enter your username:</b><br>")
    username = await websocket.receive_text()
    save_file = f"{username}.json"

    try:
        world_instance = load_game_state(save_file)
        await websocket.send_text(f"Game loaded for user '{username}'.<br>")
    except FileNotFoundError:
        world_instance = initialize_game_world()
        await websocket.send_text(f"New game started for user '{username}'.<br>")

    current_room_id = world_instance.player_state.current_room_id
    current_room = world_instance.rooms.get(current_room_id)
    if not current_room:
        await websocket.send_text("Error: Could not locate current room.\n")
        await websocket.close()
        return

    # Send room description *only* if the player is entering this room for the first time
    if world_instance.player_state.last_room_id != current_room_id:
        await websocket.send_text(f"=== {current_room.name_long} ===\n{current_room.state.description}\n")
        await websocket.send_text(f"Exits:\n{', '.join(current_room.exits.keys())}\n")
        world_instance.player_state.last_room_id = current_room_id
        save_game_state(save_file, world_instance)

    await websocket.send_text("<b>Welcome to Lunacraft: Forge! Type your commands below.</b><br>")
    
    npcs = ", ".join(npc.name_long for npc in current_room.npcs) or "None"
    exits = ", ".join(current_room.exits.keys()) or "None"
    await websocket.send_text(f"People here: {npcs}<br>")
    await websocket.send_text(f"Exits available: {exits}<br>")

    try:
        while True:
            data = await websocket.receive_text()

            if data.strip() == "__IMAGE__":
                await websocket.send_text("<br>Thinking...<br>")
                room = world_instance.rooms[world_instance.player_state.current_room_id]
                try:
                    image_url = await generate_dalle_image_for_scene(room, history_buffer)
                    await websocket.send_text(f"SCENE_IMAGE_URL:{image_url}")
                except Exception as e:
                    await websocket.send_text(f"Image generation failed: {e}")
                continue

            await websocket.send_text("<br>Thinking...<br>")

            try:
                result = await process_command(data, world_instance, save_file, history_buffer)

                # After processing command, check room change and send description as needed
                new_room_id = world_instance.player_state.current_room_id
                if new_room_id != world_instance.player_state.last_room_id:
                    new_room = world_instance.rooms.get(new_room_id)
                    if new_room:
                        await websocket.send_text(f"\n=== {new_room.name_long} ===\n{new_room.state.description}\n")
                        await websocket.send_text(f"Exits:\n{', '.join(new_room.exits.keys())}\n")
                    world_instance.player_state.last_room_id = new_room_id
                    save_game_state(save_file, world_instance)

            except Exception as e:
                result = f"<span style='color:red'>Error: {e}</span><br>"

            await websocket.send_text(result)
    except WebSocketDisconnect:
        print("WebSocket disconnected")

# ----------------------
# Refactor process_command to accept world_instance and save_file:

async def process_command(cmd: str, world_instance, save_file, history_buffer) -> str:
    """
    Process a player command:
    - Quickly handle movement, looking directions, and showing exits with minimal overhead.
    - Otherwise, use LLM intent parsing, plan creation, competitive or normal plan execution.
    """

    player_query = cmd.strip()
    current_room = world_instance.rooms.get(world_instance.player_state.current_room_id)
    if current_room is None:
        return "Error: Current room not found."

    # Optimize movement and look commands:
    normalized_query = player_query.lower()

    # Define valid directions set once
    directions = {"north", "south", "east", "west", "up", "down"}

    # Movement commands: either single direction or "go north", "move east", etc.
    for dir in DIRECTIONS:
        target_patterns = [f"go {dir}", f"move {dir}", f"walk {dir}", f"run {dir}"]
        if normalized_query == dir or any(normalized_query.startswith(pat) for pat in target_patterns):
            dest_room_id = current_room.exits.get(dir)
            if dest_room_id and dest_room_id in world_instance.rooms:
                # Only update last_room_id if movement is occurring
                if dest_room_id != world_instance.player_state.current_room_id:
                    world_instance.player_state.last_room_id = world_instance.player_state.current_room_id
                world_instance.player_state.current_room_id = dest_room_id
                new_room = world_instance.rooms[dest_room_id]
                save_game_state(save_file, world_instance)
                # Return room description on entry
                return (
                    f"\nYou move {dir}.\n"
                    f"=== {new_room.name_long} ===\n"
                    f"{new_room.state.description}\n"
                    f"{show_exits(new_room, world_instance)}"
                )
            else:
                return f"You can't go {dir} from here."

    # Look related simple commands
    if normalized_query in {"exits", "show exits", "where can i go"}:
        return show_exits(current_room, world_instance)

    for dir in directions:
        if normalized_query == f"look {dir}":
            return look_direction(current_room, dir, world_instance)

    if normalized_query == "look":
        return show_exits(current_room, world_instance)

    # If none of the above matched, proceed with LLM-based intent parsing

    # Prepare NPC and object names in full (for better matching)
    npc_names = [npc.name_long for npc in current_room.npcs]
    object_names = list(current_room.state.objects)

    # Get player intent from LLM
    try:
        messages = make_get_intent_prompt(player_query, npc_names, object_names)
        player_intent = await client.chat.completions.create(
            model=model_deployment,
            messages=messages,
            response_model=QueryAnalysis,
        )
    except Exception as e:
        return f"Intent parsing error: {e}"

    # Normalize intents per your synonym mapping
    player_intent.classification = normalize_intents(player_intent.classification)

    # Validate matched characters and objects canonically
    def canon_id(s):
        return re.sub(r'\s+', '_', re.sub(r'[^\w\s]', '', (s or '').lower().strip()))

    canon_npcs = {canon_id(npc.name_long): npc.name_long for npc in current_room.npcs}
    canon_objs = {canon_id(obj): obj for obj in current_room.state.objects}

    valid_chars = [canon_npcs.get(canon_id(c), c) for c in player_intent.characters_matched if canon_id(c) in canon_npcs]
    invalid_chars = [c for c in player_intent.characters_matched if canon_id(c) not in canon_npcs]
    valid_objs = [canon_objs.get(canon_id(o), o) for o in player_intent.interactables_matched if canon_id(o) in canon_objs]
    invalid_objs = [o for o in player_intent.interactables_matched if canon_id(o) not in canon_objs]

    # Build response for unknown references
    response = ""
    if invalid_chars or invalid_objs:
        response += "Unknown references detected:\n"
        for c in invalid_chars:
            response += f" - Unknown character: {c}\n"
        for o in invalid_objs:
            response += f" - Unknown object: {o}\n"

    # Update matched lists
    player_intent.characters_matched = valid_chars
    player_intent.interactables_matched = valid_objs

    # If Move intent still present, refer back to classic parser
    if "Move" in player_intent.classification:
        response += "Use classic parser commands (e.g., 'go north', 'move west') to travel between rooms.\n"
        response += show_exits(current_room, world_instance)
        return response

    # Create plan from LLM
    plan = await create_plan_with_llm(player_intent, current_room, player_query)
    if not plan.actions:
        if response:
            return response + "No plan generated for your command."
        return "No plan generated for your command."

    # Check for competition and route accordingly
    if getattr(player_intent, "competition", False):
        outcome, reacting_npcs = await competitive_plan_resolution(
            plan, current_room, player_query, save_file, competitors=getattr(player_intent, "competitors", None)
        )
    else:
        outcome, reacting_npcs = await process_plan_actions(
            plan, current_room, player_query, save_file, history_buffer
        )

    # Build player response
    for action in plan.actions:
        response += f"Executing: {action.intent} {action.target or ''}\n"
    if outcome and getattr(outcome, "narration", None):
        response += "\n[Plan Summary]\n" + str(outcome.narration) + "\n\n"
    if reacting_npcs:
        for npc in reacting_npcs:
            npc_goals = getattr(npc.state, 'autonomous_goals', []) or getattr(npc.state, 'goals', [])
            response += f"Updated goals for {npc.name_long}: {', '.join(npc_goals)}\n"

    # Handle any new NPCs or objects from the narration
    if outcome and getattr(outcome, "narration", None):
        new_npcs = await extract_npcs_from_narration(outcome.narration)
        for npc in new_npcs:
            if not npc_already_present(current_room, npc):
                current_room.npcs.append(npc)
                response += f"NPC added: {npc.name_long} [{npc.name_id}]\n"
        new_objs = await extract_interactables_from_narration(outcome.narration)
        for obj in new_objs:
            name = getattr(obj, 'name_long', str(obj))
            if not object_already_present(current_room, name) and not is_invalid_object(current_room, name):
                current_room.state.objects.append(name)
                response += f"Object added: {name}\n"

    # Save game state
    save_game_state(save_file, world_instance)

    # Final room and state summary
    response += f"\nYou are at {current_room.name_long}:\n"
    response += show_exits(current_room, world_instance)
    response += f"\nNPCs here: {', '.join(n.name_long for n in current_room.npcs) if current_room.npcs else 'None'}\n"
    response += f"Objects here: {', '.join(current_room.state.objects) if current_room.state.objects else 'None'}"

    return response