# Lunacraft: Forge

Lunacraft: Forge is an AI-powered sci-fi text RPG engine leveraging Azure OpenAI's GPT models. Designed for immersive and dynamic storytelling, it features autonomous NPC agents with evolving goals and rich backstories, multi-agent planning, and procedurally generated narratives powered by advanced language models.

## Project Goals

Traditional text RPGs often rely on rigid scripting or manual content creation, limiting player agency and adaptability. Lunacraft: Forge revolutionizes this space by:

- Utilizing real-time LLM intent parsing and plan generation to drive player and NPC actions.
- Maintaining persistent per-user game state for rich, personalized experiences.
- Supporting emergent multi-NPC interactions, including competitions and alliances.
- Integrating on-demand AI-generated scene illustrations via Azure DALL·E.
- Safeguarding secrets with Azure Key Vault and using secure environment-based credentials.

## Key Features

- Agentic NPCs: Each NPC has unique goals, dialogue style, and autonomous behavior simulated through LLMs.
- Dynamic Planning: Multi-step action plans are generated and executed seamlessly with nuanced interactivity.
- Competition System: Handle player-vs-NPC and NPC-vs-NPC contests with dedicated LLM adjudication.
- Narrative Memory: Persistent event history creates coherent, evolving storylines.
- Visual Scene Generation: AI-generated artwork enriches the immersive experience.
- Secure Deployment: Integrates Azure Key Vault for secret management, suitable for cloud and containerized environments.

## Architecture Overview
### Component Details

- **User Client:**  
  Responsive web app using HTML5, CSS, and JavaScript WebSocket client.  
  Sends user commands, receives text and image updates.

- **FastAPI WebSocket Server:**  
  - Accepts connections, requests username at start.  
  - Loads per-user saved `JSON` game state (distinguished by username).  
  - Maintains per-connection narrative history buffer for context.  
  - Dispatches commands to backend game logic.  
  - Sends back narrative, dialogue, images URLs, and room status.

- **Game Backend (Python):**  
  - Uses Pydantic models for strong typed game state, NPCs, rooms, player state, etc.  
  - Parses commands via Azure OpenAI LLM—the intent detection and multi-step plan generation.  
  - Supports multi-agent plans and competitions adjudicated by LLMs.  
  - Generates rich narrative summaries and post-action outcomes.  
  - Calls Azure DALL·E for visual scene generation using latest narrative context.  
  - Handles canonicalization and persistence of complex state, goals, and objects.

- **Azure OpenAI Service:**  
  - **LLMs** generate natural language understanding, planning, narration, NPC dialogue, and competitive outcomes.  
  - **DALL·E** generates vivid, on-demand scene imagery from prompts constructed using current game state and narration history.

- **Azure Key Vault:**  
  - Stores OpenAI API key and other secrets securely.  
  - FastAPI backend retrieves secrets securely via Managed Identity or Service Principal at runtime—never stores keys in code or config.

### Data Flow Summary

![Blank diagram - Logic](https://github.com/user-attachments/assets/d2dab574-28c9-496e-b007-42917446805f)

1. **User Input → WebSocket → FastAPI**   
2. **Intent Parsing & Plan Generation** via Azure OpenAI **→ Game state update & plan execution**  
3. **If competition detected:** multistep adjudication → LLM-driven outcome narration + goal updates  
4. **Plot narration & dialogue sent back to user.**  
5. **Image generation request → DALL·E service** → returned image URL via WebSocket.  
6. **State saved per user in JSON files.**
