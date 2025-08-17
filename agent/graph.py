import logging
from typing import Annotated, List, Dict
import uuid
from pathlib import Path
import json
import time

from langchain_core.messages import AnyMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.callbacks import BaseCallbackHandler
from langchain_openai import ChatOpenAI
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph, END, add_messages
from pydantic import BaseModel, Field
import openai
import json

from langchain_mcp_adapters.client import MultiServerMCPClient
from prompts import MEAL_PLAN_ANALYSIS_PROMPT, REACT_AGENT_SYSTEM_PROMPT, FALLBACK_SYSTEM_PROMPT

logger = logging.getLogger(__name__)

class ToolLogger(BaseCallbackHandler):
    def __init__(self): 
        self._t0 = {}
    
    def on_tool_start(self, serialized, input_str, *, run_id, parent_run_id=None, **kw):
        name = serialized.get("name", "unknown")
        self._t0[run_id] = time.perf_counter()
        logger.info(f"🔧 TOOL START: {name}  run_id={run_id}\n⮑ input: {input_str}")

    def on_tool_end(self, output, *, run_id, parent_run_id=None, **kw):
        dt = time.perf_counter() - self._t0.pop(run_id, time.perf_counter())
        logger.info(f"✅ TOOL END  run_id={run_id}  ({dt:.2f}s)\n⮐ output: {str(output)[:2000]}")

    def on_tool_error(self, error, *, run_id, parent_run_id=None, **kw):
        logger.exception(f"❌ TOOL ERROR run_id={run_id}: {error}")

# Initialize the models
model = ChatOpenAI(model="gpt-4.1-mini", tags=["assistant"], max_completion_tokens=40000)
weak_model = ChatOpenAI(model="gpt-4o-mini", tags=["weak_model"])

# Global MCP client and agent
mcp_client = None
react_agent = None

class MealPlanAnalysis(BaseModel):
    """Structured output for analyzing meal planning requests"""
    dietary_goal: str = Field(
        default="",
        description="Extracted dietary goal: vegetarian, vegan, low-carb, high-protein, paleo, gluten-free, healthy, or empty string if none detected"
    )
    num_days: int = Field(
        default=0,
        description="Number of days for meal planning extracted from the text (0 if not specified)"
    )
    confidence_dietary: float = Field(
        default=0.0,
        description="Confidence level for dietary goal extraction (0.0 to 1.0)"
    )
    confidence_days: float = Field(
        default=0.0,
        description="Confidence level for days extraction (0.0 to 1.0)"
    )
    reasoning: str = Field(
        default="",
        description="Brief explanation of the extraction reasoning"
    )

def prune_for_next_turn(msgs: list[AnyMessage]) -> list[AnyMessage]:
    """Drop orphan tool messages & assistant messages that were only tool_calls."""
    pruned = []
    for m in msgs:
        # Drop all ToolMessage for next turn
        if isinstance(m, ToolMessage):
            continue
        # If an AIMessage has tool_calls (function/tool request) and no text,
        # drop it for next turn; keep only natural language outputs.
        if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
            # If the AIMessage ALSO contains human-readable text, keep that text-only clone.
            if (isinstance(m.content, str) and m.content.strip()):
                pruned.append(AIMessage(content=m.content))
            # else drop it
            continue
        pruned.append(m)
    # Keep the last ~20 for safety
    return pruned[-20:]

class GraphProcessingState(BaseModel):
    """State for the meal planning agent"""
    messages: Annotated[list[AnyMessage], add_messages] = Field(default_factory=list)
    prompt: str = Field(default_factory=str, description="The prompt to be used for the model")
    tools_enabled: dict = Field(default_factory=dict, description="The tools enabled for the assistant")
    dietary_goal: str = Field(default="", description="User's dietary goal")
    num_days: int = Field(default=0, description="Number of days for meal planning")
    selected_recipes: List[Dict] = Field(default_factory=list, description="Selected recipes for the meal plan")
    shopping_list_id: str = Field(default="", description="Current shopping list ID")

async def analyze_meal_request(user_message: str) -> MealPlanAnalysis:
    """Analyze user message using GPT-4o-mini with structured output via direct OpenAI API"""
    logger.debug(f"🧠 Analyzing user message with GPT-4o-mini: '{user_message[:100]}...'")
    
    # Create the system prompt for analysis
    system_prompt = """Analyze the following user message about meal planning and extract:
1. Dietary goal (if any): vegetarian, vegan, low-carb, high-protein, paleo, gluten-free, healthy
2. Number of days for meal planning

Be smart about interpreting natural language:
- "a week" = 7 days
- "weekend" = 2 days  
- "few days" = 3 days
- "tomorrow" = 1 day
- "couple days" = 2 days
- Numbers written as words (e.g., "three days" = 3)

For dietary goals, look for:
- vegetarian: veggie, no meat, meat-free, plant-based
- vegan: no animal products, plant only
- low-carb: keto, ketogenic, no carbs, reduce carbs
- high-protein: protein-rich, lots of protein
- paleo: paleolithic, caveman diet
- gluten-free: no gluten, celiac
- healthy: balanced, nutritious, clean eating

Respond with a JSON object containing:
- dietary_goal: string (one of the categories above or empty string)
- num_days: integer (0 if not specified)
- confidence_dietary: float (0.0 to 1.0)
- confidence_days: float (0.0 to 1.0)
- reasoning: string (brief explanation)"""
    
    try:
        # Make direct OpenAI API call with structured output
        client = openai.AsyncOpenAI()
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"User message: {user_message}"}
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "meal_plan_analysis",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "dietary_goal": {
                                "type": "string",
                                "description": "Extracted dietary goal or empty string"
                            },
                            "num_days": {
                                "type": "integer",
                                "description": "Number of days for meal planning (0 if not specified)"
                            },
                            "confidence_dietary": {
                                "type": "number",
                                "description": "Confidence level for dietary goal extraction (0.0 to 1.0)"
                            },
                            "confidence_days": {
                                "type": "number",
                                "description": "Confidence level for days extraction (0.0 to 1.0)"
                            },
                            "reasoning": {
                                "type": "string",
                                "description": "Brief explanation of the extraction reasoning"
                            }
                        },
                        "required": ["dietary_goal", "num_days", "confidence_dietary", "confidence_days", "reasoning"],
                        "additionalProperties": False
                    }
                }
            },
            temperature=0.1
        )
        
        # Parse the JSON response
        response_content = response.choices[0].message.content
        analysis_data = json.loads(response_content)
        
        # Create MealPlanAnalysis object from the response
        result = MealPlanAnalysis(**analysis_data)
        
        # Log extraction results internally (not visible to user)
        logger.debug(f"  📋 Extracted dietary goal: {result.dietary_goal} (confidence: {result.confidence_dietary:.2f})")
        logger.debug(f"  📅 Extracted days: {result.num_days} (confidence: {result.confidence_days:.2f})")
        logger.debug(f"  💭 Reasoning: {result.reasoning}")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Error in GPT-4o-mini analysis: {e}")
        # Return empty analysis on error
        return MealPlanAnalysis(
            reasoning=f"Analysis failed: {str(e)}"
        )

async def get_mcp_client():
    """Get or create MCP client"""
    global mcp_client, react_agent
    
    if mcp_client is None:
        logger.info("🔌 Initializing MCP client...")
        
        try:
            # Create MultiServerMCPClient with HTTP transport
            mcp_client = MultiServerMCPClient(
                {
                    "meal-planning": {
                        "url": "http://localhost:8000/mcp",
                        "transport": "streamable_http",
                    }
                }
            )
            
            logger.info("✅ MCP client initialized")
            
            # Get tools from MCP client
            logger.info("🔧 Getting tools from MCP server...")
            tools = await mcp_client.get_tools()
            logger.info(f"📋 Available tools: {[tool.name for tool in tools]}")
            
            # Create React agent with MCP tools and callback handler
            
            react_agent = create_react_agent(
                model,
                tools,
                prompt=REACT_AGENT_SYSTEM_PROMPT
            )
            
            logger.info("✅ React agent created with MCP tools")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize MCP client: {e}")
            logger.info("💡 Make sure the MCP server is accessible")
            mcp_client = None
            react_agent = None
    
    return mcp_client, react_agent

async def assistant_node(state: GraphProcessingState, config=None):
    """Main assistant node that processes user input and generates responses"""
    logger.info("🤖 ASSISTANT NODE: Processing message")
    
    # Clean up tool messages from previous turns to avoid OpenAI API errors
    state.messages = prune_for_next_turn(state.messages)
    
    # Extract dietary goal and days from user message if not set
    if not state.dietary_goal or state.num_days == 0:
        last_human_msg = None
        for msg in reversed(state.messages):
            if isinstance(msg, HumanMessage):
                last_human_msg = msg.content
                break
        
        if last_human_msg:
            # Use GPT-4o-mini with structured output for analysis
            analysis = await analyze_meal_request(last_human_msg)
            
            # Update state with extracted information
            if not state.dietary_goal and analysis.dietary_goal:
                state.dietary_goal = analysis.dietary_goal
                
            if state.num_days == 0 and analysis.num_days > 0:
                state.num_days = analysis.num_days
            
            # Log results internally
            if state.num_days == 0:
                logger.debug("  ⚠️ Number of days not detected - will ask user")
            if not state.dietary_goal:
                logger.debug("  ⚠️ Dietary goal not detected - will proceed with general recipes")
    
    # Create shopping list ID if needed
    if not state.shopping_list_id and state.num_days > 0:
        state.shopping_list_id = f"meal_plan_{uuid.uuid4().hex[:8]}"
        logger.info(f"  🆔 Created shopping list ID: {state.shopping_list_id}")
    
    logger.debug(f"  - Dietary goal: {state.dietary_goal or 'Not specified'}")
    logger.debug(f"  - Number of days: {state.num_days or 'Not specified'}")
    logger.debug(f"  - Shopping list ID: {state.shopping_list_id or 'Not created'}")
    
    # Get MCP client and React agent
    _, agent = await get_mcp_client()
    
    if agent is None:
        logger.error("❌ No React agent available")
        # Fallback to basic model without tools
        from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", FALLBACK_SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="messages"),
        ])
        
        chain = prompt | model
        response = await chain.ainvoke({"messages": state.messages}, config=config)
        
        return {
            "messages": state.messages + [response],
            "dietary_goal": state.dietary_goal,
            "num_days": state.num_days,
            "shopping_list_id": state.shopping_list_id
        }
    
    # Use React agent to process the message
    logger.info("🧠 Using React agent with MCP tools")
    
    try:
        # Prepare messages in the format expected by React agent
        agent_input = {
            "messages": state.messages
        }
        
        # Call the React agent with tool logging
        tool_logger = ToolLogger()
        if config is None:
            config = {}
        
        # Properly handle callbacks - create a new list or add to existing list
        existing_callbacks = config.get("callbacks", [])
        if not isinstance(existing_callbacks, list):
            existing_callbacks = []
        config["callbacks"] = existing_callbacks + [tool_logger]
        
        response = await agent.ainvoke(agent_input, config=config)
        
        logger.info("✅ React agent completed")
        
        return {
            "messages": response["messages"],
            "dietary_goal": state.dietary_goal,
            "num_days": state.num_days,
            "shopping_list_id": state.shopping_list_id
        }
        
    except Exception as e:
        logger.error(f"❌ Error in React agent: {e}")
        
        # Fallback response
        fallback_msg = AIMessage(content=f"I'm experiencing technical difficulties with my meal planning tools. Error: {str(e)}")
        
        return {
            "messages": state.messages + [fallback_msg],
            "dietary_goal": state.dietary_goal,
            "num_days": state.num_days,
            "shopping_list_id": state.shopping_list_id
        }

def define_workflow() -> CompiledStateGraph:
    """Defines the workflow graph for meal planning"""
    logger.info("📊 Initializing meal planning workflow graph")
    
    # Initialize the graph - simplified since React agent handles tool routing
    workflow = StateGraph(GraphProcessingState)
    
    # Add single assistant node (React agent handles everything)
    workflow.add_node("assistant_node", assistant_node)
    
    # Set entry and exit point
    workflow.set_entry_point("assistant_node")
    workflow.set_finish_point("assistant_node")
    
    logger.info("✅ Workflow graph compiled successfully")
    return workflow.compile()

# Create the graph
graph = define_workflow()