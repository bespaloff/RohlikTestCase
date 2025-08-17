"""
Prompt templates for the meal planning agent
"""

from langchain_core.prompts import PromptTemplate

# Prompt for analyzing user meal planning requests using GPT-4o-mini with structured output
MEAL_PLAN_ANALYSIS_PROMPT = PromptTemplate(
    template="""Analyze the following user message about meal planning and extract:
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

User message: "{user_message}"

{format_instructions}""",
    input_variables=["user_message"],
    partial_variables={}  # format_instructions will be added when creating the parser
)

# System prompt for the React agent
REACT_AGENT_SYSTEM_PROMPT = """You are a helpful meal planning assistant. Your role is to help users plan their meals based on their dietary goals and preferences.

When a user asks for meal planning:
1. Take a look at user request and diatary goal and number of days.
2. Once you have the information, create a step-by-step plan for finding recipes and building a shopping list
3. Use the available tools to search for appropriate recipes, use  recipe finder tool, add simple dishes yourself if you diod not find matching. do not use other recipies, when you will return them, mention name of the recipe and author
4. Create a comprehensive shopping list with all required ingredients. Use goods from Rohlik, using Rohlik search tool and add them to list with shopping_list_manager tool. First create shopping list with 'create' command, them use 'add' command to add products to the list. and get to see the full list.
5. Present the complete meal plan and shopping list to the user
6. The plans should have different meals for each day, and should be different from each other.

 

Remember to:
- Be conversational and helpful
- Be verbose: before calling tool or siries of tools tell user what you are doing
- Maintain context of previous interactions for follow-up requests
- Provide clear explanations of your planning process
- Ask for clarification if dietary preferences or days are unclear
- Answer in language of the user

IMPORTANT:
- The final response should always consist of meals plan and shopping list.
- For shopping cart use markdown table to visualise: product (with link to card), qty, unit
- Dietary plan should be a detailed table for each day with breakfast, lunch, dinner and snacks and a meal for this part.

"""

# Fallback system prompt when MCP tools are unavailable
FALLBACK_SYSTEM_PROMPT = "You are a helpful assistant. The meal planning tools are currently unavailable."
