# Meal Planning MCP Server

This is the MCP (Model Context Protocol) server that provides meal planning tools for the agent with comprehensive logging capabilities.

## Features

- **recipe_finder**: Search recipes based on dietary criteria and keywords
- **shopping_list_manager**: Create and manage shopping lists
- **Extended Logging**: Comprehensive logging of all tool calls with parameters and return values
- **Semantic Search (optional)**: FAISS + OpenAI embeddings + GPT-4o-mini reranking

## Enhanced Logging

The server now includes comprehensive logging that tracks:

### Tool Call Logging
- **Input Parameters**: All parameters passed to each tool, including their values and types
- **Processing Details**: Step-by-step processing information during tool execution
- **Return Values**: Complete return values with metadata about the response

### Log Format
```
2025-08-14 14:39:20 - module_name - INFO - ============================================================
2025-08-14 14:39:20 - module_name - INFO - 🔍 TOOL CALL: recipe_finder
2025-08-14 14:39:20 - module_name - INFO - 📥 INPUT PARAMETERS:
2025-08-14 14:39:20 - module_name - INFO -   - dietary_goal: 'vegetarian' (type: str)
2025-08-14 14:39:20 - module_name - INFO -   - keywords: ["pasta", "tomato"] (type: list)
2025-08-14 14:39:20 - module_name - INFO -   - max_results: 3 (type: int)
2025-08-14 14:39:20 - module_name - INFO - ----------------------------------------
2025-08-14 14:39:20 - module_name - INFO - [Processing details...]
2025-08-14 14:39:20 - module_name - INFO - ----------------------------------------
2025-08-14 14:39:20 - module_name - INFO - 📤 RETURN VALUE:
2025-08-14 14:39:20 - module_name - INFO -   - Found 2 matching recipes
2025-08-14 14:39:20 - module_name - INFO -   - Return type: JSON string
2025-08-14 14:39:20 - module_name - INFO - ============================================================
```

### Log Features
- **Timestamps**: Each log entry includes precise timestamps
- **Visual Separators**: Clear visual boundaries between different tool calls
- **Type Information**: Parameter types are logged for debugging
- **JSON Formatting**: Complex objects are properly formatted as JSON
- **Emoji Icons**: Visual indicators for different log types (🔍 for search, 🛒 for shopping, etc.)

## Installation

```bash
cd mcp-server
uv pip install -e .
```

If you want to enable semantic search, set your OpenAI API key:

```bash
export OPENAI_API_KEY=sk-...
```

## Running the Server

```bash
python mcp_server.py
```

## Tools

### recipe_finder
Searches for recipes based on:
- Dietary goals (vegetarian, low-carb, high-protein, etc.)
- Keywords
- Maximum number of results

Optional parameters:
- `semantic` (bool, default: true): Enable two-step semantic retrieval (FAISS top-K + GPT-4o-mini reranking). Falls back to keyword filtering if FAISS/OpenAI are unavailable.

### shopping_list_manager
Manages shopping lists with actions:
- `create`: Create a new shopping list
- `add`: Add ingredients to a shopping list
- `get`: Retrieve shopping list with consolidated ingredients
- `clear`: Clear a shopping list

## Data

The server uses `recipes.csv` containing recipe data with:
- Recipe names
- Ingredients lists
- Cooking steps
- Author notes

The semantic index is built at startup from the recipe rows. It uses `text-embedding-3-small` to embed recipe summaries and FAISS inner-product search for cosine-similarity ranking.

## MCP Protocol

The server communicates via stdio using the MCP protocol, making it compatible with any MCP-capable client.