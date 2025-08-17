#!/usr/bin/env python

import json
import logging
import csv
import os
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set up logging early
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info("🔧 Environment variables loaded from .env file")

from tools.recipe_finder import init_semantic_index  # type: ignore
from mcp.server.fastmcp import FastMCP
import uvloop

# Load recipes from CSV
recipes_data: List[Dict[str, Any]] = []
csv_path = Path(__file__).parent / "recipes.csv"

# In-memory shopping list storage
shopping_list: Dict[str, List[Dict[str, Any]]] = {}

# Semantic search is now initialized within tools.recipe_finder

@dataclass
class AppContext:
    """Application context with meal planning data."""
    recipes: List[Dict[str, Any]]
    shopping_lists: Dict[str, List[Dict[str, Any]]]

def load_recipes():
    """Load recipes from CSV file"""
    global recipes_data
    logger.info("📂 Loading recipes from CSV...")
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            recipes_data = list(reader)
            logger.info(f"✅ Successfully loaded {len(recipes_data)} recipes from {csv_path.name}")
            if recipes_data:
                # Log sample of loaded recipes
                logger.info("📋 Sample of loaded recipes:")
                for i, recipe in enumerate(recipes_data[:3]):
                    logger.info(f"  {i+1}. {recipe.get('name', 'Unknown')}")
                if len(recipes_data) > 3:
                    logger.info(f"  ... and {len(recipes_data) - 3} more recipes")
            return recipes_data
    except FileNotFoundError:
        logger.error(f"❌ Recipe file not found: {csv_path}")
        return []
    except Exception as e:
        logger.error(f"❌ Failed to load recipes: {type(e).__name__}: {e}")
        return []


def _build_id_map(recipes: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    id_map: Dict[str, Dict[str, Any]] = {}
    for r in recipes:
        rid = str(r.get("id", ""))
        if rid:
            id_map[rid] = r
    return id_map

@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    """Manage application lifecycle with recipe loading."""
    # Initialize on startup
 
    
    recipes = load_recipes()
    shopping_lists = {}
    indexed = 0
    try:
        indexed = init_semantic_index(recipes)
    except Exception as e:
        logger.warning(f"⚠️ Semantic index build skipped: {type(e).__name__}: {e}")
 
    try:
        yield AppContext(recipes=recipes, shopping_lists=shopping_lists)
    finally:
        # Cleanup on shutdown
        logger.info("=" * 60)
      

# Create FastMCP server with lifespan
mcp = FastMCP("meal-planning-mcp", lifespan=app_lifespan)

# Import and register tools directly
from tools.recipe_finder import recipe_finder  # noqa: E402
from tools.shopping_list import shopping_list_manager  # noqa: E402
from tools.rohlik_product_search import rohlik_product_search  # noqa: E402

# Register tools with decorators
mcp.tool(description="""Search for recipes using semantic retrieval and GPT reranking.

INPUT:
- dietary_goal (str): Dietary preference or goal (e.g., "low carb", "vegetarian", "high protein"). Can be empty string.
- keywords (Optional[List[str]]): List of ingredient keywords or cooking terms to search for. Can be None or empty list.
- max_results (int): Maximum number of recipes to return (default: 5).

OUTPUT:
Returns a JSON string containing:
- found_recipes (int): Number of recipes found
- recipes (List[Dict]): Array of recipe objects, each containing:
  - id: Recipe identifier
  - name: Recipe name
  - ingredients: List of ingredients
  - author_note: Additional notes from recipe author
  - steps: Cooking instructions (truncated to 200 chars if longer)

Example output:
{
  "found_recipes": 2,
  "recipes": [
    {
      "id": "123",
      "name": "Grilled Chicken Salad",
      "ingredients": "chicken breast, lettuce, tomatoes...",
      "author_note": "Perfect for summer meals",
      "steps": "1. Season chicken... 2. Grill for 6 minutes..."
    }
  ]
}""")(recipe_finder)

mcp.tool(description="""Manage shopping lists - create, add ingredients, retrieve, or clear lists.

INPUT:
- action (str): Action to perform. Must be one of: "create", "add", "get", "clear"
- list_id (str): Unique identifier for the shopping list
- ingredients (Optional[List[Dict[str, str]]]): List of ingredient dictionaries for "add" action. Each ingredient dict can contain:
  - name: Ingredient name (required)
  - quantity: Amount needed (optional)
  - recipe_id: ID of recipe this ingredient is for (optional)
  - link: URL link to product (optional)

OUTPUT:
Returns a JSON string with different formats based on action:

For "create", "add", "clear" actions:
{
  "status": "success",
  "message": "Description of action performed",
  "total_items": <number> (only for "add" action)
}

For "get" action:
{
  "list_id": "shopping_list_id",
  "items": [
    {
      "name": "ingredient_name",
      "quantity": "amount",
      "recipes": ["recipe_id1", "recipe_id2"],
      "link": "product_url" (if available)
    }
  ],
  "total_items": <number>
}

For errors:
{
  "error": "Error description"
}""")(shopping_list_manager)

mcp.tool(description="""Search for Rohlik.cz grocery products using semantic search with FAISS index and OpenAI embeddings.

INPUT:
- ctx (Context): MCP server context (automatically provided)
- query (str): Search query describing the product you're looking for (e.g., "organic milk", "gluten-free bread", "fresh tomatoes")
- max_results (int): Maximum number of products to return (default: 10)

OUTPUT:
Returns a JSON string containing:
- found_products (int): Number of products found
- products (List[Dict]): Array of product objects, each containing:
  - id: Product ID number
  - name: Product name
  - brand: Brand name
  - description: Product description (truncated to 200 chars)
  - textual_amount: Package size/amount text
  - unit: Unit of measurement
  - link: Direct URL to product on Rohlik.cz
  - images: Array of image URLs (first image only)
  - countries: Array of country names (origin/production)
  - badges: Array of product badges (e.g., "Bio", "Fair Trade")
  - score: Semantic similarity score (float)
- query: Original search query
- error: Error message (only present if search fails)

Example output:
{
  "found_products": 3,
  "products": [
    {
      "id": 12345,
      "name": "Organic Whole Milk",
      "brand": "Farmer's Best",
      "description": "Fresh organic milk from grass-fed cows...",
      "textual_amount": "1 l",
      "unit": "l",
      "link": "https://www.rohlik.cz/en-CZ/12345-organic-whole-milk",
      "images": ["https://example.com/milk.jpg"],
      "countries": ["Czech Republic"],
      "badges": ["Bio", "Local"],
      "score": 0.95
    }
  ],
  "query": "organic milk"
}

 """)(rohlik_product_search)

if __name__ == "__main__":
    # Configure logging with more detailed format
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    import sys
    
    logger.info("=" * 60)
    logger.info("🍽️ Starting Meal Planning MCP Server...")
    logger.info(f"📁 CSV file path: {csv_path}")
    logger.info(f"📊 Logging level: INFO")
    
    # Check if HTTP mode is requested
    if len(sys.argv) > 1 and sys.argv[1] == "--http":
        logger.info("🌐 Starting HTTP server on port 8000")
        logger.info("=" * 60)
        
        mcp.settings.host = "0.0.0.0"
        mcp.settings.port = 8000
        # (optional) mcp.settings.streamable_http_path = "/mcp"  # default is /mcp

        logger.info("🌐 Starting Streamable HTTP server on http://0.0.0.0:8000/mcp")
        logger.info("=" * 60)
        mcp.run(transport="streamable-http")   
    else:
        logger.info("📡 Starting STDIO server")
        logger.info("=" * 60)
        
        # Run the server with STDIO transport (default)

        uvloop.install()
        mcp.run()