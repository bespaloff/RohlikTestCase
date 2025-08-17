import json
import logging
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import Context
from mcp.server.session import ServerSession

logger = logging.getLogger(__name__)


def shopping_list_manager(
    ctx: Context[ServerSession, Any],
    action: str,
    list_id: str,
    ingredients: Optional[List[Dict[str, str]]] = None,
) -> str:
    """Manage shopping list - create, add ingredients, or retrieve the list."""
    logger.info("=" * 60)
    logger.info(f"🛒 TOOL CALL: shopping_list_manager")
    logger.info("📥 INPUT PARAMETERS:")
    logger.info(f"  - action: '{action}' (type: {type(action).__name__})")
    logger.info(f"  - list_id: '{list_id}' (type: {type(list_id).__name__})")
    logger.info(
        f"  - ingredients: {json.dumps(ingredients, ensure_ascii=False) if ingredients else 'None'} (type: {type(ingredients).__name__})"
    )
    if ingredients:
        logger.info(f"    → {len(ingredients)} ingredient(s) provided")
    logger.info("-" * 40)

    shopping_lists = ctx.request_context.lifespan_context.shopping_lists

    if action == "create":
        shopping_lists[list_id] = []
        logger.info(f"  ✅ Created new shopping list: {list_id}")
        result = {"status": "success", "message": f"Shopping list '{list_id}' created"}
        result_json = json.dumps(result)
        logger.info("-" * 40)
        logger.info(f"📤 RETURN VALUE:")
        logger.info(f"  - Action result: {result}")
        logger.info(f"  - Return type: JSON string")
        logger.info("=" * 60)
        return result_json

    elif action == "add":
        if list_id not in shopping_lists:
            shopping_lists[list_id] = []
        if ingredients is None:
            ingredients = []
        for ingredient in ingredients:
            shopping_lists[list_id].append(ingredient)
            logger.debug(f"    + Added: {ingredient.get('name', 'Unknown')}")
        logger.info(f"  ✅ Added {len(ingredients)} ingredients to list")
        result = {
            "status": "success",
            "message": f"Added {len(ingredients)} ingredients to shopping list",
            "total_items": len(shopping_lists[list_id]),
        }
        result_json = json.dumps(result)
        logger.info("-" * 40)
        logger.info(f"📤 RETURN VALUE:")
        logger.info(f"  - Action result: {result}")
        logger.info(f"  - Return type: JSON string")
        logger.info("=" * 60)
        return result_json

    elif action == "get":
        items = shopping_lists.get(list_id, [])
        consolidated: Dict[str, Dict[str, Any]] = {}
        for item in items:
            name = item.get("name", "")
            if name in consolidated:
                consolidated[name]["recipes"].append(item.get("recipe_id", ""))
                # If new item has a link and consolidated doesn't, use it
                if item.get("link") and not consolidated[name].get("link"):
                    consolidated[name]["link"] = item.get("link")
            else:
                consolidated[name] = {
                    "name": name,
                    "quantity": item.get("quantity", ""),
                    "recipes": [item.get("recipe_id", "")],
                }
                # Include link if available
                if item.get("link"):
                    consolidated[name]["link"] = item.get("link")
        logger.info(f"  ✅ Retrieved shopping list with {len(consolidated)} unique items")
        result = {"list_id": list_id, "items": list(consolidated.values()), "total_items": len(consolidated)}
        result_json = json.dumps(result, ensure_ascii=False, indent=2)
        logger.info("-" * 40)
        logger.info(f"📤 RETURN VALUE:")
        logger.info(f"  - List ID: {list_id}")
        logger.info(f"  - Total unique items: {len(consolidated)}")
        logger.info(f"  - Return type: JSON string")
        logger.info(f"  - Return size: {len(result_json)} characters")
        logger.info("=" * 60)
        return result_json

    elif action == "clear":
        shopping_lists[list_id] = []
        logger.info(f"  ✅ Cleared shopping list: {list_id}")
        result = {"status": "success", "message": f"Shopping list '{list_id}' cleared"}
        result_json = json.dumps(result)
        logger.info("-" * 40)
        logger.info(f"📤 RETURN VALUE:")
        logger.info(f"  - Action result: {result}")
        logger.info(f"  - Return type: JSON string")
        logger.info("=" * 60)
        return result_json

    error_result = {"error": f"Unknown action: {action}"}
    error_json = json.dumps(error_result)
    logger.error(f"  ❌ Unknown action: {action}")
    logger.info("-" * 40)
    logger.info(f"📤 RETURN VALUE (ERROR):")
    logger.info(f"  - Error: {error_result}")
    logger.info(f"  - Return type: JSON string")
    logger.info("=" * 60)
    return error_json


