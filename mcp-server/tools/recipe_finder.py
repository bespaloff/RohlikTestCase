import json
import logging
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import Context
from mcp.server.session import ServerSession
from semantic_utils import SemanticRecipeSearch  # type: ignore

logger = logging.getLogger(__name__)


semantic_search = SemanticRecipeSearch()


def init_semantic_index(recipes: List[Dict[str, Any]]) -> int:
    """Build semantic index for provided recipes using module-level search instance."""
    return semantic_search.build_index(recipes)


def recipe_finder(
    ctx: Context[ServerSession, Any],
    dietary_goal: str = "",
    keywords: Optional[List[str]] = None,
    max_results: int = 5,
) -> str:
    """Search for recipes using semantic retrieval + GPT reranking."""
    logger.info("=" * 60)
    logger.info("🔍 TOOL CALL: recipe_finder")
    logger.info("📥 INPUT PARAMETERS:")
    logger.info(f"  - dietary_goal: '{dietary_goal}' (type: {type(dietary_goal).__name__})")
    logger.info(
        f"  - keywords: {json.dumps(keywords, ensure_ascii=False) if keywords else 'None'} (type: {type(keywords).__name__})"
    )
    logger.info(f"  - max_results: {max_results} (type: {type(max_results).__name__})")
    logger.info("-" * 40)

    if keywords is None:
        keywords = []

    recipes = ctx.request_context.lifespan_context.recipes
    if not semantic_search.has_index:
        logger.warning("⚠️ Semantic index not available; returning empty result.")
        result = {"found_recipes": 0, "recipes": []}
        return json.dumps(result, ensure_ascii=False, indent=2)

    logger.info("  🧠 Using semantic retrieval (FAISS + GPT rerank)")
    parts: List[str] = []
    if dietary_goal:
        parts.append(f"Dietary goal: {dietary_goal}")
    if keywords:
        parts.append(f"Keywords: {', '.join(keywords)}")
    if not parts:
        parts.append("General tasty recipes")
    query_text = "; ".join(parts)

    # Build ID map for fast lookup
    id_map: Dict[str, Dict[str, Any]] = {}
    for r in recipes:
        rid = str(r.get("id", ""))
        if rid:
            id_map[rid] = r

    candidates = semantic_search.search(query_text, top_k=max(20, max_results * 3), id_to_recipe=id_map)
    selected_ids = semantic_search.rerank(query_text, candidates, max_results)

    results: List[Dict[str, Any]] = []
    for rid in selected_ids:
        r = id_map.get(str(rid))
        if not r:
            continue
        results.append(
            {
                "id": r.get("id", ""),
                "name": r.get("name", ""),
                "ingredients": r.get("ingredients", ""),
                "author_note": r.get("author_note", ""),
                "steps": r.get("steps", "")[:200] + "..."
                if len(r.get("steps", "")) > 200
                else r.get("steps", ""),
            }
        )

    result = {"found_recipes": len(results), "recipes": results}
    result_json = json.dumps(result, ensure_ascii=False, indent=2)
    logger.info("-" * 40)
    logger.info("📤 RETURN VALUE:")
    logger.info(f"  - Found {len(results)} matching recipes")
    logger.info(f"  - Return type: JSON string")
    logger.info(f"  - Return size: {len(result_json)} characters")
    if len(results) > 0:
        logger.info(f"  - Recipe names: {[r['name'] for r in results]}")
    logger.info("=" * 60)
    return result_json


