import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    np = None  # type: ignore
try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover
    faiss = None  # type: ignore
try:
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore


class SemanticRecipeSearch:
    """FAISS + OpenAI based semantic search over recipes with GPT reranking."""

    def __init__(self) -> None:
        self._faiss_index = None
        self._recipe_ids: List[str] = []
        self._recipe_texts: List[str] = []
        self._openai_client = self._init_openai_client()
        # Cache paths for persisting index and metadata
        self._cache_dir = Path(__file__).parent / ".cache"
        self._index_path = self._cache_dir / "recipes_faiss.index"
        self._meta_path = self._cache_dir / "recipes_meta.json"

    @property
    def is_available(self) -> bool:
        return np is not None and faiss is not None and self._openai_client is not None

    @property
    def has_index(self) -> bool:
        return self._faiss_index is not None and len(self._recipe_ids) > 0

    def _init_openai_client(self):  # type: ignore[no-untyped-def]
        api_key = os.environ.get("OPENAI_API_KEY")
        if OpenAI is None or not api_key:
            logger.warning("⚠️ OpenAI client not available. Set OPENAI_API_KEY and install openai>=1.0.0.")
            return None
        try:
            return OpenAI(api_key=api_key)
        except Exception as e:
            logger.error(f"❌ Failed to initialize OpenAI client: {type(e).__name__}: {e}")
            return None

    def _build_recipe_text(self, recipe: Dict[str, Any]) -> str:
        name = recipe.get("name", "")
        ingredients = recipe.get("ingredients", "")
        author_note = recipe.get("author_note", "")
        return f"Name: {name}\nIngredients: {ingredients}\nNote: {author_note}"

    def _normalize_rows(self, matrix):  # type: ignore[no-untyped-def]
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms

    def build_index(self, recipes: List[Dict[str, Any]]) -> int:
        # If an index is already cached on disk, load it and skip rebuild
        try:
            if self._index_path.exists() and self._meta_path.exists():
                index = faiss.read_index(str(self._index_path))  # type: ignore[arg-type]
                with open(self._meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                recipe_ids = meta.get("recipe_ids", [])
                if isinstance(recipe_ids, list) and len(recipe_ids) > 0:
                    self._faiss_index = index
                    self._recipe_ids = [str(x) for x in recipe_ids]
                    logger.info(
                        f"✅ Loaded FAISS index from disk with {len(self._recipe_ids)} recipes."
                    )
                    return len(self._recipe_ids)
                else:
                    logger.warning("⚠️ Cached metadata missing valid recipe_ids; rebuilding index.")
        except Exception as e:
            logger.warning(f"⚠️ Failed to load cached index; will rebuild. {type(e).__name__}: {e}")

        if not self.is_available:
            logger.warning("⚠️ Semantic components unavailable. Index build skipped.")
            return 0
        if not recipes:
            return 0

        texts: List[str] = []
        ids: List[str] = []
        for r in recipes:
            rid = str(r.get("id", ""))
            if not rid:
                continue
            texts.append(self._build_recipe_text(r))
            ids.append(rid)

        logger.info("🧠 Building embeddings for recipes (OpenAI text-embedding-3-small)...")
        try:
            response = self._openai_client.embeddings.create(  # type: ignore[union-attr]
                model="text-embedding-3-small",
                input=texts,
            )
            vectors = [d.embedding for d in response.data]  # type: ignore[attr-defined]
            matrix = np.array(vectors, dtype=np.float32)
            matrix = self._normalize_rows(matrix)

            dim = matrix.shape[1]
            index = faiss.IndexFlatIP(dim)
            index.add(matrix)

            self._faiss_index = index
            self._recipe_texts = texts
            self._recipe_ids = ids
            logger.info(f"✅ FAISS index built with {len(ids)} recipes (dim={dim}).")
            # Persist to disk for faster startup next time
            try:
                self._cache_dir.mkdir(parents=True, exist_ok=True)
                faiss.write_index(self._faiss_index, str(self._index_path))  # type: ignore[arg-type]
                with open(self._meta_path, "w", encoding="utf-8") as f:
                    json.dump({"recipe_ids": self._recipe_ids}, f)
                logger.info(
                    f"💾 Saved FAISS index and metadata to '{self._cache_dir.relative_to(Path(__file__).parent)}'"
                )
            except Exception as e:
                logger.warning(f"⚠️ Failed to persist index: {type(e).__name__}: {e}")
            return len(ids)
        except Exception as e:
            logger.error(f"❌ Failed to build FAISS index: {type(e).__name__}: {e}")
            return 0

    def search(self, query_text: str, top_k: int, id_to_recipe: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not self.has_index or not self.is_available:
            return []
        try:
            emb = self._openai_client.embeddings.create(  # type: ignore[union-attr]
                model="text-embedding-3-small",
                input=[query_text]
            )
            q = np.array([emb.data[0].embedding], dtype=np.float32)  # type: ignore[attr-defined]
            q = self._normalize_rows(q)
            k = min(top_k, len(self._recipe_ids))
            scores, indices = self._faiss_index.search(q, k)  # type: ignore[union-attr]
            result: List[Dict[str, Any]] = []
            for score, idx in zip(scores[0].tolist(), indices[0].tolist()):
                if idx < 0 or idx >= len(self._recipe_ids):
                    continue
                rid = self._recipe_ids[idx]
                r = id_to_recipe.get(rid)
                if not r:
                    continue
                result.append({
                    "id": rid,
                    "name": r.get("name", ""),
                    "ingredients": r.get("ingredients", ""),
                    "score": float(score),
                })
            return result
        except Exception as e:
            logger.error(f"❌ Semantic search failed: {type(e).__name__}: {e}")
            return []

    def rerank(self, query_text: str, candidates: List[Dict[str, Any]], max_results: int) -> List[str]:
        if not candidates or self._openai_client is None:
            return [c["id"] for c in candidates[:max_results]]
        condensed: List[Dict[str, str]] = []
        for c in candidates[: min(25, len(candidates))]:
            condensed.append({
                "id": str(c.get("id", "")),
                "name": str(c.get("name", ""))[:100],
                "ingredients": str(c.get("ingredients", ""))[:300],
            })
        system_msg = (
            "You are a culinary assistant. Given a user request, choose the best matching recipes "
            "from a candidate list. Prefer strong ingredient/goal matches. Return compact JSON."
        )
        user_msg = {
            "query": query_text,
            "candidates": condensed,
            "instructions": {
                "return_format": {"selected_ids": "list[str]"},
                "max_results": max_results,
            },
        }
        try:
            completion = self._openai_client.chat.completions.create(  # type: ignore[union-attr]
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": json.dumps(user_msg, ensure_ascii=False)}
                ],
                temperature=0.2,
            )
            content = completion.choices[0].message.content or ""
            start = content.find("{")
            end = content.rfind("}")
            if start != -1 and end != -1 and end > start:
                obj = json.loads(content[start:end + 1])
                ids = obj.get("selected_ids", [])
                ids = [str(x) for x in ids][:max_results]
                return ids
            return [c["id"] for c in candidates[:max_results]]
        except Exception as e:
            logger.error(f"❌ GPT rerank failed: {type(e).__name__}: {e}")
            return [c["id"] for c in candidates[:max_results]]


