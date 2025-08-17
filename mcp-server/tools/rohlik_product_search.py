import json
import logging
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import sys

 
from tools.product_models import Product
 
 
try:
    from mcp.server.fastmcp import Context
    from mcp.server.session import ServerSession
except ImportError:
    # For testing without MCP environment
    Context = None  # type: ignore
    ServerSession = None  # type: ignore

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



class RohlikProductSearch:
    """FAISS-based product search for Rohlik.cz products"""
    
    def __init__(self, data_dir: Optional[str] = None):
        self.data_dir = Path(data_dir) if data_dir else Path(__file__).parent / "rohlik_data"
        self.index_file = self.data_dir / "faiss_index.bin"
        self.products_file = self.data_dir / "products.pkl"
        self.products_json_file = self.data_dir / "products.json"
        self.id_mapping_file = self.data_dir / "faiss_id_mapping.json"
        
        self._faiss_index = None
        self._products: List[Product] = []
        self._id_mapping: Dict[int, int] = {}  # FAISS index -> product ID
        self._product_id_to_obj: Dict[int, Product] = {}  # product ID -> Product object
        self._openai_client = self._init_openai_client()
        
        # Load index and products if available
        self._load_index_and_products()
    
    @property
    def is_available(self) -> bool:
        return (np is not None and faiss is not None and 
                self._openai_client is not None and self.has_index)
    
    @property
    def has_index(self) -> bool:
        return (self._faiss_index is not None and 
                len(self._id_mapping) > 0 and 
                len(self._product_id_to_obj) > 0)
    
    def _init_openai_client(self):
        api_key = os.environ.get("OPENAI_API_KEY")
        if OpenAI is None or not api_key:
            logger.warning("⚠️ OpenAI client not available. Set OPENAI_API_KEY and install openai>=1.0.0.")
            return None
        try:
            return OpenAI(api_key=api_key)
        except Exception as e:
            logger.error(f"❌ Failed to initialize OpenAI client: {type(e).__name__}: {e}")
            return None
    
    def _load_index_and_products(self):
        """Load FAISS index, ID mapping, and products from disk"""
        try:
            if not self.index_file.exists():
                logger.warning(f"⚠️ FAISS index not found at {self.index_file}")
                return
            
            # Load FAISS index
            self._faiss_index = faiss.read_index(str(self.index_file))
            
            # Load ID mapping
            if self.id_mapping_file.exists():
                try:
                    with open(self.id_mapping_file, 'r', encoding='utf-8') as f:
                        id_mapping_data = json.load(f)
                    
                    # Handle both old and new mapping formats
                    if id_mapping_data and isinstance(list(id_mapping_data.values())[0], dict):
                        # New enhanced format: FAISS index -> {product_id, search_text}
                        self._id_mapping = {int(k): v["product_id"] for k, v in id_mapping_data.items()}
                        logger.info(f"✅ Loaded enhanced ID mapping with {len(self._id_mapping)} entries")
                    else:
                        # Old simple format: FAISS index -> product_id
                        self._id_mapping = {int(k): int(v) for k, v in id_mapping_data.items()}
                        logger.info(f"✅ Loaded simple ID mapping with {len(self._id_mapping)} entries")
                    
                except Exception as e:
                    logger.error(f"❌ Failed to load ID mapping: {e}")
                    return
            else:
                logger.warning(f"⚠️ ID mapping file not found at {self.id_mapping_file}")
                return
            
            # Try to load products from pickle first, then fallback to JSON
            products_data = None
            
            if self.products_file.exists():
                try:
                    # Ensure the tools directory is in the Python path for pickle loading
                    tools_dir = str(Path(__file__).parent)
                    if tools_dir not in sys.path:
                        sys.path.insert(0, tools_dir)
                    
                    with open(self.products_file, 'rb') as f:
                        products_data = pickle.load(f)
                    logger.info("✅ Loaded products from pickle file")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to load pickle file: {e}, trying JSON...")
                    products_data = None
            
            # Fallback to JSON file
            if products_data is None and self.products_json_file.exists():
                try:
                    with open(self.products_json_file, 'r', encoding='utf-8') as f:
                        json_products_data = json.load(f)
                    # Convert to Product objects
                    products_data = [self._dict_to_product(p) for p in json_products_data]
                    logger.info("✅ Loaded products from JSON file")
                except Exception as e:
                    logger.error(f"❌ Failed to load JSON file: {e}")
                    return
            
            if products_data is None:
                logger.warning("⚠️ No product data files found")
                return
            
            # Store products and create ID mapping
            self._products = products_data
            self._product_id_to_obj = {product.id: product for product in products_data}
            
            logger.info(f"✅ Loaded FAISS index with {self._faiss_index.ntotal} entries, "
                       f"{len(self._id_mapping)} ID mappings, and {len(self._products)} products")
            
        except Exception as e:
            logger.error(f"❌ Failed to load index and products: {type(e).__name__}: {e}")
            self._faiss_index = None
            self._products = []
            self._id_mapping = {}
            self._product_id_to_obj = {}
    
    def _dict_to_product(self, product_dict: Dict[str, Any]) -> Product:
        """Convert dictionary to Product object"""
        return Product(
            id=product_dict.get('id', 0),
            name=product_dict.get('name', ''),
            slug=product_dict.get('slug', ''),
            brand=product_dict.get('brand', ''),
            description=product_dict.get('description', ''),
            textual_amount=product_dict.get('textual_amount', ''),
            unit=product_dict.get('unit', ''),
            main_category_id=product_dict.get('main_category_id', 0),
            images=product_dict.get('images', []),
            countries=product_dict.get('countries', []),
            badges=product_dict.get('badges', []),
            filters=product_dict.get('filters', []),
            information=product_dict.get('information', [])
        )
    
    def search_products(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """Search products using the FAISS index"""
        if not self.is_available:
            logger.warning("⚠️ Product search not available. Check FAISS index and OpenAI client.")
            return []
        
        try:
            # Create query embedding
            response = self._openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=[query]
            )
            query_embedding = np.array([response.data[0].embedding], dtype=np.float32)
            
            # Normalize for cosine similarity (same as in index building)
            faiss.normalize_L2(query_embedding)
            
            # Search
            k = min(k, self._faiss_index.ntotal)
            scores, indices = self._faiss_index.search(query_embedding, k)
            
            # Format results
            results = []
            for score, faiss_idx in zip(scores[0], indices[0]):
                if faiss_idx >= 0 and faiss_idx in self._id_mapping:
                    # Get product ID from FAISS index mapping
                    product_id = self._id_mapping[faiss_idx]
                    
                    # Get product object from ID
                    product = self._product_id_to_obj.get(product_id)
                    
                    if product is not None:
                        clean_desc = product.clean_description() or ""
                        description = clean_desc[:200] + "..." if len(clean_desc) > 200 else clean_desc
                        
                        results.append({
                            "id": product.id,
                            "name": product.name or "",
                            "brand": product.brand or "",
                            "description": description,
                            "textual_amount": product.textual_amount or "",
                            "unit": product.unit or "",
                            "link": f"https://www.rohlik.cz/en-CZ/{product.id}-{product.slug or ''}",
                            "images": product.images[:1] if product.images else [],  # Just first image
                            "countries": [c.get('name', '') for c in (product.countries or [])],
                            "badges": [b.get('title', '') for b in (product.badges or [])],
                            "score": float(score)
                        })
                    else:
                        logger.warning(f"⚠️ Product with ID {product_id} not found in product mapping")
            logger.info(f"✅ {json.dumps(results, ensure_ascii=False, indent=2)}")
            return results
            
        except Exception as e:
            logger.error(f"❌ Product search failed: {type(e).__name__}: {e}")
            return []


# Global search instance
rohlik_search = RohlikProductSearch()


def rohlik_product_search(
    ctx: Optional[Any] = None,
    query: str = "",
    max_results: int = 10,
) -> str:
    """Search for Rohlik.cz products using semantic search."""
    logger.info("=" * 60)
    logger.info("🔍 TOOL CALL: rohlik_product_search")
    logger.info("📥 INPUT PARAMETERS:")
    logger.info(f"  - query: '{query}' (type: {type(query).__name__})")
    logger.info(f"  - max_results: {max_results} (type: {type(max_results).__name__})")
    logger.info("-" * 40)
    
    if not query.strip():
        logger.warning("⚠️ Empty query provided")
        result = {"found_products": 0, "products": [], "error": "Empty query provided"}
        return json.dumps(result, ensure_ascii=False, indent=2)
    
    if not rohlik_search.is_available:
        logger.warning("⚠️ Rohlik product search not available")
        result = {
            "found_products": 0, 
            "products": [], 
            "error": "Product search not available. Check FAISS index and OpenAI API key."
        }
        return json.dumps(result, ensure_ascii=False, indent=2)
    
    logger.info(f"  🛒 Searching Rohlik products with query: '{query}'")
    
    try:
        products = rohlik_search.search_products(query, max_results)
        
        result = {
            "found_products": len(products),
            "products": products,
            "query": query
        }
        
        result_json = json.dumps(result, ensure_ascii=False, indent=2)
        
        logger.info("-" * 40)
        logger.info("📤 RETURN VALUE:")
        logger.info(f"  - Found {len(products)} matching products")
        logger.info(f"  - Return type: JSON string")
        logger.info(f"  - Return size: {len(result_json)} characters")
        if len(products) > 0:
            logger.info(f"  - Product names: {[p['name'] for p in products[:3]]}{'...' if len(products) > 3 else ''}")
        logger.info("=" * 60)
        
        return result_json
        
    except Exception as e:
        logger.error(f"❌ Product search failed: {type(e).__name__}: {e}")
        result = {
            "found_products": 0,
            "products": [],
            "error": f"Search failed: {str(e)}"
        }
        return json.dumps(result, ensure_ascii=False, indent=2)
