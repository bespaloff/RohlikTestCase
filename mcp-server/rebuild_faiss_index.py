#!/usr/bin/env python3
"""
Rebuild FAISS index for Rohlik products with proper ID mapping.

This script:
1. Loads products from products.json
2. Creates embeddings for "name + brand + description" text for each product
3. Builds FAISS index with proper mapping between FAISS indices and product IDs
4. Saves the index and mapping files
"""

import json
import logging
import os
import pickle
from pathlib import Path
from typing import Dict, List, Any
import sys

# Add tools directory to path for imports
tools_dir = Path(__file__).parent / "tools"
sys.path.insert(0, str(tools_dir))

from tools.product_models import Product

try:
    import numpy as np
    import faiss
    from openai import OpenAI
except ImportError as e:
    print(f"❌ Missing required dependencies: {e}")
    print("Install with: pip install numpy faiss-cpu openai")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FAISSIndexBuilder:
    """Builds FAISS index for Rohlik products with proper ID mapping."""
    
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.products_json_file = data_dir / "products.json"
        self.faiss_index_file = data_dir / "faiss_index.bin"
        self.id_mapping_file = data_dir / "faiss_id_mapping.json"
        self.products_pkl_file = data_dir / "products.pkl"
        self.progress_file = data_dir / "rebuild_progress.json"
        
        # Initialize OpenAI client
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("❌ OPENAI_API_KEY environment variable not set")
        
        self.openai_client = OpenAI(api_key=api_key)
        logger.info("✅ OpenAI client initialized")
    
    def load_products(self) -> List[Product]:
        """Load products from JSON file."""
        logger.info(f"📂 Loading products from {self.products_json_file}")
        
        if not self.products_json_file.exists():
            raise FileNotFoundError(f"❌ Products file not found: {self.products_json_file}")
        
        with open(self.products_json_file, 'r', encoding='utf-8') as f:
            products_data = json.load(f)
        
        products = []
        for product_dict in products_data:
            try:
                product = Product(
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
                products.append(product)
            except Exception as e:
                logger.warning(f"⚠️ Failed to parse product {product_dict.get('id', 'unknown')}: {e}")
                continue
        
        logger.info(f"✅ Loaded {len(products)} products")
        
     
        
        return products
    
    def create_search_text(self, product: Product) -> str:
        """Create search text from product name, brand, and description."""
        parts = []
        
        # Add name
        if product.name:
            parts.append(product.name)
        
        # Add brand
        if product.brand:
            parts.append(product.brand)
        
        # Add cleaned description (truncated to avoid token limits)
        if product.description:
            clean_desc = product.clean_description()
            if clean_desc:
                # Truncate description to max 1000 characters to avoid token limits
                if len(clean_desc) > 1000:
                    clean_desc = clean_desc[:1000] + "..."
                parts.append(clean_desc)
        
        search_text = " ".join(parts)

        # Final safety check - limit total length to ~2000 characters (~500 tokens)
        if len(search_text) > 2000:
            search_text = search_text[:2000] + "..."
            
        return search_text
    
    def create_embeddings_individually(self, products: List[Product]) -> tuple[np.ndarray, Dict[int, Dict], List[Product]]:
        """Create embeddings for products in batches of 20 and track mapping.
        
        Returns:
            - embeddings array
            - index mapping (FAISS index -> product info)
            - list of successfully processed products (aligned with embeddings)
        """
        logger.info(f"🧠 Creating embeddings for {len(products)} products in batches of 20 using text-embedding-3-small")
        
        all_embeddings = []
        index_mapping = {}  # FAISS index -> {product_id, search_text}
        successful_products = []  # Products that were successfully processed
        failed_count = 0
        batch_size = 20
        
        # Process products in batches of 20
        for batch_start in range(0, len(products), batch_size):
            batch_end = min(batch_start + batch_size, len(products))
            batch_products = products[batch_start:batch_end]
            
            logger.info(f"  Processing batch {batch_start//batch_size + 1}/{(len(products) + batch_size - 1)//batch_size} "
                       f"(products {batch_start + 1}-{batch_end})")
            
            # Prepare batch data
            batch_texts = []
            batch_product_info = []
            
            # Create search texts for all products in this batch
            for product in batch_products:
                search_text = self.create_search_text(product)
    
                
                if not search_text.strip():
                    logger.warning(f"⚠️ Skipping product {product.id} - no search text")
                    failed_count += 1
                    continue
                
                batch_texts.append(search_text)
                batch_product_info.append({
                    'product': product,
                    'search_text': search_text
                })
            
            if not batch_texts:
                logger.warning(f"⚠️ No valid texts in batch {batch_start//batch_size + 1}, skipping")
                continue
            
            try:
                # Create embeddings for the entire batch
                response = self.openai_client.embeddings.create(
                    model="text-embedding-3-small",
                    input=batch_texts  # List of texts for batch processing
                )
                
                # Process each embedding in the batch
                for i, (embedding_data, product_info) in enumerate(zip(response.data, batch_product_info)):
                    embedding = embedding_data.embedding
                    all_embeddings.append(embedding)
                    
                    product = product_info['product']
                    search_text = product_info['search_text']
                    
                    # Record mapping: FAISS index -> product info
                    # CRITICAL: Use len(all_embeddings) - 1 to ensure alignment
                    faiss_index = len(all_embeddings) - 1
                    index_mapping[faiss_index] = {
                        "product_id": product.id,
                        "search_text": search_text[:200] + "..." if len(search_text) > 200 else search_text
                    }
                    
                    # Keep track of successfully processed products
                    successful_products.append(product)
                
                logger.info(f"    ✅ Batch success: {len(batch_texts)} products processed")
                
            except Exception as e:
                logger.error(f"❌ Failed to create embeddings for batch {batch_start//batch_size + 1}: {e}")
                failed_count += len(batch_texts)
                continue
            
            # Save progress every few batches
            if (batch_start // batch_size + 1) % 5 == 0:  # Every 5 batches (100 products)
                progress = {
                    "processed": batch_end,
                    "total": len(products),
                    "successful": len(successful_products),
                    "failed": failed_count
                }
                try:
                    with open(self.progress_file, 'w') as f:
                        json.dump(progress, f, indent=2)
                    logger.info(f"💾 Progress saved: {batch_end}/{len(products)} processed")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to save progress: {e}")
        
        if not all_embeddings:
            raise ValueError("❌ No embeddings were created")
        
        embeddings_array = np.array(all_embeddings, dtype=np.float32)
        logger.info(f"✅ Created embeddings with shape {embeddings_array.shape}")
        logger.info(f"📊 Success: {len(successful_products)} products, Failed: {failed_count} products")
        
        # Verify alignment
        if len(all_embeddings) != len(index_mapping) or len(all_embeddings) != len(successful_products):
            raise ValueError(f"❌ Alignment error: embeddings={len(all_embeddings)}, mapping={len(index_mapping)}, products={len(successful_products)}")
        
        logger.info("✅ Verified alignment between embeddings, mapping, and products")
        
        return embeddings_array, index_mapping, successful_products
    
    def build_faiss_index(self, embeddings: np.ndarray) -> faiss.Index:
        """Build FAISS index from embeddings."""
        logger.info(f"🔍 Building FAISS index for {embeddings.shape[0]} embeddings")
        
        # Use IndexFlatIP for inner product (cosine similarity with normalized vectors)
        dimension = embeddings.shape[1]
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Create index
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings)
        
        logger.info(f"✅ Built FAISS index with {index.ntotal} vectors, dimension {dimension}")
        
        return index
    
    def save_index_and_mapping(self, index: faiss.Index, index_mapping: Dict[int, Dict], successful_products: List[Product]):
        """Save FAISS index and enhanced ID mapping."""
        logger.info("💾 Saving FAISS index and mapping files")
        
        # Final alignment check before saving
        if index.ntotal != len(index_mapping) or index.ntotal != len(successful_products):
            raise ValueError(f"❌ Final alignment check failed: index={index.ntotal}, mapping={len(index_mapping)}, products={len(successful_products)}")
        
        # Save FAISS index
        faiss.write_index(index, str(self.faiss_index_file))
        logger.info(f"✅ Saved FAISS index to {self.faiss_index_file}")
        
        # Save enhanced ID mapping as JSON (includes FAISS index, product ID, and search text preview)
        with open(self.id_mapping_file, 'w', encoding='utf-8') as f:
            json.dump(index_mapping, f, indent=2, ensure_ascii=False)
        logger.info(f"✅ Saved enhanced ID mapping to {self.id_mapping_file}")
        
        # Save only successfully processed products as pickle for compatibility
        with open(self.products_pkl_file, 'wb') as f:
            pickle.dump(successful_products, f)
        logger.info(f"✅ Saved products pickle to {self.products_pkl_file}")
        
        # Log mapping statistics
        logger.info(f"📊 Mapping contains {len(index_mapping)} entries")
        logger.info(f"📊 FAISS index size: {index.ntotal} vectors")
        logger.info(f"📊 Successfully processed products: {len(successful_products)} items")
        logger.info("✅ All files are perfectly aligned!")
    
    def rebuild_index(self):
        """Main method to rebuild the FAISS index."""
        logger.info("🚀 Starting FAISS index rebuild")
        
        # Load products
        products = self.load_products()
        
        if not products:
            logger.error("❌ No products loaded, cannot build index")
            return
        
        # Create embeddings individually for each product
        embeddings, index_mapping, successful_products = self.create_embeddings_individually(products)
        
        # Build FAISS index
        index = self.build_faiss_index(embeddings)
        
        # Save everything (using only successfully processed products)
        self.save_index_and_mapping(index, index_mapping, successful_products)
        
        logger.info("🎉 FAISS index rebuild completed successfully!")
        logger.info(f"📊 Index contains {index.ntotal} products")
        logger.info(f"📁 Files saved in {self.data_dir}")


def main():
    """Main function."""
    # Get data directory
    script_dir = Path(__file__).parent
    data_dir = script_dir / "tools" / "rohlik_data"
    
    if not data_dir.exists():
        logger.error(f"❌ Data directory not found: {data_dir}")
        return
    
    try:
        builder = FAISSIndexBuilder(data_dir)
        builder.rebuild_index()
        
    except Exception as e:
        logger.error(f"❌ Failed to rebuild index: {type(e).__name__}: {e}")
        return


if __name__ == "__main__":
    main()
