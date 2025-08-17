"""Comprehensive tests for Rohlik product search with AI-powered semantic search."""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import Mock, MagicMock, patch
import pytest

# Add the mcp-server directory and tests directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from tools.rohlik_product_search import rohlik_product_search, RohlikProductSearch, Product

# Configure logging at module level to ensure INFO level
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True
)

# Test scenarios for different product search queries
PRODUCT_SEARCH_SCENARIOS = [
    {
        "name": "Fresh Fruits Search",
        "query": "fresh apples and bananas",
        "max_results": 10,
        "expected_categories": ["fruit", "apple", "banana", "fresh"],
        "description": "Should return fresh fruit products, especially apples and bananas"
    },
    {
        "name": "Organic Vegetables",
        "query": "organic vegetables for salad",
        "max_results": 10,
        "expected_categories": ["organic", "vegetable", "salad", "fresh"],
        "description": "Should prioritize organic vegetables suitable for salads"
    },
    {
        "name": "Dairy Products",
        "query": "milk cheese yogurt dairy products",
        "max_results": 10,
        "expected_categories": ["milk", "cheese", "yogurt", "dairy"],
        "description": "Should return various dairy products"
    },
    {
        "name": "Bread and Bakery",
        "query": "fresh bread rolls pastries",
        "max_results": 10,
        "expected_categories": ["bread", "bakery", "pastry", "roll"],
        "description": "Should return bread and bakery items"
    },
    {
        "name": "Meat Products",
        "query": "chicken beef pork meat",
        "max_results": 10,
        "expected_categories": ["chicken", "beef", "pork", "meat"],
        "description": "Should return various meat products"
    },
    {
        "name": "Chocolate and Sweets",
        "query": "chocolate candy sweets dessert",
        "max_results": 10,
        "expected_categories": ["chocolate", "candy", "sweet", "dessert"],
        "description": "Should return chocolate and confectionery products"
    },
    {
        "name": "Coffee and Tea",
        "query": "coffee beans tea leaves beverages",
        "max_results": 10,
        "expected_categories": ["coffee", "tea", "beverage", "drink"],
        "description": "Should return coffee and tea products"
    },
    {
        "name": "Breakfast Items",
        "query": "breakfast cereal oats granola morning",
        "max_results": 10,
        "expected_categories": ["breakfast", "cereal", "oats", "granola"],
        "description": "Should return breakfast and cereal products"
    },
    {
        "name": "Frozen Foods",
        "query": "frozen vegetables ice cream frozen meals",
        "max_results": 10,
        "expected_categories": ["frozen", "ice cream", "vegetables"],
        "description": "Should return frozen food products"
    },
    {
        "name": "Snacks and Chips",
        "query": "snacks chips crackers nuts",
        "max_results": 10,
        "expected_categories": ["snack", "chips", "crackers", "nuts"],
        "description": "Should return snack foods and chips"
    }
]


class MockContext:
    """Mock context object for testing rohlik_product_search function."""
    
    def __init__(self):
        self.request_context = Mock()
        self.request_context.lifespan_context = Mock()


class TestRohlikProductSearch:
    """Test suite for Rohlik product search functionality."""
    
    def setup_method(self):
        """Set up test environment before each test."""
        self.mock_ctx = MockContext()
        
        # Configure logging to see detailed output during tests
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            force=True  # Ensure this configuration takes precedence
        )
        
        # Set specific loggers to INFO level
        logging.getLogger('tools.rohlik_product_search').setLevel(logging.INFO)
        logging.getLogger('tests.test_rohlik_product_search').setLevel(logging.INFO)
        logging.getLogger(__name__).setLevel(logging.INFO)
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize the search instance
        self.search_instance = RohlikProductSearch()
        
        self.logger.info("=" * 80)
        self.logger.info("🧪 SETTING UP ROHLIK PRODUCT SEARCH TEST")
        self.logger.info(f"   Search available: {self.search_instance.is_available}")
        self.logger.info(f"   Index loaded: {self.search_instance.has_index}")
        if self.search_instance.has_index:
            self.logger.info(f"   Total products: {len(self.search_instance._products)}")
            self.logger.info(f"   FAISS index entries: {self.search_instance._faiss_index.ntotal}")
        self.logger.info("=" * 80)

    def test_search_instance_initialization(self):
        """Test that the search instance initializes correctly."""
        self.logger.info("🔧 Testing search instance initialization...")
        
        assert self.search_instance is not None
        self.logger.info(f"   ✅ Search instance created")
        
        # Check if we have the required dependencies
        try:
            import numpy as np
            import faiss
            from openai import OpenAI
            self.logger.info(f"   ✅ All dependencies available (numpy, faiss, openai)")
        except ImportError as e:
            self.logger.warning(f"   ⚠️ Missing dependency: {e}")
        
        # Check if OpenAI API key is available
        openai_key = os.environ.get("OPENAI_API_KEY")
        if openai_key:
            self.logger.info(f"   ✅ OpenAI API key available (length: {len(openai_key)})")
        else:
            self.logger.warning(f"   ⚠️ OpenAI API key not found in environment")

    def test_data_loading(self):
        """Test that product data and FAISS index load correctly."""
        self.logger.info("📂 Testing data loading...")
        
        # Check if data files exist
        data_dir = Path(__file__).parent.parent / "tools" / "rohlik_data"
        
        faiss_file = data_dir / "faiss_index.bin"
        products_json_file = data_dir / "products.json"
        products_pkl_file = data_dir / "products.pkl"
        
        self.logger.info(f"   Data directory: {data_dir}")
        self.logger.info(f"   FAISS index exists: {faiss_file.exists()}")
        self.logger.info(f"   Products JSON exists: {products_json_file.exists()}")
        self.logger.info(f"   Products PKL exists: {products_pkl_file.exists()}")
        
        if self.search_instance.has_index:
            self.logger.info(f"   ✅ Data loaded successfully")
            self.logger.info(f"   📊 Products count: {len(self.search_instance._products)}")
            self.logger.info(f"   📊 FAISS index size: {self.search_instance._faiss_index.ntotal}")
            
            # Show sample products
            if len(self.search_instance._products) > 0:
                self.logger.info("   📋 Sample products:")
                for i, product in enumerate(self.search_instance._products[:5]):
                    self.logger.info(f"      {i+1}. {product.name} ({product.brand})")
        else:
            self.logger.warning(f"   ⚠️ Data not loaded - check files and dependencies")

    @pytest.mark.parametrize("scenario", PRODUCT_SEARCH_SCENARIOS)
    def test_product_search_scenarios(self, scenario):
        """Test product search with various scenarios."""
        self.logger.info("=" * 80)
        self.logger.info(f"🔍 TESTING SCENARIO: {scenario['name']}")
        self.logger.info(f"   Query: '{scenario['query']}'")
        self.logger.info(f"   Expected categories: {scenario['expected_categories']}")
        self.logger.info(f"   Description: {scenario['description']}")
        self.logger.info("-" * 80)
        
        if not self.search_instance.is_available:
            pytest.skip("Search not available - check OpenAI API key and data files")
        
        try:
            # Call the product search function
            result_json = rohlik_product_search(
                ctx=self.mock_ctx,
                query=scenario['query'],
                max_results=scenario['max_results']
            )
            
            # Parse the JSON result
            result = json.loads(result_json)
            
            self.logger.info(f"   📊 SEARCH RESULTS:")
            self.logger.info(f"      Found products: {result['found_products']}")
            self.logger.info(f"      Query processed: '{result['query']}'")
            
            # Basic assertions
            assert 'found_products' in result
            assert 'products' in result
            assert 'query' in result
            assert result['query'] == scenario['query']
            
            products = result['products']
            
            if products:
                self.logger.info(f"   🛒 DETAILED PRODUCT RESULTS:")
                for i, product in enumerate(products, 1):
                    self.logger.info(f"      {i:2d}. [{product['score']:.3f}] {product['name']}")
                    self.logger.info(f"          Brand: {product['brand']}")
                    self.logger.info(f"          Amount: {product['textual_amount']} {product['unit']}")
                    if product['countries']:
                        self.logger.info(f"          Countries: {', '.join(product['countries'])}")
                    if product['badges']:
                        self.logger.info(f"          Badges: {', '.join(product['badges'])}")
                    self.logger.info(f"          Link: {product['link']}")
                    
                    # Show description preview
                    desc = product['description'][:100] + "..." if len(product['description']) > 100 else product['description']
                    if desc:
                        self.logger.info(f"          Description: {desc}")
                    self.logger.info("")
                
                # Analyze search quality
                self.logger.info(f"   🎯 SEARCH QUALITY ANALYSIS:")
                
                # Check for expected categories in results
                found_categories = []
                all_text = " ".join([
                    (p.get('name', '') or '').lower() + " " + 
                    (p.get('brand', '') or '').lower() + " " + 
                    (p.get('description', '') or '').lower()
                    for p in products
                ])
                
                for category in scenario['expected_categories']:
                    if category.lower() in all_text:
                        found_categories.append(category)
                
                self.logger.info(f"      Expected categories: {scenario['expected_categories']}")
                self.logger.info(f"      Found categories: {found_categories}")
                self.logger.info(f"      Category match rate: {len(found_categories)}/{len(scenario['expected_categories'])}")
                
                # Check score distribution
                scores = [p['score'] for p in products]
                if scores:
                    self.logger.info(f"      Score range: {min(scores):.3f} - {max(scores):.3f}")
                    self.logger.info(f"      Average score: {sum(scores)/len(scores):.3f}")
                
                # Verify we got some results
                assert len(products) > 0, f"No products found for query: {scenario['query']}"
                assert len(products) <= scenario['max_results'], f"Too many results returned"
                
                # Verify basic product structure
                for product in products:
                    assert 'id' in product
                    assert 'name' in product
                    assert 'brand' in product
                    assert 'score' in product
                    assert 'link' in product
                    assert isinstance(product['score'], (int, float))
                    assert product['score'] > 0, "Score should be positive"
            else:
                self.logger.warning(f"      ⚠️ No products found for query: {scenario['query']}")
                
        except Exception as e:
            self.logger.error(f"   ❌ Search failed: {e}")
            pytest.fail(f"Search failed for scenario '{scenario['name']}': {e}")
        
        self.logger.info("=" * 80)

    def test_empty_query(self):
        """Test behavior with empty query."""
        self.logger.info("🔍 Testing empty query handling...")
        
        result_json = rohlik_product_search(
            ctx=self.mock_ctx,
            query="",
            max_results=5
        )
        
        result = json.loads(result_json)
        
        self.logger.info(f"   Empty query result: {result}")
        
        assert 'error' in result
        assert result['found_products'] == 0
        assert len(result['products']) == 0

    def test_large_result_set(self):
        """Test requesting a large number of results."""
        self.logger.info("🔍 Testing large result set...")
        
        if not self.search_instance.is_available:
            pytest.skip("Search not available")
        
        result_json = rohlik_product_search(
            ctx=self.mock_ctx,
            query="food",
            max_results=50
        )
        
        result = json.loads(result_json)
        
        self.logger.info(f"   Large query result count: {result['found_products']}")
        self.logger.info(f"   Requested: 50, Got: {len(result['products'])}")
        
        assert result['found_products'] > 0
        assert len(result['products']) <= 50

    def test_specific_product_search(self):
        """Test searching for specific product types with detailed analysis."""
        specific_queries = [
            ("Chiquita banana", ["Chiquita", "banana"]),
            ("Lindt chocolate", ["Lindt", "chocolate"]),
            ("organic apple", ["organic", "apple"]),
            ("Czech bread", ["Czech", "bread"]),
            ("fresh milk", ["fresh", "milk"])
        ]
        
        for query, expected_terms in specific_queries:
            self.logger.info(f"🔍 Testing specific search: '{query}'")
            
            if not self.search_instance.is_available:
                self.logger.warning("   Skipping - search not available")
                continue
            
            result_json = rohlik_product_search(
                ctx=self.mock_ctx,
                query=query,
                max_results=5
            )
            
            result = json.loads(result_json)
            
            self.logger.info(f"   Results for '{query}': {result['found_products']}")
            
            if result['products']:
                # Show top result
                top_result = result['products'][0]
                self.logger.info(f"   Top result: {top_result['name']} ({top_result['brand']}) - Score: {top_result['score']:.3f}")
                
                # Check if expected terms appear in results
                found_terms = []
                result_text = " ".join([
                    (top_result.get('name', '') or '').lower(),
                    (top_result.get('brand', '') or '').lower(),
                    (top_result.get('description', '') or '').lower()
                ])
                
                for term in expected_terms:
                    if term.lower() in result_text:
                        found_terms.append(term)
                
                self.logger.info(f"   Expected terms found: {found_terms}/{expected_terms}")
            
            self.logger.info("")

    def test_product_structure_validation(self):
        """Test that returned products have correct structure and data types."""
        self.logger.info("🔍 Testing product structure validation...")
        
        if not self.search_instance.is_available:
            pytest.skip("Search not available")
        
        result_json = rohlik_product_search(
            ctx=self.mock_ctx,
            query="chocolate",
            max_results=3
        )
        
        result = json.loads(result_json)
        
        if result['products']:
            product = result['products'][0]
            
            self.logger.info(f"   Validating product structure...")
            self.logger.info(f"   Sample product: {product['name']}")
            
            # Required fields
            required_fields = ['id', 'name', 'brand', 'description', 'textual_amount', 'unit', 'link', 'score']
            for field in required_fields:
                assert field in product, f"Missing required field: {field}"
                self.logger.info(f"   ✅ {field}: {type(product[field]).__name__}")
            
            # Optional list fields
            list_fields = ['images', 'countries', 'badges']
            for field in list_fields:
                assert field in product, f"Missing field: {field}"
                assert isinstance(product[field], list), f"{field} should be a list"
                self.logger.info(f"   ✅ {field}: list with {len(product[field])} items")
            
            # Type validations
            assert isinstance(product['id'], int), "ID should be integer"
            assert isinstance(product['score'], (int, float)), "Score should be numeric"
            assert product['score'] > 0, "Score should be positive"
            
            # Link validation
            assert product['link'].startswith('https://www.rohlik.cz'), "Invalid link format"
            
            self.logger.info(f"   ✅ All validations passed")

if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s", "--tb=short"])
