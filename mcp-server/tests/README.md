# MCP Server Tests

Comprehensive test suite for the MCP server functionality including recipe finder and shopping list features.

## Quick Start

```bash
# Install dependencies
uv sync

# Set OpenAI API key (required for Rohlik product search tests)
export OPENAI_API_KEY="your-api-key-here"

# Note: Rohlik product search tests require pre-built FAISS index and product data
# Files should be present in tools/rohlik_data/:
# - faiss_index.bin (FAISS index with 43,331 embeddings)
# - products.json (43,631 products in JSON format)
# - products.pkl (same products in pickle format)

# Run all tests
uv run pytest tests/ -v

# Run specific test files
uv run pytest tests/test_recipe_finder.py -v
uv run pytest tests/test_shopping_list.py -v
uv run pytest tests/test_rohlik_product_search.py -v

# Run with detailed INFO logs visible (recommended for Rohlik tests)
uv run pytest tests/test_rohlik_product_search.py -v -s --log-cli-level=INFO
```

## Test Coverage

### Recipe Finder Tests (15 tests)
- **Load real data**: Uses actual recipes from `recipes.csv` (100 Czech recipes)
- **Build FAISS index**: Creates semantic search index with real embeddings
- **Test search scenarios**: 6 different search scenarios (comfort food, vegetarian, Czech cuisine, etc.)
- **AI evaluation**: Uses OpenAI to evaluate search result quality (optional)
- **Mock OpenAI for consistency**: Tests use mocked OpenAI client for reliable results

### Shopping List Tests (13 tests)
- **List management**: Create, add, get, clear shopping lists
- **Ingredient handling**: Add ingredients with quantities and recipe references
- **Consolidation**: Merge duplicate ingredients from multiple recipes
- **Multiple lists**: Independent management of multiple shopping lists
- **JSON format**: Validate proper JSON output structure
- **Error handling**: Test invalid actions and edge cases

### Rohlik Product Search Tests (16 tests)
- **Search functionality**: Semantic search through 43,631 Rohlik.cz products
- **FAISS integration**: Uses pre-built FAISS index with 43,331 embeddings
- **Product scenarios**: 10 comprehensive search scenarios (fruits, dairy, chocolate, etc.)
- **Detailed results**: Returns 10 products per search with full details (name, brand, price, links)
- **Quality analysis**: Category matching, score distribution, relevance validation
- **Data validation**: Product structure validation and error handling
- **Multiple formats**: Supports both pickle and JSON product data loading

## Test Results

All 44 tests should pass:
- **Recipe Finder**: 15 tests covering search functionality, AI evaluation, error handling
- **Shopping List**: 13 tests covering list operations, ingredient management, data validation
- **Rohlik Product Search**: 16 tests covering semantic product search, data loading, quality validation

## Verbose Test Output

For detailed output showing search results:

```bash
# Run Rohlik product search with verbose logging
cd tests && python -c "
from test_rohlik_product_search import TestRohlikProductSearch
test = TestRohlikProductSearch()
test.setup_method()

# Run chocolate search scenario
scenario = {
    'name': 'Chocolate Search',
    'query': 'chocolate candy sweets dessert',
    'max_results': 10,
    'expected_categories': ['chocolate', 'candy', 'sweet', 'dessert'],
    'description': 'Chocolate and confectionery products'
}
test.test_product_search_scenarios(scenario)
"

# Run specific test scenarios with pytest
python -m pytest 'test_rohlik_product_search.py::TestRohlikProductSearch::test_product_search_scenarios[scenario5]' -v -s
```

The tests validate that the MCP server correctly:
- Loads and searches real recipe data (100 Czech recipes)
- Manages shopping lists with proper consolidation
- Searches through 43,631 Rohlik.cz products using semantic AI
- Handles edge cases and errors gracefully
- Returns properly formatted JSON responses