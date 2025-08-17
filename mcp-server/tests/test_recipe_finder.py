"""Comprehensive tests for recipe finder with AI-powered evaluation."""

import csv
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import Mock, MagicMock, patch

import pytest
from openai import OpenAI

# Add the mcp-server directory and tests directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from test_data import TEST_SCENARIOS
from tools.recipe_finder import recipe_finder, init_semantic_index
from semantic_utils import SemanticRecipeSearch


def load_recipes_from_csv(csv_path: str) -> List[Dict[str, Any]]:
    """Load recipes from CSV file like the actual application does."""
    recipes = []
    try:
        with open(csv_path, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                # Convert CSV row to recipe format
                recipe = {
                    'id': row.get('id', ''),
                    'name': row.get('name', ''),
                    'ingredients': row.get('ingredients', ''),
                    'author_note': row.get('author_note', ''),
                    'steps': row.get('steps', '')
                }
                recipes.append(recipe)
        logging.info(f"Loaded {len(recipes)} recipes from {csv_path}")
        return recipes
    except Exception as e:
        logging.error(f"Failed to load recipes from {csv_path}: {e}")
        return []


class MockContext:
    """Mock context object for testing recipe_finder function."""
    
    def __init__(self, recipes: List[Dict[str, Any]]):
        self.request_context = Mock()
        self.request_context.lifespan_context = Mock()
        self.request_context.lifespan_context.recipes = recipes


class AIEvaluator:
    """Uses OpenAI to evaluate the quality and relevance of recipe finder results."""
    
    def __init__(self):
        self.client = self._init_openai_client()
        
    def _init_openai_client(self) -> Optional[OpenAI]:
        """Initialize OpenAI client for evaluation."""
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            logging.warning("OPENAI_API_KEY not set. AI evaluation will be skipped.")
            return None
        try:
            return OpenAI(api_key=api_key)
        except Exception as e:
            logging.error(f"Failed to initialize OpenAI client: {e}")
            return None
    
    def evaluate_results(
        self, 
        scenario: Dict[str, Any], 
        results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Evaluate recipe finder results using AI."""
        if not self.client:
            return {
                "score": 0.5,  # Neutral score when AI evaluation unavailable
                "reasoning": "AI evaluation not available - OPENAI_API_KEY not set",
                "criteria_scores": {},
                "recommendations": []
            }
        
        try:
            # Prepare evaluation prompt
            evaluation_prompt = self._create_evaluation_prompt(scenario, results)
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert culinary AI evaluator. Assess recipe search results based on relevance, quality, and adherence to user requirements. Provide detailed, objective analysis."
                    },
                    {
                        "role": "user",
                        "content": evaluation_prompt
                    }
                ],
                temperature=0.1,  # Low temperature for consistent evaluation
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            if content:
                return json.loads(content)
            else:
                raise ValueError("Empty response from OpenAI")
                
        except Exception as e:
            logging.error(f"AI evaluation failed: {e}")
            return {
                "score": 0.0,
                "reasoning": f"AI evaluation failed: {str(e)}",
                "criteria_scores": {},
                "recommendations": []
            }
    
    def _create_evaluation_prompt(
        self, 
        scenario: Dict[str, Any], 
        results: Dict[str, Any]
    ) -> str:
        """Create detailed evaluation prompt for OpenAI."""
        prompt = f"""
Please evaluate the following recipe search results based on how well they match the user's requirements.

**Search Request:**
- Dietary Goal: "{scenario['dietary_goal']}"
- Keywords: {scenario['keywords']}
- Max Results: {scenario['max_results']}
- Expected Themes: {scenario['expected_themes']}
- Description: {scenario['description']}

**Search Results:**
- Found Recipes: {results['found_recipes']}
- Recipes: {json.dumps(results['recipes'], indent=2)}

**Evaluation Criteria:**
1. **Relevance** (0-1): How well do the results match the dietary goal and keywords?
2. **Quality** (0-1): Are the recipes well-described with complete ingredients and steps?
3. **Diversity** (0-1): Do results show appropriate variety within the request scope?
4. **Accuracy** (0-1): Do results match expected themes and avoid irrelevant items?
5. **Completeness** (0-1): Are the right number of results returned with sufficient detail?

**Required Response Format (JSON):**
{{
    "overall_score": <float 0-1>,
    "reasoning": "<detailed explanation of the evaluation>",
    "criteria_scores": {{
        "relevance": <float 0-1>,
        "quality": <float 0-1>,
        "diversity": <float 0-1>,
        "accuracy": <float 0-1>,
        "completeness": <float 0-1>
    }},
    "recommendations": [
        "<specific recommendation 1>",
        "<specific recommendation 2>"
    ],
    "best_match": "<name of best matching recipe or null>",
    "concerns": [
        "<any quality concerns or issues>"
    ]
}}
"""
        return prompt


class TestRecipeFinder:
    """Test suite for recipe finder functionality."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test environment."""
        self.ai_evaluator = AIEvaluator()
        
        # Load real recipes from CSV file
        csv_path = Path(__file__).parent.parent / "recipes.csv"
        self.recipes = load_recipes_from_csv(str(csv_path))
        
        if not self.recipes:
            pytest.skip("No recipes loaded from CSV file - cannot run tests")
        
        # Reset the semantic search instance before each test
        from tools.recipe_finder import semantic_search
        semantic_search._faiss_index = None
        semantic_search._recipe_ids = []
        semantic_search._recipe_texts = []
        
        # Clear any cached index files to ensure fresh index build
        cache_dir = Path(__file__).parent.parent / ".cache"
        if cache_dir.exists():
            import shutil
            shutil.rmtree(cache_dir, ignore_errors=True)
    
    @pytest.fixture
    def mock_openai_client(self):
        """Mock OpenAI client for controlled testing."""
        with patch('semantic_utils.OpenAI') as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client
            
            # Mock embeddings response
            mock_embeddings_response = Mock()
            mock_embeddings_response.data = [Mock(embedding=[0.1] * 1536)]
            mock_client.embeddings.create.return_value = mock_embeddings_response
            
            # Mock chat completion response for reranking
            # Use actual recipe IDs from the loaded data
            recipe_ids = [str(recipe['id']) for recipe in self.recipes[:3]] if hasattr(self, 'recipes') and self.recipes else ["10", "11", "12"]
            mock_completion_response = Mock()
            mock_completion_response.choices = [Mock()]
            mock_completion_response.choices[0].message.content = f'{{"selected_ids": {json.dumps(recipe_ids)}}}'
            mock_client.chat.completions.create.return_value = mock_completion_response
            
            yield mock_client
    
    def test_recipe_finder_basic_functionality(self, mock_openai_client):
        """Test basic recipe finder functionality with mocked OpenAI."""
        # Initialize semantic index with real CSV data
        init_semantic_index(self.recipes)
        
        # Create mock context with real data
        ctx = MockContext(self.recipes)
        
        # Test basic search
        result_json = recipe_finder(
            ctx=ctx,
            dietary_goal="comfort food",
            keywords=["hearty", "warming"],
            max_results=3
        )
        
        # Parse and validate result
        result = json.loads(result_json)
        assert isinstance(result, dict)
        assert "found_recipes" in result
        assert "recipes" in result
        assert isinstance(result["recipes"], list)
        assert result["found_recipes"] == len(result["recipes"])
        assert result["found_recipes"] <= 3
        
        # Validate recipe structure
        for recipe in result["recipes"]:
            assert "id" in recipe
            assert "name" in recipe
            assert "ingredients" in recipe
            assert "author_note" in recipe
            assert "steps" in recipe
    
    def test_recipe_finder_empty_parameters(self, mock_openai_client):
        """Test recipe finder with empty parameters."""
        init_semantic_index(self.recipes)
        ctx = MockContext(self.recipes)
        
        result_json = recipe_finder(
            ctx=ctx,
            dietary_goal="",
            keywords=None,
            max_results=5
        )
        
        result = json.loads(result_json)
        assert result["found_recipes"] >= 0
        assert len(result["recipes"]) <= 5
    
    def test_recipe_finder_no_index(self):
        """Test recipe finder behavior when semantic index is not available."""
        # Don't initialize the index
        ctx = MockContext(self.recipes)
        
        result_json = recipe_finder(
            ctx=ctx,
            dietary_goal="any food",
            keywords=["test"],
            max_results=3
        )
        
        result = json.loads(result_json)
        assert result["found_recipes"] == 0
        assert result["recipes"] == []
    
    def test_recipe_finder_max_results_limits(self, mock_openai_client):
        """Test that max_results parameter is respected."""
        init_semantic_index(self.recipes)
        ctx = MockContext(self.recipes)
        
        for max_results in [1, 3, 5, 10]:
            result_json = recipe_finder(
                ctx=ctx,
                dietary_goal="any recipe",
                keywords=[],
                max_results=max_results
            )
            
            result = json.loads(result_json)
            assert result["found_recipes"] <= max_results
            assert len(result["recipes"]) <= max_results
    
    @pytest.mark.parametrize("scenario", TEST_SCENARIOS)
    def test_recipe_finder_scenarios(self, scenario, mock_openai_client):
        """Test recipe finder with various realistic scenarios."""
        init_semantic_index(self.recipes)
        ctx = MockContext(self.recipes)
        
        result_json = recipe_finder(
            ctx=ctx,
            dietary_goal=scenario["dietary_goal"],
            keywords=scenario["keywords"],
            max_results=scenario["max_results"]
        )
        
        result = json.loads(result_json)
        
        # Basic validation
        assert isinstance(result, dict)
        assert result["found_recipes"] <= scenario["max_results"]
        assert len(result["recipes"]) == result["found_recipes"]
        
        # AI-powered evaluation
        evaluation = self.ai_evaluator.evaluate_results(scenario, result)
        
        # Log evaluation results
        print(f"\n=== Scenario: {scenario['name']} ===")
        print(f"Query: {scenario['dietary_goal']}")
        print(f"Keywords: {scenario['keywords']}")
        print(f"Results found: {result['found_recipes']}")
        print(f"AI Evaluation Score: {evaluation.get('overall_score', 'N/A')}")
        print(f"Reasoning: {evaluation.get('reasoning', 'N/A')}")
        
        if evaluation.get('criteria_scores'):
            print("Criteria Scores:")
            for criterion, score in evaluation['criteria_scores'].items():
                print(f"  {criterion}: {score}")
        
        if evaluation.get('recommendations'):
            print("Recommendations:")
            for rec in evaluation['recommendations']:
                print(f"  - {rec}")
        
        # Assert minimum quality threshold (can be adjusted based on requirements)
        if evaluation.get('overall_score') is not None:
            assert evaluation['overall_score'] >= 0.3, f"AI evaluation score too low: {evaluation['overall_score']}"
    
    def test_semantic_search_integration(self, mock_openai_client):
        """Test integration with SemanticRecipeSearch class."""
        semantic_search = SemanticRecipeSearch()
        
        # Test availability check
        # Note: This might fail in CI without proper dependencies
        # assert semantic_search.is_available or not semantic_search.is_available  # Either is fine for testing
        
        # Test index building with real data (use subset for faster testing)
        result_count = semantic_search.build_index(self.recipes[:5])
        
        if semantic_search.is_available:
            assert result_count == 5
            assert semantic_search.has_index
        else:
            assert result_count == 0
            assert not semantic_search.has_index
    
    def test_recipe_steps_truncation(self, mock_openai_client):
        """Test that long recipe steps are properly truncated."""
        # Create a recipe with very long steps
        long_recipe = {
            "id": "9999",
            "name": "Test Recipe",
            "ingredients": "Test ingredients",
            "author_note": "Test note",
            "steps": "This is a very long recipe step. " * 20  # Much longer than 200 chars
        }
        
        test_recipes = self.recipes + [long_recipe]
        init_semantic_index(test_recipes)
        ctx = MockContext(test_recipes)
        
        result_json = recipe_finder(
            ctx=ctx,
            dietary_goal="test recipe",
            keywords=["test"],
            max_results=10
        )
        
        result = json.loads(result_json)
        
        # Check if any recipe has truncated steps
        for recipe in result["recipes"]:
            if recipe["id"] == "9999":
                assert len(recipe["steps"]) <= 203  # 200 chars + "..."
                assert recipe["steps"].endswith("...")
    
    def test_json_output_format(self, mock_openai_client):
        """Test that output is valid JSON with correct structure."""
        init_semantic_index(self.recipes)
        ctx = MockContext(self.recipes)
        
        result_json = recipe_finder(
            ctx=ctx,
            dietary_goal="any recipe",
            keywords=["test"],
            max_results=2
        )
        
        # Should be valid JSON
        result = json.loads(result_json)
        
        # Should have correct top-level structure
        assert set(result.keys()) == {"found_recipes", "recipes"}
        assert isinstance(result["found_recipes"], int)
        assert isinstance(result["recipes"], list)
        
        # Each recipe should have correct structure
        required_fields = {"id", "name", "ingredients", "author_note", "steps"}
        for recipe in result["recipes"]:
            assert set(recipe.keys()) == required_fields
            for field in required_fields:
                assert isinstance(recipe[field], str)


class TestAIEvaluator:
    """Test suite for AI-powered evaluation functionality."""
    
    def test_ai_evaluator_initialization(self):
        """Test AI evaluator initialization."""
        evaluator = AIEvaluator()
        
        # Should handle missing API key gracefully
        if os.environ.get("OPENAI_API_KEY"):
            assert evaluator.client is not None
        else:
            assert evaluator.client is None
    
    def test_ai_evaluator_fallback(self):
        """Test AI evaluator fallback when OpenAI is unavailable."""
        evaluator = AIEvaluator()
        evaluator.client = None  # Force unavailable state
        
        scenario = TEST_SCENARIOS[0]
        results = {"found_recipes": 2, "recipes": [{"name": "Test Recipe"}]}
        
        evaluation = evaluator.evaluate_results(scenario, results)
        
        assert evaluation["score"] == 0.5  # Neutral fallback score
        assert "not available" in evaluation["reasoning"]


if __name__ == "__main__":
    # Configure logging for test runs
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s", "--tb=short"])
