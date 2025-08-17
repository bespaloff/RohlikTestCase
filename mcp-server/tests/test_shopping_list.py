"""Comprehensive tests for shopping list functionality."""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import Mock

import pytest

# Add the mcp-server directory and tests directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from tools.shopping_list import shopping_list_manager


class MockContext:
    """Mock context object for testing shopping_list_manager function."""
    
    def __init__(self, shopping_lists: Dict[str, List[Dict[str, Any]]] = None):
        self.request_context = Mock()
        self.request_context.lifespan_context = Mock()
        self.request_context.lifespan_context.shopping_lists = shopping_lists or {}


class TestShoppingListManager:
    """Test suite for shopping list manager functionality."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test environment."""
        self.ctx = MockContext()
    
    def test_create_new_list(self):
        """Test creating a new shopping list."""
        result_json = shopping_list_manager(
            ctx=self.ctx,
            action="create",
            list_id="weekly_shopping",
            ingredients=None
        )
        
        result = json.loads(result_json)
        
        assert result["status"] == "success"
        assert "weekly_shopping" in result["message"]
        assert "weekly_shopping" in self.ctx.request_context.lifespan_context.shopping_lists
        assert self.ctx.request_context.lifespan_context.shopping_lists["weekly_shopping"] == []
    
    def test_add_ingredients_to_new_list(self):
        """Test adding ingredients to a new list (auto-creates list)."""
        ingredients = [
            {"name": "Beef chuck roast", "quantity": "2 lbs", "recipe_id": "1001"},
            {"name": "Carrots", "quantity": "4 large", "recipe_id": "1001"},
            {"name": "Potatoes", "quantity": "6 medium", "recipe_id": "1001"}
        ]
        
        result_json = shopping_list_manager(
            ctx=self.ctx,
            action="add",
            list_id="beef_stew_list",
            ingredients=ingredients
        )
        
        result = json.loads(result_json)
        
        assert result["status"] == "success"
        assert result["total_items"] == 3
        assert "beef_stew_list" in self.ctx.request_context.lifespan_context.shopping_lists
        assert len(self.ctx.request_context.lifespan_context.shopping_lists["beef_stew_list"]) == 3
    
    def test_add_ingredients_to_existing_list(self):
        """Test adding ingredients to an existing list."""
        # Create list first
        self.ctx.request_context.lifespan_context.shopping_lists["existing_list"] = [
            {"name": "Onions", "quantity": "2 medium", "recipe_id": "1002"}
        ]
        
        new_ingredients = [
            {"name": "Garlic", "quantity": "3 cloves", "recipe_id": "1002"},
            {"name": "Tomatoes", "quantity": "4 large", "recipe_id": "1002"}
        ]
        
        result_json = shopping_list_manager(
            ctx=self.ctx,
            action="add",
            list_id="existing_list",
            ingredients=new_ingredients
        )
        
        result = json.loads(result_json)
        
        assert result["status"] == "success"
        assert result["total_items"] == 3  # 1 existing + 2 new
        assert len(self.ctx.request_context.lifespan_context.shopping_lists["existing_list"]) == 3
    
    def test_get_empty_list(self):
        """Test getting an empty shopping list."""
        result_json = shopping_list_manager(
            ctx=self.ctx,
            action="get",
            list_id="nonexistent_list",
            ingredients=None
        )
        
        result = json.loads(result_json)
        
        assert result["list_id"] == "nonexistent_list"
        assert result["items"] == []
        assert result["total_items"] == 0
    
    def test_get_populated_list(self):
        """Test getting a populated shopping list."""
        # Set up a list with items
        self.ctx.request_context.lifespan_context.shopping_lists["test_list"] = [
            {"name": "Flour", "quantity": "2 cups", "recipe_id": "1001"},
            {"name": "Sugar", "quantity": "1 cup", "recipe_id": "1001"},
            {"name": "Flour", "quantity": "1 cup", "recipe_id": "1002"},  # Duplicate item
            {"name": "Eggs", "quantity": "3 large", "recipe_id": "1002"}
        ]
        
        result_json = shopping_list_manager(
            ctx=self.ctx,
            action="get",
            list_id="test_list",
            ingredients=None
        )
        
        result = json.loads(result_json)
        
        assert result["list_id"] == "test_list"
        assert result["total_items"] == 3  # Unique items: Flour, Sugar, Eggs
        assert len(result["items"]) == 3
        
        # Check that flour appears in both recipes
        flour_item = next(item for item in result["items"] if item["name"] == "Flour")
        assert len(flour_item["recipes"]) == 2
        assert "1001" in flour_item["recipes"]
        assert "1002" in flour_item["recipes"]
    
    def test_clear_list(self):
        """Test clearing a shopping list."""
        # Set up a list with items
        self.ctx.request_context.lifespan_context.shopping_lists["clear_test"] = [
            {"name": "Item1", "quantity": "1", "recipe_id": "1001"},
            {"name": "Item2", "quantity": "2", "recipe_id": "1001"}
        ]
        
        result_json = shopping_list_manager(
            ctx=self.ctx,
            action="clear",
            list_id="clear_test",
            ingredients=None
        )
        
        result = json.loads(result_json)
        
        assert result["status"] == "success"
        assert "clear_test" in result["message"]
        assert self.ctx.request_context.lifespan_context.shopping_lists["clear_test"] == []
    
    def test_unknown_action(self):
        """Test handling of unknown action."""
        result_json = shopping_list_manager(
            ctx=self.ctx,
            action="invalid_action",
            list_id="test_list",
            ingredients=None
        )
        
        result = json.loads(result_json)
        
        assert "error" in result
        assert "Unknown action" in result["error"]
        assert "invalid_action" in result["error"]
    
    def test_add_empty_ingredients_list(self):
        """Test adding an empty ingredients list."""
        result_json = shopping_list_manager(
            ctx=self.ctx,
            action="add",
            list_id="empty_add",
            ingredients=[]
        )
        
        result = json.loads(result_json)
        
        assert result["status"] == "success"
        assert result["total_items"] == 0
        assert "empty_add" in self.ctx.request_context.lifespan_context.shopping_lists
    
    def test_add_none_ingredients(self):
        """Test adding with None ingredients parameter."""
        result_json = shopping_list_manager(
            ctx=self.ctx,
            action="add",
            list_id="none_add",
            ingredients=None
        )
        
        result = json.loads(result_json)
        
        assert result["status"] == "success"
        assert result["total_items"] == 0
        assert "none_add" in self.ctx.request_context.lifespan_context.shopping_lists
    
    def test_ingredient_consolidation(self):
        """Test that ingredients with same name are properly consolidated."""
        ingredients = [
            {"name": "Salt", "quantity": "1 tsp", "recipe_id": "recipe1"},
            {"name": "Pepper", "quantity": "1/2 tsp", "recipe_id": "recipe1"},
            {"name": "Salt", "quantity": "2 tsp", "recipe_id": "recipe2"},
            {"name": "Onion", "quantity": "1 medium", "recipe_id": "recipe2"}
        ]
        
        # Add ingredients
        shopping_list_manager(
            ctx=self.ctx,
            action="add",
            list_id="consolidation_test",
            ingredients=ingredients
        )
        
        # Get the list to check consolidation
        result_json = shopping_list_manager(
            ctx=self.ctx,
            action="get",
            list_id="consolidation_test",
            ingredients=None
        )
        
        result = json.loads(result_json)
        
        assert result["total_items"] == 3  # Salt, Pepper, Onion (Salt consolidated)
        
        # Find salt item and verify it's linked to both recipes
        salt_item = next(item for item in result["items"] if item["name"] == "Salt")
        assert len(salt_item["recipes"]) == 2
        assert "recipe1" in salt_item["recipes"]
        assert "recipe2" in salt_item["recipes"]
    
    def test_multiple_lists_independence(self):
        """Test that multiple shopping lists are independent."""
        # Create and populate first list
        shopping_list_manager(
            ctx=self.ctx,
            action="create",
            list_id="list1",
            ingredients=None
        )
        
        shopping_list_manager(
            ctx=self.ctx,
            action="add",
            list_id="list1",
            ingredients=[{"name": "Item1", "quantity": "1", "recipe_id": "1001"}]
        )
        
        # Create and populate second list
        shopping_list_manager(
            ctx=self.ctx,
            action="create",
            list_id="list2",
            ingredients=None
        )
        
        shopping_list_manager(
            ctx=self.ctx,
            action="add",
            list_id="list2",
            ingredients=[{"name": "Item2", "quantity": "2", "recipe_id": "1002"}]
        )
        
        # Check that lists are independent
        result1_json = shopping_list_manager(
            ctx=self.ctx,
            action="get",
            list_id="list1",
            ingredients=None
        )
        
        result2_json = shopping_list_manager(
            ctx=self.ctx,
            action="get",
            list_id="list2",
            ingredients=None
        )
        
        result1 = json.loads(result1_json)
        result2 = json.loads(result2_json)
        
        assert result1["total_items"] == 1
        assert result2["total_items"] == 1
        assert result1["items"][0]["name"] == "Item1"
        assert result2["items"][0]["name"] == "Item2"
    
    def test_json_output_format(self):
        """Test that all outputs are valid JSON with correct structure."""
        actions_to_test = [
            ("create", "test_format", None),
            ("add", "test_format", [{"name": "Test", "quantity": "1", "recipe_id": "1001"}]),
            ("get", "test_format", None),
            ("clear", "test_format", None),
            ("invalid", "test_format", None)
        ]
        
        for action, list_id, ingredients in actions_to_test:
            result_json = shopping_list_manager(
                ctx=self.ctx,
                action=action,
                list_id=list_id,
                ingredients=ingredients
            )
            
            # Should be valid JSON
            result = json.loads(result_json)
            assert isinstance(result, dict)
            
            # Should have appropriate structure based on action
            if action in ["create", "add", "clear"]:
                if action != "invalid":
                    assert "status" in result
                    assert "message" in result
                else:
                    assert "error" in result
            elif action == "get":
                assert "list_id" in result
                assert "items" in result
                assert "total_items" in result
                assert isinstance(result["items"], list)
                assert isinstance(result["total_items"], int)
    
    def test_ingredient_structure_handling(self):
        """Test handling of different ingredient structures."""
        # Test with complete ingredient structure
        complete_ingredient = {
            "name": "Complete Ingredient",
            "quantity": "1 cup",
            "recipe_id": "recipe123"
        }
        
        # Test with minimal ingredient structure
        minimal_ingredient = {
            "name": "Minimal Ingredient"
        }
        
        # Test with extra fields (should be ignored)
        extra_fields_ingredient = {
            "name": "Extra Fields Ingredient",
            "quantity": "2 tbsp",
            "recipe_id": "recipe456",
            "extra_field": "should be ignored",
            "another_extra": 123
        }
        
        ingredients = [complete_ingredient, minimal_ingredient, extra_fields_ingredient]
        
        # Add ingredients
        shopping_list_manager(
            ctx=self.ctx,
            action="add",
            list_id="structure_test",
            ingredients=ingredients
        )
        
        # Get the list
        result_json = shopping_list_manager(
            ctx=self.ctx,
            action="get",
            list_id="structure_test",
            ingredients=None
        )
        
        result = json.loads(result_json)
        
        assert result["total_items"] == 3
        
        # Verify all ingredients are present
        names = [item["name"] for item in result["items"]]
        assert "Complete Ingredient" in names
        assert "Minimal Ingredient" in names
        assert "Extra Fields Ingredient" in names
    
    def test_ingredient_link_field_handling(self):
        """Test handling of link field in ingredients."""
        # Test ingredients with links (like from Rohlik product search)
        ingredients_with_links = [
            {
                "name": "Milk",
                "quantity": "1 liter",
                "recipe_id": "recipe1",
                "link": "https://www.rohlik.cz/en-CZ/123-milk"
            },
            {
                "name": "Bread",
                "quantity": "1 loaf",
                "recipe_id": "recipe1",
                "link": "https://www.rohlik.cz/en-CZ/456-bread"
            },
            {
                "name": "Eggs",
                "quantity": "12 pieces",
                "recipe_id": "recipe1"
                # No link field
            }
        ]
        
        # Add ingredients
        shopping_list_manager(
            ctx=self.ctx,
            action="add",
            list_id="link_test",
            ingredients=ingredients_with_links
        )
        
        # Get the list
        result_json = shopping_list_manager(
            ctx=self.ctx,
            action="get",
            list_id="link_test",
            ingredients=None
        )
        
        result = json.loads(result_json)
        
        assert result["total_items"] == 3
        
        # Find items and verify link handling
        items_by_name = {item["name"]: item for item in result["items"]}
        
        # Milk should have link
        milk_item = items_by_name["Milk"]
        assert "link" in milk_item
        assert milk_item["link"] == "https://www.rohlik.cz/en-CZ/123-milk"
        
        # Bread should have link
        bread_item = items_by_name["Bread"]
        assert "link" in bread_item
        assert bread_item["link"] == "https://www.rohlik.cz/en-CZ/456-bread"
        
        # Eggs should not have link field (or link should be None/empty)
        eggs_item = items_by_name["Eggs"]
        assert "link" not in eggs_item or eggs_item.get("link") is None
    
    def test_duplicate_items_with_different_links(self):
        """Test consolidation of duplicate items with different links."""
        # Add same ingredient from different sources with different links
        ingredients_batch1 = [
            {
                "name": "Tomatoes",
                "quantity": "500g",
                "recipe_id": "recipe1",
                "link": "https://www.rohlik.cz/en-CZ/111-tomatoes"
            }
        ]
        
        ingredients_batch2 = [
            {
                "name": "Tomatoes",
                "quantity": "300g", 
                "recipe_id": "recipe2"
                # No link
            }
        ]
        
        # Add first batch
        shopping_list_manager(
            ctx=self.ctx,
            action="add",
            list_id="duplicate_link_test",
            ingredients=ingredients_batch1
        )
        
        # Add second batch (same item, no link)
        shopping_list_manager(
            ctx=self.ctx,
            action="add",
            list_id="duplicate_link_test",
            ingredients=ingredients_batch2
        )
        
        # Get the list
        result_json = shopping_list_manager(
            ctx=self.ctx,
            action="get",
            list_id="duplicate_link_test",
            ingredients=None
        )
        
        result = json.loads(result_json)
        
        assert result["total_items"] == 1  # Should be consolidated into one item
        
        tomatoes_item = result["items"][0]
        assert tomatoes_item["name"] == "Tomatoes"
        assert len(tomatoes_item["recipes"]) == 2  # From both recipes
        assert "recipe1" in tomatoes_item["recipes"]
        assert "recipe2" in tomatoes_item["recipes"]
        
        # Should keep the link from the first item
        assert "link" in tomatoes_item
        assert tomatoes_item["link"] == "https://www.rohlik.cz/en-CZ/111-tomatoes"