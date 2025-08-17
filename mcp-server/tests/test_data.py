"""Test scenarios and configuration for recipe finder tests."""

from typing import Any, Dict, List

# Test scenarios with expected behavior descriptions for real CSV data
TEST_SCENARIOS = [
    {
        "name": "Comfort Food Search",
        "dietary_goal": "comfort food for cold weather",
        "keywords": ["hearty", "warming", "soup"],
        "max_results": 3,
        "expected_themes": ["guláš", "polévka", "warm dishes"],
        "description": "Should return hearty, warming comfort foods like Czech goulash and soups"
    },
    {
        "name": "Healthy Vegetarian",
        "dietary_goal": "healthy vegetarian meals",
        "keywords": ["vegetables", "fresh", "light"],
        "max_results": 4,
        "expected_themes": ["salad", "vegetarian", "fresh ingredients", "vegetables"],
        "description": "Should prioritize vegetarian dishes with fresh ingredients"
    },
    {
        "name": "Quick Dinner",
        "dietary_goal": "quick and easy dinner for busy weeknight",
        "keywords": ["simple", "fast", "easy"],
        "max_results": 3,
        "expected_themes": ["grilled", "simple preparation", "quick cooking"],
        "description": "Should return recipes with simple preparation methods"
    },
    {
        "name": "Traditional Czech",
        "dietary_goal": "traditional Czech cuisine",
        "keywords": ["Czech", "traditional", "classic"],
        "max_results": 5,
        "expected_themes": ["guláš", "knedlík", "Czech", "traditional"],
        "description": "Should return authentic Czech dishes with traditional ingredients"
    },
    {
        "name": "Soup Recipes",
        "dietary_goal": "warm soups for dinner",
        "keywords": ["soup", "broth", "warm"],
        "max_results": 3,
        "expected_themes": ["polévka", "soup", "broth"],
        "description": "Should return various soup recipes"
    },
    {
        "name": "Dessert Options",
        "dietary_goal": "sweet desserts and baked goods",
        "keywords": ["sweet", "dessert", "baking"],
        "max_results": 3,
        "expected_themes": ["bábovka", "sweet", "dessert"],
        "description": "Should return dessert and baking recipes"
    }
]
