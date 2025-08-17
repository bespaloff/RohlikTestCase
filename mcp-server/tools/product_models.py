"""
Shared data models for Rohlik product data.
This module contains the Product class that can be used across different scripts
to avoid pickle import issues.
"""

import re
from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class Product:
    """Product data structure"""
    id: int
    name: str
    slug: str
    brand: str
    description: str
    textual_amount: str
    unit: str
    main_category_id: int
    images: List[str]
    countries: List[Dict[str, str]]
    badges: List[Dict[str, str]]
    filters: List[Dict[str, Any]]
    information: List[Dict[str, str]]

    def clean_description(self) -> str:
        """Clean HTML from description"""
        if not self.description:
            return ""
        
        # Simple HTML tag removal
        text = re.sub(r'<[^>]+>', '', self.description)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def to_text(self) -> str:
        """Convert product to searchable text"""
        text_parts = [
            f"Name: {self.name}",
            f"Brand: {self.brand}",
            f"Amount: {self.textual_amount}",
            f"Description: {self.clean_description()}"
        ]
        
        # Add country information
        if self.countries:
            countries = ", ".join([c.get('name', '') for c in self.countries])
            text_parts.append(f"Country: {countries}")
        
        # Add badges
        if self.badges:
            badges = ", ".join([b.get('title', '') for b in self.badges])
            text_parts.append(f"Badges: {badges}")
            
        # Add additional information
        if self.information:
            info = "; ".join([f"{i.get('name', '')}: {i.get('value', '')}" for i in self.information])
            text_parts.append(f"Information: {info}")
        
        return "\n".join(text_parts)
