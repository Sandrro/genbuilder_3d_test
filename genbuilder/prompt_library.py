"""Utilities for reading the shared texture prompt library."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping

import yaml


@dataclass
class PromptLibrary:
    """Wrapper around the YAML prompt library.

    The library is stored as a nested mapping. Recipes are high-level material
    sets that reference individual presets defined under the ``presets``
    section of the YAML file.
    """

    data: Dict[str, Any]

    @classmethod
    def from_file(cls, path: Path) -> "PromptLibrary":
        content = path.read_text(encoding="utf-8")
        data: Dict[str, Any] = yaml.safe_load(content)
        if not isinstance(data, Mapping):
            raise ValueError("Prompt library root must be a mapping")
        return cls(dict(data))

    def has_recipe(self, name: str) -> bool:
        return name in self.data.get("recipes", {})

    def recipe_names(self) -> list[str]:
        return list(self.data.get("recipes", {}).keys())

    def get_recipe(self, name: str) -> Dict[str, Any]:
        recipes = self.data.get("recipes", {})
        if name not in recipes:
            raise KeyError(f"Recipe '{name}' not found in prompt library")
        recipe = recipes[name]
        if not isinstance(recipe, Mapping):
            raise ValueError(f"Recipe '{name}' must be a mapping")
        return dict(recipe)

    def default_recipe(self) -> str:
        names = self.recipe_names()
        if not names:
            raise ValueError("Prompt library does not define any recipes")
        return names[0]
