"""Backbone-specific training recipes behind one CLI entrypoint."""

from .registry import get_training_recipe

__all__ = ["get_training_recipe"]
