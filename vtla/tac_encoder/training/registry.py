"""Training recipe registry used by the unified entrypoint."""

from .anytouch1 import RECIPE as ANYTOUCH1_RECIPE
from .anytouch2 import RECIPE as ANYTOUCH2_RECIPE
from .sparsh_vjepa import RECIPE as SPARSH_VJEPA_RECIPE


TRAINING_RECIPES = {
    ANYTOUCH1_RECIPE.model_id: ANYTOUCH1_RECIPE,
    ANYTOUCH2_RECIPE.model_id: ANYTOUCH2_RECIPE,
    SPARSH_VJEPA_RECIPE.model_id: SPARSH_VJEPA_RECIPE,
}


def get_training_recipe(model_id: str):
    try:
        return TRAINING_RECIPES[model_id]
    except KeyError as error:
        raise ValueError(
            f"No training recipe registered for {model_id!r}; expected {sorted(TRAINING_RECIPES)}"
        ) from error
