"""
Inspired by satlaspretrain_models/utils.py, adjusted to use ImageNet weights.
"""

from enum import Enum, auto


class Backbone(Enum):
    SWINB = auto()
    SWINT = auto()
    RESNET50 = auto()
    RESNET152 = auto()


class Head(Enum):
    CLASSIFY = auto()
    MULTICLASSIFY = auto()
    DETECT = auto()
    INSTANCE = auto()
    SEGMENT = auto()
    BINSEGMENT = auto()
    REGRESS = auto()


def adjust_state_dict_prefix(
    state_dict, needed, prefix=None, prefix_allowed_count=None
):
    """
    Adjusts the keys in the state dictionary by replacing 'backbone.backbone' prefix with 'backbone'.

    Args:
        state_dict (dict): Original state dictionary with 'backbone.backbone' prefixes.

    Returns:
        dict: Modified state dictionary with corrected prefixes.
    """
    new_state_dict = {}
    for key, value in state_dict.items():
        # Assure we're only keeping keys that we need for the current model component.
        if needed not in key:
            continue

        # Update the key prefixes to match what the model expects.
        if prefix is not None:
            while key.count(prefix) > prefix_allowed_count:
                key = key.replace(prefix, "", 1)

        new_state_dict[key] = value
    return new_state_dict
