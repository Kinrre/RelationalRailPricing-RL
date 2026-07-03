"""Constants for the supply module."""

DEFAULT_LIFT_CONSTRAINTS = 1
"""Default number of anticipation days to lift capacity constraints."""

MAX_DEPTH = 3
"""Maximum depth (services) for the journey search algorithm."""

EDGE_TYPE_SAME_MARKET = 0
"""Edge type for services competing in the same market (same origin-destination)."""

EDGE_TYPE_SAME_AGENT = 1
"""Edge type for services operated by the same TSP (internal coordination)."""

EDGE_TYPE_DEST_ORIGIN = 2
"""Edge type for services where one's destination connects to another's origin."""
