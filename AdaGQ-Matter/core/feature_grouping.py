"""
Matter-Aware Feature Grouping Module.

Maps generic IoT flow features to Matter device categories (Section 4.3):
- FG1 (Sensors): Temperature, humidity, occupancy → High κ priority
- FG2 (Actuators): Switches, dimmers, locks → High κ priority
- FG3 (Controllers): Hubs, bridges → Base κ priority
- FG4 (Network): Routers, border routers → Base κ priority

During Top-k selection, coordinates in FG1/FG2 receive 1.2× priority boost
on their absolute magnitude before ranking.

Reference: AdaGQ-Matter, Section 4.3 (Matter-Aware Feature Grouping)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional


# ============================================================
# IoTID20 Feature Grouping (79 features → 4 Matter groups)
# ============================================================

# IoTID20 features typically include flow-level statistics.
# We group them by Matter device category based on their relevance.

IOTID20_FEATURE_GROUPS = {
    "FG1_Sensors": {
        "matter_type": "Sensors (temperature, humidity, occupancy)",
        "priority_boost": 1.2,
        "description": "Protocol type, packet rate, payload entropy, flow duration",
        # In a 4-layer DNN with 79→64→32→16→2 architecture,
        # the first layer (79→64) processes all input features.
        # We assign the first layer's parameters as FG1-prioritized.
        "layer_ids": [0],  # Layer 0: nn.Linear(79, 64) weights + bias
    },
    "FG2_Actuators": {
        "matter_type": "Actuators (switches, dimmers, locks)",
        "priority_boost": 1.2,
        "description": "Command frequency, state change ratio, response time",
        "layer_ids": [1],  # Layer 1: nn.Linear(64, 32)
    },
    "FG3_Controllers": {
        "matter_type": "Controllers (hubs, bridges)",
        "priority_boost": 1.0,
        "description": "Connection count, multicast ratio, fragmentation rate",
        "layer_ids": [2],  # Layer 2: nn.Linear(32, 16)
    },
    "FG4_Network": {
        "matter_type": "Network (routers, border routers)",
        "priority_boost": 1.0,
        "description": "Hop count, TTL variance, retransmission rate",
        "layer_ids": [3],  # Layer 3: nn.Linear(16, 2)
    },
}


# ============================================================
# CICIoT2023 Feature Grouping (46 features → 4 Matter groups)
# ============================================================

CICIOT2023_FEATURE_GROUPS = {
    "FG1_Sensors": {
        "matter_type": "Sensors",
        "priority_boost": 1.2,
        "layer_ids": [0],
    },
    "FG2_Actuators": {
        "matter_type": "Actuators",
        "priority_boost": 1.2,
        "layer_ids": [1],
    },
    "FG3_Controllers": {
        "matter_type": "Controllers",
        "priority_boost": 1.0,
        "layer_ids": [2],
    },
    "FG4_Network": {
        "matter_type": "Network",
        "priority_boost": 1.0,
        "layer_ids": [3],
    },
}


def get_priority_boost_dict(
    dataset: str = "iotid20",
    feature_groups: Optional[Dict] = None,
) -> Dict[int, float]:
    """
    Generate priority boost mapping from layer_id → boost factor.

    Args:
        dataset: Dataset name ("iotid20" or "ciciot2023")
        feature_groups: Custom feature group definition (None = use default)

    Returns:
        Dict mapping layer_id → priority_boost factor
    """
    groups = feature_groups or (
        IOTID20_FEATURE_GROUPS if dataset == "iotid20" else CICIOT2023_FEATURE_GROUPS
    )

    priority_dict = {}
    for group_name, group_info in groups.items():
        boost = group_info["priority_boost"]
        for layer_id in group_info["layer_ids"]:
            priority_dict[layer_id] = boost

    return priority_dict


def apply_feature_grouping_boost(
    gradient_flat: np.ndarray,
    layer_ranges: Dict[int, Tuple[int, int]],
    priority_boost: Dict[int, float],
    dataset: str = "iotid20",
) -> np.ndarray:
    """
    Apply priority boost to gradient coordinates belonging to FG1/FG2.

    During Top-k selection, coordinates in FG1/FG2 receive 1.2× on
    their absolute magnitude before ranking.

    Args:
        gradient_flat: Full flat gradient vector (d-dim)
        layer_ranges: Dict mapping layer_id → (start, end) in flat vector
        priority_boost: Dict mapping layer_id → boost factor
        dataset: Dataset name

    Returns:
        Modified gradient for ranking (original gradient unchanged)
    """
    # Create ranking scores (copy of absolute values)
    ranking_scores = np.abs(gradient_flat).copy()

    # Apply boost to FG1/FG2 layer coordinates
    for layer_id, boost in priority_boost.items():
        if layer_id in layer_ranges:
            start, end = layer_ranges[layer_id]
            ranking_scores[start:end] *= boost

    return ranking_scores


def get_feature_groups_for_dataset(dataset: str) -> Dict:
    """Get feature group definition for a dataset."""
    if dataset == "iotid20":
        return IOTID20_FEATURE_GROUPS
    elif dataset == "ciciot2023":
        return CICIOT2023_FEATURE_GROUPS
    else:
        return IOTID20_FEATURE_GROUPS  # Default fallback


if __name__ == "__main__":
    # Quick test: show priority boost mapping
    print("IoTID20 priority boost mapping:")
    pb_iotid = get_priority_boost_dict("iotid20")
    for layer, boost in pb_iotid.items():
        print(f"  Layer {layer}: boost = {boost}×")

    print("\nCICIoT2023 priority boost mapping:")
    pb_ciciot = get_priority_boost_dict("ciciot2023")
    for layer, boost in pb_ciciot.items():
        print(f"  Layer {layer}: boost = {boost}×")
