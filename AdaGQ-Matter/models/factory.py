"""
Model factory — creates models based on config.

Supported models:
- "dnn": AnomalyDNN (4-layer, d≈7,762 for IoTID20, CPU-friendly)
- "lstm_ae": LSTMAnomalyDetector (GPU-only, d≈45k+)
"""

from models.dnn import AnomalyDNN, build_model
from models.lstm_ae import LSTMAnomalyDetector
import torch.nn as nn


def create_model(config: dict) -> nn.Module:
    """
    Create a model based on config dict.

    Args:
        config: Configuration dict with keys:
            - model_type: "dnn" or "lstm_ae"
            - dataset: determines input_dim
            - input_dim_iotid20: 79
            - input_dim_ciciot2023: 46
            - hidden_dims: [64, 32, 16] (for DNN)
            - output_dim: 2
            - lstm_hidden_dim: 128 (for LSTM-AE)
            - lstm_num_layers: 2 (for LSTM-AE)

    Returns:
        nn.Module model ready for training
    """
    model_type = config.get("model_type", "dnn")
    output_dim = config.get("output_dim", 2)

    # Determine input dimension from dataset
    dataset = config.get("dataset", "iotid20")
    if dataset == "iotid20":
        input_dim = config.get("input_dim_iotid20", 79)
    elif dataset == "ciciot2023":
        input_dim = config.get("input_dim_ciciot2023", 46)
    elif dataset == "both":
        # Use IoTID20 as default for "both" (will be overridden per run)
        input_dim = config.get("input_dim_iotid20", 79)
    else:
        input_dim = config.get("input_dim_iotid20", 79)

    if model_type == "dnn":
        hidden_dims = config.get("hidden_dims", [64, 32, 16])
        model = build_model(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
        )
    elif model_type == "lstm_ae":
        hidden_dim = config.get("lstm_hidden_dim", 128)
        num_layers = config.get("lstm_num_layers", 2)
        model = LSTMAnomalyDetector(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            output_dim=output_dim,
        )
        print(f"[Model] {model}")
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    return model


def get_input_dim(config: dict, dataset: str = None) -> int:
    """Get input dimension for a specific dataset."""
    ds = dataset or config.get("dataset", "iotid20")
    if ds == "iotid20":
        return config.get("input_dim_iotid20", 79)
    elif ds == "ciciot2023":
        return config.get("input_dim_ciciot2023", 46)
    else:
        return config.get("input_dim_iotid20", 79)
