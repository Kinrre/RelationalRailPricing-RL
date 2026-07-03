"""Graph Neural Network models for services graph embedding generation."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from robin.rl.constants import NODE_FEATURE_PROFIT_POSITION
from robin.rl.algorithms.constants import (
    BASIC_FEATURES, CATEGORICAL_BASIC_FEATURES, CATEGORICAL_PER_MARKET_SEAT, CONTINUOUS_PER_MARKET_SEAT,
    DEFAULT_CATEGORICAL_EMBEDDING_DIM, DEFAULT_CONTINUOUS_HIDDEN_DIM, DEFAULT_CONTINUOUS_INPUT_FEATURES,
    DEFAULT_CONTINUOUS_OUTPUT_DIM, DEFAULT_DROPOUT, DEFAULT_EMBEDDING_DIM, DEFAULT_EMBEDDING_TOTAL_DIM,
    DEFAULT_HIDDEN_DIM, DEFAULT_NUM_LAYERS, FEATURES_PER_MARKET_SEAT, NUM_RELATIONS
)
from robin.supply.entities import Supply

from functools import cached_property
from torch_geometric.nn import RGCNConv
from torch_geometric.data import Data
from typing import Dict, Optional


class ServicesRGCN(nn.Module):
    """
    Graph Relational Convolutional Network for generating service embeddings from services graph.
    
    This model takes services graph with node features and edge connections and generates
    compact service embeddings that capture service characteristics and market relationships.
    
    Attributes:
        embedding_dim (int): Output embedding dimension. 
        hidden_dim (int): Hidden layer dimension.
        num_layers (int): Number of GCN layers.
        dropout (float): Dropout rate.
        categorical_embedding_dim (int): Dimension for all categorical embeddings.
        categorical_cardinalities (Dict[str, int], optional): Actual cardinalities for categorical features.
    """
    
    def __init__(
        self,
        embedding_dim: int = DEFAULT_EMBEDDING_DIM,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
        num_layers: int = DEFAULT_NUM_LAYERS,
        dropout: float = DEFAULT_DROPOUT,
        categorical_num_embeddings: Optional[Dict[str, int]] = None,
        categorical_embedding_dim: int = DEFAULT_CATEGORICAL_EMBEDDING_DIM,
        continuous_input_features: int = DEFAULT_CONTINUOUS_INPUT_FEATURES,
        continuous_hidden_dim: int = DEFAULT_CONTINUOUS_HIDDEN_DIM,
        continuous_output_dim: int = DEFAULT_CONTINUOUS_OUTPUT_DIM,
        embedding_total_dim: Optional[int] = DEFAULT_EMBEDDING_TOTAL_DIM,
        num_relations: int = NUM_RELATIONS
    ) -> None:
        """
        Initialize ServicesRGCN model.

        Args:
            embedding_dim (int): Output embedding dimension.
            hidden_dim (int): Hidden layer dimension.
            num_layers (int): Number of GCN layers.
            dropout (float): Dropout rate.
            categorical_num_embeddings (Dict[str, int], optional): Number of embeddings for each categorical feature.
            categorical_embedding_dim (int): Dimension for all categorical embeddings.
            continuous_input_features (int): Input dimension for continuous features MLP.
            continuous_hidden_dim (int): Hidden dimension for continuous features MLP.
            continuous_output_dim (int): Output dimension for continuous features MLP.
            embedding_total_dim (int): Total dimension of concatenated embeddings. It should be
                categorical_embedding_dim * (number of categorical features + number of market-seat combinations).
            num_relations (int): Number of edge relation types in the services graph.
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Categorical feature names
        self.categorical_features = ['tsp', 'line', 'corridor', 'time_slot', 'rolling_stock', 'origin', 'destination', 'seat_type']
        self.categorical_embedding_dim = categorical_embedding_dim
        self.continuous_input_features = continuous_input_features
        self.continuous_hidden_dim = continuous_hidden_dim
        self.continuous_output_dim = continuous_output_dim
        
        # Categorical embeddings
        self.embeddings = nn.ModuleDict()
        for cat_name in self.categorical_features:
            num_embeddings = categorical_num_embeddings[cat_name]
            self.embeddings[cat_name] = nn.Embedding(num_embeddings, categorical_embedding_dim)
        
        # Continuous features MLP
        self.continuous_mlp = nn.Sequential(
            nn.Linear(continuous_input_features, continuous_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(continuous_hidden_dim, continuous_output_dim)
        )
        
        # Input projection: embeddings + processed continuous features -> hidden_dim
        total_input_dim = embedding_total_dim + continuous_output_dim
        self.input_projection = nn.Linear(total_input_dim, hidden_dim)
        
        # R-GCN layers
        self.rgcn_layers = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.rgcn_layers.append(RGCNConv(hidden_dim, hidden_dim, num_relations))
        self.rgcn_layers.append(RGCNConv(hidden_dim, embedding_dim, num_relations))

        # Layer normalization
        self.layer_norms = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.layer_norms.append(nn.LayerNorm(hidden_dim))
        self.layer_norms.append(nn.LayerNorm(embedding_dim))
        
        # Dropout
        self.dropout_layer = nn.Dropout(dropout)

    def _process_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process raw node features into embeddings.
        
        Extracts categorical features and applies embeddings, then combines with continuous features (prices, ticket counts).
        
        Args:
            x (torch.Tensor): Raw node features [num_nodes, feature_dim].
            
        Returns:
            torch.Tensor: Processed features [num_nodes, hidden_dim].
        """
        # Process all categorical features
        embeddings = []
        for pos, feature_name in zip(self.categorical_positions, self.categorical_feature_names):
            emb = self.embeddings[feature_name](x[:, :, pos].long())
            embeddings.append(emb)
        all_embeddings = torch.cat(embeddings, dim=2)
        
        # Process continuous features
        continuous_features = x[:, :, self.continuous_positions]
        processed_continuous = self.continuous_mlp(continuous_features)

        # Concatenate all embeddings and processed continuous features
        combined_features = torch.cat([all_embeddings, processed_continuous], dim=2)
        return self.input_projection(combined_features)

    @cached_property
    def categorical_feature_names(self) -> list[str]:
        """Names of categorical features corresponding to positions."""
        names = []
        # Basic categorical features
        for i in range(CATEGORICAL_BASIC_FEATURES):
            names.append(self.categorical_features[i])

        # Market seat features
        for _ in range(self.max_market_seats):
            for i in range(CATEGORICAL_PER_MARKET_SEAT):
                names.append(self.categorical_features[CATEGORICAL_BASIC_FEATURES + i])
        return names

    @cached_property
    def categorical_positions(self) -> list[int]:
        """Positions of all categorical features in the tensor."""
        positions = []
        # Basic categorical features (positions 0-4)
        for i in range(CATEGORICAL_BASIC_FEATURES):
            positions.append(i)

        # Market seat features (positions 6 onwards)
        for market_seat_idx in range(self.max_market_seats):
            start_idx = BASIC_FEATURES + market_seat_idx * FEATURES_PER_MARKET_SEAT
            # Categorical: origin, destination, seat_type
            for i in range(CATEGORICAL_PER_MARKET_SEAT):
                positions.append(start_idx + i)
        return positions

    @cached_property
    def continuous_positions(self) -> list[int]:
        """Positions of all continuous features in the tensor."""
        positions = [NODE_FEATURE_PROFIT_POSITION]
        for market_seat_idx in range(self.max_market_seats):
            start_idx = BASIC_FEATURES + market_seat_idx * FEATURES_PER_MARKET_SEAT
            # Continuous: price, tickets_sold
            for i in range(CONTINUOUS_PER_MARKET_SEAT):
                positions.append(start_idx + CATEGORICAL_PER_MARKET_SEAT + i)
        return positions
    
    def forward(self, data: Data) -> torch.Tensor:
        """
        Forward pass through GCN model.
        
        Args:
            data (Data): PyG Data object with node features (x) and edge index.
            
        Returns:
            torch.Tensor: Service embeddings [batch_size, num_nodes, embedding_dim].
        """
        x, edge_index, edge_type = data.x, data.edge_index, data.edge_type
        
        # Store original batch and node dimensions
        batch_size, num_nodes = x.shape[:2]

        # Reshape from [batch_size, 2, num_edges] to [2, batch_size * num_edges]
        edge_index = edge_index.transpose(0, 1).flatten(1, 2)

        # Reshape edge_type from [batch_size, num_edges] to [batch_size * num_edges]
        edge_type = edge_type.flatten()

        # Process input features
        x = self._process_features(x)

        # Apply R-GCN layers
        for i, rgcn_layer in enumerate(self.rgcn_layers):
            # Reshape from [batch_size, num_nodes, hidden_dim] to [batch_size * num_nodes, hidden_dim]
            x = x.flatten(0, -2)
            residual = x
            x = rgcn_layer(x, edge_index, edge_type)

            # Apply layer norm and dropout (except for last layer)
            if i < len(self.rgcn_layers) - 1:
                x = self.layer_norms[i](x)
                x = F.relu(x)
                x = self.dropout_layer(x)
                x = x + residual
            else:
                # Final layer - just layer norm
                x = self.layer_norms[i](x)

        # Restore original batch and node dimensions
        x = x.unflatten(0, (batch_size, num_nodes))
        return x

    @classmethod
    def from_supply_config(cls, supply: Supply, **kwargs) -> 'ServicesRGCN':
        """
        Create ServicesRGCN model from Supply configuration.

        Args:
            supply (Supply): Supply configuration object containing service data.
            **kwargs: Additional keyword arguments for model initialization.
        
        Returns:
            ServicesRGCN: Initialized RGCN model with appropriate categorical cardinalities.
        """
        categorical_num_embeddings = {
            'tsp': len(supply.tsps),
            'line': len(supply.lines),
            'corridor': len(supply.corridors),
            'time_slot': len(supply.time_slots),
            'rolling_stock': len(supply.rolling_stocks),
            'origin': len(supply.stations),
            'destination': len(supply.stations),
            'seat_type': len(supply.seats)
        }
        
        # Calculate maximum number of market seat entries from supply config
        max_market_seat_entries = 0
        for service in supply.services:
            total_market_seats = 0
            for (origin, destination), seats in service.prices.items():
                total_market_seats += len(seats)
            max_market_seat_entries = max(max_market_seat_entries, total_market_seats)

        # Each market seat entry has 2 continuous features (price + tickets_sold) plus 1 for profit
        continuous_input_features = max_market_seat_entries * CONTINUOUS_PER_MARKET_SEAT + 1

        # Calculate embedding dimensions: basic categorical features + market seat features (3 per entry)
        categorical_embedding_dim = kwargs.get('categorical_embedding_dim', DEFAULT_CATEGORICAL_EMBEDDING_DIM)
        embedding_total_dim = (CATEGORICAL_BASIC_FEATURES * categorical_embedding_dim) + \
            (max_market_seat_entries * CATEGORICAL_PER_MARKET_SEAT * categorical_embedding_dim)
        return cls(
            categorical_num_embeddings=categorical_num_embeddings,
            continuous_input_features=continuous_input_features,
            embedding_total_dim=embedding_total_dim,
            **kwargs
        )

    @cached_property
    def max_market_seats(self) -> int:
        """Maximum number of market-seat entries."""
        return (self.continuous_input_features - 1) // CONTINUOUS_PER_MARKET_SEAT
