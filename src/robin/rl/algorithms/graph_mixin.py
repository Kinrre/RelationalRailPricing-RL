"""Graph processing mixin for RL algorithms."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from robin.rl.algorithms.constants import DEFAULT_NUM_LAYERS, NUM_RELATIONS
from robin.rl.algorithms.gnn import ServicesRGCN
from robin.rl.entities import StatsSubprocVectorEnv, VectorDummyEnvEmbeddingWrapper

from torch_geometric.data import Data


class ServiceAttentionPooling(nn.Module):
    """
    Attention-based pooling for service embeddings.

    This module learns to attend to the most relevant services when pooling
    from [batch_size, num_services, embedding_dim] to [batch_size, embedding_dim].
    """

    def __init__(self, embedding_dim: int, uniform_attention: bool = False) -> None:
        super().__init__()
        self.uniform_attention = uniform_attention
        if not uniform_attention:
            self.attention = nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim // 2),
                nn.ReLU(),
                nn.Linear(embedding_dim // 2, 1)
            )

    def forward(self, embeddings: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply attention pooling to service embeddings.

        Args:
            embeddings (torch.Tensor): Service embeddings.

        Returns:
            tuple: Service attention-pooled embeddings and attention weights.
        """
        if self.uniform_attention:
            batch_size, num_services, _ = embeddings.shape
            attention_scores = torch.ones(batch_size, num_services, 1, device=embeddings.device)
        else:
            attention_scores = self.attention(embeddings)  # [batch_size, num_services, 1]

        attention_weights = F.softmax(attention_scores, dim=1)  # [batch_size, num_services, 1]
        pooled = torch.sum(embeddings * attention_weights, dim=1)  # [batch_size, embedding_dim]
        return pooled, attention_weights


class GraphProcessingMixin:
    """
    Mixin providing graph neural network functionality for RL algorithms.
    
    This mixin provides common graph processing functionality that can be shared
    across different RL algorithms like GraphMADDPG and GraphMATD3.
    
    Attributes:
        embedding_dim (int): The dimension of the embedding for graph observations.
        services_rgcn (list[ServicesRGCN]): RGCN networks for each agent.
        services_rgcn_optims (list[torch.optim.Optimizer]): Optimizers for RGCN and attention pooling parameters.
        attention_poolings (list[ServiceAttentionPooling]): Attention pooling modules for each agent.
    """
    
    def _create_embedding_wrapper(self, env: StatsSubprocVectorEnv) -> VectorDummyEnvEmbeddingWrapper:
        """
        Create the embedding wrapper for graph observations.

        Args:
            env (StatsSubprocVectorEnv): The original environment.

        Returns:
            VectorDummyEnvEmbeddingWrapper: Environment wrapper with embedding observation space.
        """
        num_agents = env.get_env_attr('num_agents')[0]
        supply_config = env.get_env_attr('kernel')[0].supply
        sample_rgcn = ServicesRGCN.from_supply_config(supply_config)
        self.embedding_dim = sample_rgcn.embedding_dim
        return VectorDummyEnvEmbeddingWrapper(env, self.embedding_dim, num_agents)

    def _eval_rgcns(self) -> None:
        """
        Set the RGCN networks and attention pooling to evaluation mode.
        """
        for nets in zip(self.services_rgcns, self.attention_poolings):
            for net in nets:
                net.eval()

    def _get_save_dict_with_rgcn(self) -> dict:
        """
        Get the state dictionary for saving the model including RGCN networks, attention pooling, and optimizers.

        Returns:
            dict: Dictionary containing the model parameters including RGCN and attention pooling components.
        """
        save_dict = super()._get_save_dict()
        save_dict.update({
            'services_rgcns': [rgcn.state_dict() for rgcn in self.services_rgcns],
            'services_rgcn_optims': [opt.state_dict() for opt in self.services_rgcn_optims],
            'attention_poolings': [pooling.state_dict() for pooling in self.attention_poolings]
        })
        return save_dict

    def _init_graph_processing(
        self,
        env: StatsSubprocVectorEnv,
        device: torch.device,
        services_rgcn_lr: float = 0.001,
        rgcn_num_layers: int = DEFAULT_NUM_LAYERS
    ) -> None:
        """
        Initialize graph processing components.

        Args:
            env (StatsSubprocVectorEnv): The original environment.
            device (torch.device): The device to run the algorithm on.
            services_rgcn_lr (float): The learning rate of the RGCN optimizer.
            rgcn_num_layers (int): Number of RGCN layers.
        """
        num_agents = env.get_env_attr('num_agents')[0]
        supply_config = env.get_env_attr('kernel')[0].supply

        # Store shared static edge topology (same for all parallel envs and agents)
        services_graphs = env.get_env_attr('services_graph')
        self.edge_index = services_graphs[0][0].edge_index
        self.edge_type = services_graphs[0][0].edge_type

        # Create single RGCN and attention pooling for each agent
        self.services_rgcns: list[ServicesRGCN] = []
        self.services_rgcn_optims: list[torch.optim.Optimizer] = []
        self.attention_poolings: list[ServiceAttentionPooling] = []
        exclude_edge_type = env.get_env_attr('exclude_edge_type')[0]
        num_relations = NUM_RELATIONS - (1 if exclude_edge_type is not None else 0)

        for _ in range(num_agents):
            services_rgcn = ServicesRGCN.from_supply_config(supply_config, num_layers=rgcn_num_layers, num_relations=num_relations).to(device)
            self.services_rgcns.append(services_rgcn)
            
            # Create attention pooling module
            attention_pooling = ServiceAttentionPooling(services_rgcn.embedding_dim).to(device)
            self.attention_poolings.append(attention_pooling)
            
            # Create optimizer for both RGCN and attention pooling parameters
            combined_params = list(services_rgcn.parameters()) + list(attention_pooling.parameters())
            self.services_rgcn_optims.append(
                torch.optim.Adam(combined_params, lr=services_rgcn_lr)
            )

    def _load_from_save_dict_with_rgcn(self, save_dict: dict) -> None:
        """
        Load the model parameters from a save dictionary including RGCN networks, attention pooling, and optimizers.

        Args:
            save_dict (dict): Dictionary containing the model parameters.
        """
        super()._load_from_save_dict(save_dict)
        for agent_idx in range(self.num_agents):
            self.services_rgcns[agent_idx].load_state_dict(save_dict['services_rgcns'][agent_idx])
            self.services_rgcn_optims[agent_idx].load_state_dict(save_dict['services_rgcn_optims'][agent_idx])
            self.attention_poolings[agent_idx].load_state_dict(save_dict['attention_poolings'][agent_idx])

    def _process_graph_observations(self, obs: np.ndarray, agent_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Process graph observations through ServicesRGCN and attention pooling for a specific agent.
        
        Args:
            obs (np.ndarray): Node features of the graph observation.
            agent_idx (int): The index of the agent whose RGCN to use for processing.

        Returns:
            tuple: Pooled service embeddings and attention weights for the agent.
        """
        batch_size = obs.shape[0]
        x = torch.stack([torch.from_numpy(env_obs) for env_obs in obs]) # [batch_size, num_services, feature_dim]
        edge_index = self.edge_index.expand(batch_size, -1, -1) # [batch_size, 2, num_edges]
        edge_type = self.edge_type.expand(batch_size, -1) # [batch_size, num_edges]
        services_graphs = Data(x=x, edge_index=edge_index, edge_type=edge_type).to(self.device)
        service_embeddings = self.services_rgcns[agent_idx](services_graphs)  # [batch_size, num_services, embedding_dim]
        pooled_embeddings, attention_weights = self.attention_poolings[agent_idx](service_embeddings)
        return pooled_embeddings, attention_weights

    def _train_rgcns(self) -> None:
        """
        Set the RGCN networks and attention pooling to training mode.
        """
        for nets in zip(self.services_rgcns, self.attention_poolings):
            for net in nets:
                net.train()

    def compute_attention_entropy(self, attention_weights: torch.Tensor) -> torch.Tensor:
        """
        Compute entropy of attention weights to measure attention concentration.
        
        Args:
            attention_weights (torch.Tensor): Attention weights.
            
        Returns:
            torch.Tensor: Entropy values for each service.
        """
        # Remove the last dimension [batch_size, num_services, 1]
        weights = attention_weights.squeeze(-1)  # [batch_size, num_services]
        
        # Add small epsilon to avoid log(0)
        eps = 1e-8
        weights = weights + eps
        entropy = -torch.sum(weights * torch.log(weights), dim=1)  # [batch_size]
        return entropy
