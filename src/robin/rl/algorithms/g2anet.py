"""G2ANet module and G2ANetCritic for the GA-AC algorithm."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from robin.rl.algorithms.constants import DEFAULT_EMBEDDING_DIM, HIDDEN_SIZE


class G2ANet(nn.Module):
    """
    Two-stage graph attention network (hard BiLSTM + soft scaled dot-product attention).

    Attributes:
        hard_bilstm (nn.LSTM): Bidirectional LSTM for hard attention gates.
        hard_fc (nn.Linear): Linear layer mapping BiLSTM output to binary logits.
        W_key (nn.Linear): Key projection for soft attention.
        W_query (nn.Linear): Query projection for soft attention.
        V (nn.Linear): Value projection for aggregation.
        temperature (float): Gumbel-softmax temperature.
        min_temperature (float): Minimum temperature bound.
    """

    def __init__(
        self,
        encoding_dim: int,
        hard_hidden_dim: int = DEFAULT_EMBEDDING_DIM,
        soft_key_dim: int = DEFAULT_EMBEDDING_DIM,
        initial_temperature: float = 1.0,
        min_temperature: float = 0.1
    ) -> None:
        """
        Initialize G2ANet.

        Args:
            encoding_dim (int): Dimensionality of agent encodings.
            hard_hidden_dim (int): Hidden size for the BiLSTM (per direction).
            soft_key_dim (int): Key/query dimensionality for soft attention.
            initial_temperature (float): Initial Gumbel-softmax temperature.
            min_temperature (float): Minimum Gumbel-softmax temperature after annealing.
        """
        super().__init__()
        self.encoding_dim = encoding_dim
        self.hard_bilstm = nn.LSTM(
            input_size=2 * encoding_dim,
            hidden_size=hard_hidden_dim,
            bidirectional=True,
            batch_first=True
        )
        self.hard_fc = nn.Linear(2 * hard_hidden_dim, 2)
        self.W_key = nn.Linear(encoding_dim, soft_key_dim, bias=False)
        self.W_query = nn.Linear(encoding_dim, soft_key_dim, bias=False)
        self.V = nn.Linear(encoding_dim, encoding_dim, bias=False)
        self.temperature = initial_temperature
        self.min_temperature = min_temperature

    def forward(self, e_i: torch.Tensor, e_others: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """
        Compute attention-weighted aggregation of other agents' encodings.

        Args:
            e_i (torch.Tensor): Encoding of the current agent.
            e_others (torch.Tensor): Encodings of the other agents.

        Returns:
            tuple[torch.Tensor, dict]: Aggregated encoding and dict with 'hard_gates' and 'soft_weights'.
        """
        batch_size, n_others, _ = e_others.shape

        # Hard attention via BiLSTM and Gumbel-softmax
        e_i_expanded = e_i.unsqueeze(1).expand(-1, n_others, -1)   # [batch, N-1, encoding_dim]
        lstm_input = torch.cat([e_i_expanded, e_others], dim=2)    # [batch, N-1, 2*encoding_dim]
        lstm_out, _ = self.hard_bilstm(lstm_input)                 # [batch, N-1, 2*hard_hidden_dim]
        hard_logits = self.hard_fc(lstm_out)                       # [batch, N-1, 2]

        if self.training:
            # Gumbel-softmax with straight-through: hard=True gives discrete 0/1
            hard_gates = F.gumbel_softmax(hard_logits, tau=self.temperature, hard=True)[:, :, 1]
        else:
            hard_gates = hard_logits.argmax(dim=-1).float()        # [batch, N-1]

        # Soft attention via scaled dot-product
        query = self.W_query(e_i)                                  # [batch, soft_key_dim]
        keys = self.W_key(e_others)                                # [batch, N-1, soft_key_dim]
        scale = query.shape[-1] ** 0.5
        scores = torch.bmm(keys, query.unsqueeze(2)).squeeze(2) / scale  # [batch, N-1]
        soft_weights = F.softmax(scores, dim=-1)                   # [batch, N-1]

        # Value projection and aggregation
        values = F.relu(self.V(e_others))                          # [batch, N-1, encoding_dim]
        combined = (hard_gates * soft_weights).unsqueeze(2)        # [batch, N-1, 1]
        x_i = (combined * values).sum(dim=1)                       # [batch, encoding_dim]

        return x_i, {'hard_gates': hard_gates, 'soft_weights': soft_weights}


class G2ANetCritic(nn.Module):
    """
    Critic network for GA-AC using G2ANet for inter-agent attention.

    Attributes:
        encoder_self (nn.Sequential): MLP encoding agent i's obs-action pair.
        encoders_others (nn.ModuleList): Per-other-agent MLPs encoding each other agent's obs-action.
        g2anet (G2ANet): Graph attention module for aggregating other agents' encodings.
        output_mlp (nn.Sequential): MLP mapping combined encoding to Q-value.
    """

    def __init__(
        self,
        obs_dim_i: int,
        action_dim_i: int,
        obs_dims_others: list[int],
        action_dims_others: list[int],
        encoding_dim: int = HIDDEN_SIZE
    ) -> None:
        """
        Initialize G2ANetCritic.

        Args:
            obs_dim_i (int): Observation dimensionality of agent i.
            action_dim_i (int): Action dimensionality of agent i.
            obs_dims_others (list[int]): Observation dimensionalities of the other agents.
            action_dims_others (list[int]): Action dimensionalities of the other agents.
            encoding_dim (int): Size of agent encoding vectors.
        """
        super().__init__()
        self.encoder_self = nn.Sequential(
            nn.Linear(obs_dim_i + action_dim_i, encoding_dim),
            nn.ReLU()
        )
        self.encoders_others = nn.ModuleList([
            nn.Sequential(nn.Linear(obs_dim_j + action_dim_j, encoding_dim), nn.ReLU(), nn.Linear(encoding_dim, encoding_dim))
            for obs_dim_j, action_dim_j in zip(obs_dims_others, action_dims_others)
        ])
        self.g2anet = G2ANet(encoding_dim)
        self.output_mlp = nn.Sequential(
            nn.Linear(2 * encoding_dim, encoding_dim),
            nn.ReLU(),
            nn.Linear(encoding_dim, 1)
        )

    def forward(
        self,
        obs_i: torch.Tensor,
        action_i: torch.Tensor,
        obs_others_list: list[torch.Tensor],
        actions_others_list: list[torch.Tensor]
    ) -> tuple[torch.Tensor, dict]:
        """
        Compute Q-value for a single agent given its own obs-action and other agents' obs-actions.

        Args:
            obs_i (torch.Tensor): Agent i's observation, shape [batch, obs_dim_i].
            action_i (torch.Tensor): Agent i's action, shape [batch, action_dim_i].
            obs_others_list (list[torch.Tensor]): Other agents' observations, each [batch, obs_dim_j].
            actions_others_list (list[torch.Tensor]): Other agents' actions, each [batch, action_dim_j].

        Returns:
            tuple[torch.Tensor, dict]: Q-value estimate and attention info dict with 'hard_gates' and 'soft_weights'.
        """
        # Encode agent i
        g_i = self.encoder_self(torch.cat([obs_i, action_i], dim=-1))   # [batch, encoding_dim]

        # Encode each other agent with its own encoder
        g_others = torch.stack([
            self.encoders_others[k](torch.cat([obs_others_list[k], actions_others_list[k]], dim=-1))
            for k in range(len(obs_others_list))
        ], dim=1)                                                        # [batch, N-1, encoding_dim]

        # G2ANet attention
        x_i, attn_info = self.g2anet(g_i, g_others)

        # Q-value
        q = self.output_mlp(torch.cat([g_i, x_i], dim=-1))              # [batch, 1]
        return q, attn_info
