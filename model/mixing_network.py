import torch
import torch.nn as nn



class QMixingNetwork(nn.Module):
    def __init__(self, n_agents, state_dim, hypernet_embed=64):
        super(QMixingNetwork, self).__init__()
        self.n_agents = n_agents
        self.state_dim = state_dim

        # Hypernetworks for mixing weights
        self.hyper_w_1 = nn.Sequential(
            nn.Linear(state_dim, hypernet_embed),
            nn.ReLU(),
            nn.Linear(hypernet_embed, n_agents * 32)
        )
        self.hyper_w_final = nn.Sequential(
            nn.Linear(state_dim, hypernet_embed),
            nn.ReLU(),
            nn.Linear(hypernet_embed, 32)
        )

        # Hypernetworks for mixing biases
        self.hyper_b_1 = nn.Linear(state_dim, 32)
        self.hyper_b_final = nn.Sequential(
            nn.Linear(state_dim, hypernet_embed),
            nn.ReLU(),
            nn.Linear(hypernet_embed, 1)
        )

    def forward(self, agent_qs, states):
        """
        agent_qs: shape (batch_size, n_agents)
        states: shape (batch_size, state_dim)
        """
        batch_size = agent_qs.size(0)

        # Generate weights and biases using hypernetworks
        w1 = torch.abs(self.hyper_w_1(states)).view(batch_size, self.n_agents, 32)
        b1 = self.hyper_b_1(states).view(batch_size, 1, 32)
        w_final = torch.abs(self.hyper_w_final(states)).view(batch_size, 32, 1)
        b_final = self.hyper_b_final(states).view(batch_size, 1, 1)

        # First layer mixing
        hidden = torch.relu(torch.bmm(agent_qs.unsqueeze(1), w1) + b1)
        # Final layer mixing
        q_tot = torch.bmm(hidden, w_final) + b_final
        return q_tot.squeeze(-1)
