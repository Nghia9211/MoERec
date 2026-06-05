import torch
import torch.nn as nn


class LightGCN(nn.Module):
    def __init__(self, num_users: int, num_items: int, embedding_dim: int, num_layers: int = 3):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.num_layers = num_layers
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.item_embedding.weight, std=0.1)

    def forward(self, A: torch.Tensor):
        E0 = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        all_embeddings = [E0]
        E_k = E0
        for _ in range(self.num_layers):
            E_k = torch.sparse.mm(A, E_k)
            all_embeddings.append(E_k)
        E_final = torch.stack(all_embeddings, dim=1).mean(dim=1)

        return E_final, E0