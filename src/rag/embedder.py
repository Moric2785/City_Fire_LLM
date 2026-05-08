# FLLM/src/rag/embedder.py
import torch
import torch.optim as optim
try:
    from torch_geometric.nn import TransE
except ImportError:
    print("Warning: torch_geometric not installed, please run: pip install torch-geometric")
    TransE = None
from tqdm import tqdm
import logging

class KGEmbedder:
    """
    Class for generating embedding vectors for knowledge graphs.
    Uses PyTorch Geometric's TransE model.
    """

    def __init__(self, num_nodes: int, num_relations: int, embedding_dim: int = 128):
        """
        Initialize KGEmbedder.

        Args:
            num_nodes (int): Total number of nodes in the graph.
            num_relations (int): Total number of relations in the graph.
            embedding_dim (int): Dimension of embedding vectors.
        """
        if TransE is None:
            raise ImportError("torch_geometric not installed, please run: pip install torch-geometric")
            
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = TransE(
            num_nodes=num_nodes,
            num_relations=num_relations,
            hidden_channels=embedding_dim,
        ).to(self.device)
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"KGEmbedder initialized on {self.device} with embedding dimension {embedding_dim}.")

    def train(self, triplets: list, epochs: int, batch_size: int, lr: float):
        """
        Train TransE model.

        Args:
            triplets (list): List of triplets in (head, relation, tail) format.
            epochs (int): Number of training epochs.
            batch_size (int): Batch size.
            lr (float): Learning rate.
        """
        self.logger.info(f"Starting knowledge graph embedding training for {epochs} epochs.")
        edge_index = torch.tensor(triplets, dtype=torch.long).t().contiguous().to(self.device)
        head_index, rel_type, tail_index = edge_index

        loader = self.model.loader(
            head_index=head_index,
            rel_type=rel_type,
            tail_index=tail_index,
            batch_size=batch_size,
            shuffle=True,
        )
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        for epoch in range(1, epochs + 1):
            total_loss = 0
            for head, rel, tail in tqdm(loader, desc=f"Epoch {epoch}/{epochs}"):
                optimizer.zero_grad()
                loss = self.model.loss(head, rel, tail)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(loader)
            self.logger.info(f"Epoch: {epoch:03d}, Average loss: {avg_loss:.4f}")

    def get_node_embeddings(self) -> torch.Tensor:
        """
        Get trained node embeddings.

        Returns:
            torch.Tensor: Node embedding tensor.
        """
        return self.model.node_emb.weight.data.cpu()
