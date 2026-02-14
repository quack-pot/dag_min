import torch
import dataclasses

@dataclasses.dataclass
class GraphCreateInfo:
    input_node_count: int
    output_node_count: int

    weights: torch.Tensor
    biases: torch.Tensor
    weight_threshold: float = 0.1

class Graph:
    ## *=================================================
    ## *
    ## * __init__
    ## *
    ## *=================================================

    def __init__(self, create_info: GraphCreateInfo) -> None:
        self.input_node_count: int = create_info.input_node_count
        self.output_node_count: int = create_info.output_node_count

        raw_node_count: int = create_info.biases.numel()
        keep_mask: torch.Tensor = torch.ones(
            raw_node_count,
            dtype=torch.bool,
            device=create_info.weights.device,
        )

        pruned_weights: torch.Tensor = create_info.weights.clone()
        pruned_weights[torch.abs(pruned_weights) <= create_info.weight_threshold] = 0.0

        incoming_strength: torch.Tensor = torch.abs(pruned_weights).sum(dim=0)
        dead_nodes: torch.Tensor = (incoming_strength == 0)

        hidden_start: int = self.input_node_count
        hidden_end: int = raw_node_count - self.output_node_count
        keep_mask[hidden_start:hidden_end] &= ~dead_nodes[hidden_start:hidden_end]

        self.weights: torch.Tensor = pruned_weights[keep_mask][:, keep_mask]
        self.biases: torch.Tensor = create_info.biases.clone()[keep_mask]

        self.hidden_node_count: int = self.biases.numel() - self.input_node_count - self.output_node_count

    ## *=================================================
    ## *
    ## * getInputNodeCount
    ## *
    ## *=================================================

    def getInputNodeCount(self) -> int:
        return self.input_node_count
    
    ## *=================================================
    ## *
    ## * getHiddenNodeCount
    ## *
    ## *=================================================

    def getHiddenNodeCount(self) -> int:
        return self.hidden_node_count
    
    ## *=================================================
    ## *
    ## * getOutputNodeCount
    ## *
    ## *=================================================

    def getOutputNodeCount(self) -> int:
        return self.output_node_count
    
    ## *=================================================
    ## *
    ## * getTotalNodeCount
    ## *
    ## *=================================================

    def getTotalNodeCount(self) -> int:
        return self.getInputNodeCount() + self.getHiddenNodeCount() + self.getOutputNodeCount()