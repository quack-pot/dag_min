import math
import dataclasses
from .node import Node

@dataclasses.dataclass
class NeuralNodeIncomingEdge:
    node: Node
    weight: float

class NeuralNode(Node):
    ## *=================================================
    ## *
    ## * __init__
    ## *
    ## *=================================================

    def __init__(self, incoming_edges: list[NeuralNodeIncomingEdge], bias: float) -> None:
        super().__init__()

        self.incoming_edges: list[NeuralNodeIncomingEdge] = incoming_edges
        self.bias: float = bias

    ## *=================================================
    ## *
    ## * __evaluate__
    ## *
    ## *=================================================

    def __evaluate__(self) -> None:
        weighted_sum: float = self.bias
        for incoming_edge in self.incoming_edges:
            incoming_edge.node.evaluate()
            weighted_sum += incoming_edge.weight * incoming_edge.node.getValue()

        self.value = 1 / (1 + math.exp(-weighted_sum))