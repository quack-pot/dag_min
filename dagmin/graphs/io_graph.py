from .node import Node
from .input_node import InputNode

class IOGraph:
    ## *=================================================
    ## *
    ## * __init__
    ## *
    ## *=================================================

    def __init__(
        self,
        input_nodes: list[InputNode],
        output_nodes: list[Node],
        intermediate_nodes: list[Node],
    ) -> None:
        self.input_nodes: list[InputNode] = input_nodes
        self.output_nodes: list[Node] = output_nodes
        self.intermediate_nodes: list[Node] = intermediate_nodes

    ## *=================================================
    ## *
    ## * evaluate
    ## *
    ## *=================================================

    def evaluate(self, values: list[float]) -> list[float]:
        for idx in range(min(len(self.input_nodes), len(values))):
            self.input_nodes[idx].setValue(values[idx])

        for output_node in self.output_nodes:
            output_node.evaluate()

        results: list[float] = [self.output_nodes[idx].getValue() for idx in range(len(self.output_nodes))]

        for input_node in self.input_nodes:
            input_node.reset()
        for intermediate_node in self.intermediate_nodes:
            intermediate_node.reset()
        for output_node in self.output_nodes:
            output_node.reset()

        return results
    
    ## *=================================================
    ## *
    ## * getInputNodeCount
    ## *
    ## *=================================================

    def getInputNodeCount(self) -> int:
        return len(self.input_nodes)
    
    ## *=================================================
    ## *
    ## * getOutputNodeCount
    ## *
    ## *=================================================

    def getOutputNodeCount(self) -> int:
        return len(self.output_nodes)
    
    ## *=================================================
    ## *
    ## * getIntermediateNodeCount
    ## *
    ## *=================================================

    def getIntermediateNodeCount(self) -> int:
        return len(self.intermediate_nodes)

    ## *=================================================
    ## *
    ## * getNodeCount
    ## *
    ## *=================================================

    def getNodeCount(self) -> int:
        return len(self.input_nodes) + len(self.output_nodes) + len(self.intermediate_nodes)