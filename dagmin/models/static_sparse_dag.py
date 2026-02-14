import torch
import dataclasses
from .. import graphs

@dataclasses.dataclass
class StaticSparseDAGCreateInfo:
    input_node_count: int = 2
    hidden_node_count: int = 5
    output_node_count: int = 1

@dataclasses.dataclass
class StaticSparseDAGTrainInfo:
    input_data: list[list[float]]
    output_data: list[list[float]]

    epochs: int = 5000
    epoch_print_cadence: int = 500

    learning_rate: float = 0.05

    lambda_edges: float = 1e-5
    lambda_nodes: float = 1.35e-4
    node_threshold_strength: float = 10.0

class StaticSparseDAG():
    ## *=================================================
    ## *
    ## * __init__
    ## *
    ## *=================================================

    def __init__(self, create_info: StaticSparseDAGCreateInfo) -> None:
        super().__init__()

        # ? Static Sparse DAG Model Parameters

        self.input_node_count: int = int(max(0, create_info.input_node_count))
        self.hidden_node_count: int = int(max(0, create_info.hidden_node_count))
        self.output_node_count: int = int(max(1, create_info.output_node_count))
        self.total_node_count: int = self.input_node_count + self.hidden_node_count + self.output_node_count

        # ? Weights and Biases (Initialized Randomly)

        self.weights: torch.nn.Parameter = torch.nn.Parameter(
            torch.rand(self.total_node_count, self.total_node_count),
        )
        self.biases: torch.nn.Parameter = torch.nn.Parameter(
            torch.rand(self.total_node_count),
        )

        # ? Upper Triangular Mask (Enforces DAG Structure)

        self.mask: torch.Tensor = torch.triu(
            torch.ones(self.total_node_count, self.total_node_count),
            diagonal=1,
        )

    ## *=================================================
    ## *
    ## * __internalEvaluateTensor__
    ## *
    ## *=================================================

    def __internalEvaluateTensor__(self, values: torch.Tensor) -> torch.Tensor:
        dtype: torch.dtype = self.weights.dtype
        device: torch.device = self.weights.device

        states: list[torch.Tensor] = []

        input_size: int = min(self.input_node_count, values.numel())
        for idx in range(input_size):
            states.append(values[idx])

        for idx in range(input_size, self.input_node_count):
            states.append(torch.zeros((), dtype=dtype, device=device))

        for idx in range(self.input_node_count, self.total_node_count):
            prev_states: torch.Tensor = torch.stack(states)

            weights: torch.Tensor = self.weights[:idx, idx] * self.mask[:idx, idx]
            weighted_sum: torch.Tensor = torch.dot(weights, prev_states)

            node_value: torch.Tensor = torch.sigmoid(weighted_sum + self.biases[idx])
            states.append(node_value)

        return torch.stack(states[-self.output_node_count:])
    
    ## *=================================================
    ## *
    ## * evaluateTensor
    ## *
    ## *=================================================

    def evaluateTensor(self, values: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.__internalEvaluateTensor__(values)

    ## *=================================================
    ## *
    ## * evaluate
    ## *
    ## *=================================================

    def evaluate(self, values: list[float]) -> list[float]:
        dtype: torch.dtype = self.weights.dtype
        device: torch.device = self.weights.device

        if len(values) > 0:
            values_tensor: torch.Tensor = torch.tensor(values, dtype=dtype, device=device)
            return self.evaluateTensor(values_tensor).tolist()
        
        return self.evaluateTensor(torch.empty(0, dtype=dtype, device=device)).tolist()
        
    ## *=================================================
    ## *
    ## * __trainEpoch__
    ## *
    ## *=================================================

    def __trainEpoch__(
        self,
        data: torch.Tensor,
        target: torch.Tensor,
        computeLoss: torch.nn.BCELoss,
        optimizer: torch.optim.Adam,
        lambda_edges: float,
        lambda_nodes: float,
        node_threshold_strength: float
    ) -> float:
        optimizer.zero_grad()

        output: torch.Tensor = self.__internalEvaluateTensor__(data)
        loss: torch.Tensor = computeLoss(output, target)

        outgoing_edges: torch.Tensor = torch.abs(self.weights * self.mask)

        active_edge_count: torch.Tensor = outgoing_edges.sum()
        loss += lambda_edges * active_edge_count

        outgoing_sums: torch.Tensor = torch.sum(outgoing_edges, dim=1)
        node_activity = 1 - torch.exp(-node_threshold_strength * outgoing_sums)
        active_node_count: torch.Tensor = node_activity.sum()
        loss += lambda_nodes * active_node_count

        loss.backward()
        optimizer.step()

        return float(loss.item())

    ## *=================================================
    ## *
    ## * train
    ## *
    ## *=================================================

    def train(self, train_info: StaticSparseDAGTrainInfo) -> None:
        assert (
            len(train_info.input_data) == len(train_info.output_data)
        ), f"Input and output sets do not match in length! (Input = {len(train_info.input_data)}; Output = {len(train_info.output_data)})"

        dtype: torch.dtype = self.weights.dtype
        device: torch.device = self.weights.device

        data_set: torch.Tensor = torch.tensor(train_info.input_data, dtype=dtype, device=device)
        targets: torch.Tensor = torch.tensor(train_info.output_data, dtype=dtype, device=device)

        computeLoss: torch.nn.BCELoss = torch.nn.BCELoss()
        optimizer = torch.optim.Adam([self.weights, self.biases], lr=train_info.learning_rate)

        if train_info.epoch_print_cadence > 0:
            for epoch in range(train_info.epochs):
                total_loss: float = 0.0

                for x, y in zip(data_set, targets):
                    total_loss += self.__trainEpoch__(
                        x,
                        y,
                        computeLoss,
                        optimizer,
                        train_info.lambda_edges,
                        train_info.lambda_nodes,
                        train_info.node_threshold_strength,
                    )

                if epoch % train_info.epoch_print_cadence == 0:
                    print(f"StaticSparseDAG Training (Epoch = {epoch}; Loss = {total_loss:.4f})")
        else:
            for epoch in range(train_info.epochs):
                for x, y in zip(data_set, targets):
                    self.__trainEpoch__(
                        x,
                        y,
                        computeLoss,
                        optimizer,
                        train_info.lambda_edges,
                        train_info.lambda_nodes,
                        train_info.node_threshold_strength,
                    )

    ## *=================================================
    ## *
    ## * extractNeuralNode
    ## *
    ## *=================================================

    def extractNeuralNode(
        self,
        idx: int,
        idx_to_node: dict[int, graphs.Node],
        activity_threshold: float,
    ) -> graphs.NeuralNode | None:
        if idx in idx_to_node:
            return None

        incoming_edges: list[graphs.NeuralNodeIncomingEdge] = []
        bias: float = float(self.biases[idx].item())

        edge_weights: torch.Tensor = self.weights[:idx, idx] * self.mask[:idx, idx]
        for jdx in range(idx):
            weight: float = float(edge_weights[jdx].item())
            if abs(weight) < activity_threshold:
                continue

            existing_node: graphs.Node | None = idx_to_node.get(jdx)
            if not existing_node is None:
                incoming_edges.append(graphs.NeuralNodeIncomingEdge(existing_node, weight))
                continue

            existing_node = self.extractNeuralNode(jdx, idx_to_node, activity_threshold)
            if existing_node is None:
                continue

            incoming_edges.append(graphs.NeuralNodeIncomingEdge(existing_node, weight))

        neural_node: graphs.NeuralNode = graphs.NeuralNode(incoming_edges, bias)
        idx_to_node[idx] = neural_node

        return neural_node

    ## *=================================================
    ## *
    ## * extractDAG
    ## *
    ## *=================================================

    def extractDAG(self, activity_threshold: float = 0.1) -> graphs.IOGraph:
        input_nodes: list[graphs.InputNode] = []
        hidden_nodes: list[graphs.Node] = []
        output_nodes: list[graphs.Node] = []

        idx_to_node: dict[int, graphs.Node] = {}

        for idx in range(0, self.input_node_count):
            input_node: graphs.InputNode = graphs.InputNode()

            idx_to_node[idx] = input_node

            input_nodes.append(input_node)
        
        for idx in range(self.input_node_count + self.hidden_node_count, self.total_node_count):
            output_node: graphs.NeuralNode | None = self.extractNeuralNode(idx, idx_to_node, activity_threshold)

            if output_node is None:
                continue

            output_nodes.append(output_node)

        for idx in range(self.input_node_count, self.total_node_count - self.output_node_count):
            hidden_node: graphs.Node | None = idx_to_node.get(idx)

            if hidden_node is None:
                continue

            hidden_nodes.append(hidden_node)

        return graphs.IOGraph(
            input_nodes,
            output_nodes,
            hidden_nodes,
        )