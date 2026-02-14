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

    lambda_edges: float = 1e-4
    lambda_nodes: float = 1e-3
    lambda_discrete: float = 1e-5
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
        lambda_discrete: float,
        node_threshold_strength: float
    ) -> float:
        optimizer.zero_grad()

        output: torch.Tensor = self.__internalEvaluateTensor__(data)
        loss: torch.Tensor = computeLoss(output, target)

        outgoing_edges: torch.Tensor = torch.abs(self.weights * self.mask)

        if lambda_edges != 0.0:
            active_edge_count: torch.Tensor = outgoing_edges.sum()
            loss += lambda_edges * active_edge_count

        if lambda_nodes != 0.0:
            outgoing_sums: torch.Tensor = torch.sum(outgoing_edges, dim=1)
            node_activity: torch.Tensor = 1 - torch.exp(-node_threshold_strength * outgoing_sums)
            active_node_count: torch.Tensor = node_activity.sum()
            loss += lambda_nodes * active_node_count

        if lambda_discrete != 0.0:
            binary_penalty: torch.Tensor = (output * (1.0 - output)).sum()
            loss += lambda_discrete * binary_penalty

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
                        train_info.lambda_discrete,
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
                        train_info.lambda_discrete,
                        train_info.node_threshold_strength,
                    )

    ## *=================================================
    ## *
    ## * extractDAG
    ## *
    ## *=================================================

    def extractDAG(self, activity_threshold: float = 0.1) -> graphs.Graph:
        return graphs.Graph(graphs.GraphCreateInfo(
            self.input_node_count,
            self.output_node_count,
            self.weights * self.mask,
            self.biases,
            activity_threshold
        ))