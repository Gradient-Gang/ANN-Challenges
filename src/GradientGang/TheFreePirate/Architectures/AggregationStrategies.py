from abc import ABC, abstractmethod
import torch
import warnings


class ClassificationAggregationStrategy(ABC):
    @abstractmethod
    def aggregate(self, logits: torch.Tensor) -> torch.Tensor:
        pass


class MajorityVotingAggregationStrategy(ClassificationAggregationStrategy):
    def aggregate(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Aggregate logits using majority voting.

        Args:
            logits (torch.Tensor): Logits tensor of shape (..., nModels, num_classes).

        Returns:
            torch.Tensor: Aggregated class predictions
        """
        if logits.shape[-2] % logits.shape[-1] == 0:
            warnings.warn(
                "Majority voting may lead to ties when the number of models is a multiple of the number of classes."
            )

        # Get predicted classes from logits
        predicted_classes = torch.argmax(logits, dim=-1)  # (..., nModels)

        # Perform majority voting along the nModels dimension
        majority_votes, _ = torch.mode(predicted_classes, dim=-1)

        return majority_votes


class AverageLogitsAggregationStrategy(ClassificationAggregationStrategy):
    def aggregate(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Aggregate logits by averaging.

        Args:
            logits (torch.Tensor): Logits tensor of shape (..., nModels, num_classes).

        Returns:
            torch.Tensor: Aggregated class predictions
        """
        # Average logits along the nModels dimension
        avg_logits = torch.mean(logits, dim=-2)  # (..., num_classes)

        return avg_logits


class EntropyWeightedMajorityVotingAggregationStrategy(
    ClassificationAggregationStrategy
):
    def computeEntropy(self, logits: torch.Tensor):
        """
        Compute entropy for each model's logits.

        Args:
            logits (torch.Tensor): Logits tensor of shape (..., nModels, num_classes).
        Returns:
            torch.Tensor: Entropy tensor of shape (..., nModels).
        """
        logProbs = torch.nn.functional.log_softmax(logits, dim=-1)
        entropy = -torch.sum(torch.exp(logProbs) * logProbs, dim=-1)  # (..., nModels)
        return entropy

    def computeWeights(self, entropy: torch.Tensor):
        """
        Compute weights based on entropy.

        Args:
            entropy (torch.Tensor): Entropy tensor of shape (..., nModels).
        Returns:
            torch.Tensor: Weights tensor of shape (..., nModels).
        """
        invEntropy = 1.0 / (entropy + 1e-8)  # Avoid division by zero
        weights = invEntropy / torch.sum(
            invEntropy, dim=-1, keepdim=True
        )  # Normalize weights
        return weights

    def aggregate(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Aggregate logits using entropy-weighted averaging.

        Args:
            logits (torch.Tensor): Logits tensor of shape (..., nModels, num_classes).

        Returns:
            torch.Tensor: Aggregated class predictions
        """
        entropy = self.computeEntropy(logits)  # (..., nModels)
        weights = self.computeWeights(entropy)  # (..., nModels)

        # Get predicted classes from logits
        predicted_classes = torch.argmax(logits, dim=-1)  # (..., nModels)

        # Perform weighted voting using one-hot encoding
        num_classes = logits.shape[-1]
        nModels = predicted_classes.shape[-1]

        # Convert predictions to one-hot: (..., nModels, num_classes)
        one_hot = torch.nn.functional.one_hot(
            predicted_classes, num_classes=num_classes
        ).float()

        # Weight each one-hot vector: (..., nModels, num_classes) * (..., nModels, 1)
        weighted_one_hot = one_hot * weights.unsqueeze(-1)

        # Sum across models: (..., num_classes)
        weighted_votes = weighted_one_hot.sum(dim=-2)

        majority_votes = torch.argmax(weighted_votes, dim=-1)  # (...,)
        return majority_votes
