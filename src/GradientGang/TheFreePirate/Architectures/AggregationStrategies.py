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
