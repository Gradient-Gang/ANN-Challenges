import torch
from ...Lookups import activationFunctionsLookup


class FeedForwardModel(torch.nn.Module):
    @staticmethod
    def linearlyInterpolateLayers(
        inputDim,
        outputDim,
        nLayers: int,
        activation: str,
        dropoutProb: float,
        logitToFix: int | None = None,
    ):
        """
        Create a feed-forward network with linearly interpolated hidden layer sizes.

        Args:
            inputDim (int): Dimension of the input features.
            outputDim (int): Dimension of the output layer.
            nLayers (int): Number of hidden layers.
            activation (str): Activation function to use.
            dropoutProb (float): Dropout probability.

        Returns:
            torch.nn.Sequential: The constructed feed-forward network.
        """
        if nLayers < 1:
            raise ValueError("nLayers must be at least 1")

        hiddenDims = []
        for i in range(nLayers):
            ratio = (i + 1) / (nLayers + 1)
            hiddenDim = int(inputDim + ratio * (outputDim - inputDim))
            hiddenDims.append(hiddenDim)
        return FeedForwardModel(
            inputDim=inputDim,
            hiddenDims=hiddenDims,
            outputDim=outputDim,
            activation=activation,
            dropoutProb=dropoutProb,
            logitToFix=logitToFix,
        )

    def __init__(
        self,
        inputDim: int,
        hiddenDims: list[int],
        outputDim: int,
        activation: str = "relu",
        dropoutProb: float = 0.0,
        logitToFix: int | None = None,
    ):
        """
        FeedForwardModel constructs a feed-forward neural network.

        Args:
            inputDim (int): Dimension of the input features.
            hiddenDims (list[int]): List of hidden layer dimensions.
            outputDim (int): Dimension of the output layer.
            activation (str): Activation function to use (default: "relu").
            dropoutProb (float): Dropout probability (default: 0.0).
        """
        super(FeedForwardModel, self).__init__()
        self.inputDim = inputDim
        self.hiddenDims = hiddenDims
        self.outputDim = outputDim - (logitToFix is not None)
        self.activation = activation
        self.dropoutProb = dropoutProb
        self.logitToFix = logitToFix

        layers = []
        prevDim = inputDim

        # Create hidden layers
        for hiddenDim in hiddenDims:
            layers.append(torch.nn.Linear(prevDim, hiddenDim))
            layers.append(activationFunctionsLookup[activation]())
            if dropoutProb > 0.0:
                layers.append(torch.nn.Dropout(dropoutProb))
            prevDim = hiddenDim

        # Output layer
        layers.append(torch.nn.Linear(prevDim, self.outputDim))

        self.network = torch.nn.Sequential(*layers)
        self.heInitialize()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the feed-forward network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, inputDim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, outputDim).
        """
        logits = self.network(x)

        if self.logitToFix is not None:
            # Fix the specified logit to zero
            # Logits (batch_size, outputDim + 1)
            fixedLogits = torch.zeros_like(logits[:, 0], device=logits.device)
            logits = torch.cat(
                [
                    logits[:, : self.logitToFix],
                    fixedLogits.unsqueeze(-1),
                    logits[:, self.logitToFix :],
                ],
                dim=-1,
            )

        return logits

    def heInitialize(self):
        """
        Initialize weights using He initialization.
        """
        for m in self.network:
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight, nonlinearity=self.activation)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0.0)


class FeedForwardAutoencoder(torch.nn.Module):
    @staticmethod
    def linearlyInterpolate(
        inputDim,
        embeddingDim,
        nLayers: int,
        activation: str,
        dropoutProb: float,
    ):
        """
        Create a feed-forward autoencoder with linearly interpolated hidden layer sizes.
        Args:
            inputDim (int): Dimension of the input features.
            outputDim (int): Dimension of the output layer.
            nLayers (int): Number of hidden layers.
            activation (str): Activation function to use.
            dropoutProb (float): Dropout probability.
        Returns:
            FFAutoencoder: The constructed feed-forward autoencoder.
        """

        encoder = FeedForwardModel.linearlyInterpolateLayers(
            inputDim,
            embeddingDim,
            nLayers,
            activation,
            dropoutProb,
        )

        decoder = FeedForwardModel.linearlyInterpolateLayers(
            embeddingDim,
            inputDim,
            nLayers,
            activation,
            dropoutProb,
        )

        return FeedForwardAutoencoder(encoder, decoder)

    def __init__(self, encoder: FeedForwardModel, decoder: FeedForwardModel):
        """
        FFAutoencoder constructs a feed-forward autoencoder.

        Args:
            encoder (FeedForwardModel): The encoder part of the autoencoder.
            decoder (FeedForwardModel): The decoder part of the autoencoder.
        """
        super(FeedForwardAutoencoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the autoencoder.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, inputDim).

        Returns:
            torch.Tensor: Reconstructed tensor of shape (batch_size, inputDim).
        """
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed
