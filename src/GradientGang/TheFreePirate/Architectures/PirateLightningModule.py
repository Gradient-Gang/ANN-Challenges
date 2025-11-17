import pytorch_lightning
import torch
from .Models.FeedForwardModel import FeedForwardModel
from .AggregationStrategies import (
    AverageLogitsAggregationStrategy,
    MajorityVotingAggregationStrategy,
)
from ..Utils import getRaise
from torchmetrics import F1Score
from copy import deepcopy
from ..Lookups import classificationAggregationStrategies


class PirateLightningModule(pytorch_lightning.LightningModule):
    def __init__(self, params: dict, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters(params)
        params = deepcopy(params)
        paramsName = "modelParams in LightningModule.__init__"

        # Get aggregation strategy
        aggregationStrategyForClassification = getRaise(
            params, "aggregationStrategyForClassification", paramsName
        )

        self.aggregationStrategyForLogits = AverageLogitsAggregationStrategy()
        self.aggregationStrategyForClassification = getRaise(
            classificationAggregationStrategies,
            aggregationStrategyForClassification,
            "Aggregation Strategy Lookup",
        )()

        # Setup Losses
        self.classificationLoss = torch.nn.CrossEntropyLoss()
        self.reconstructionLoss = torch.nn.MSELoss()

        # Get Dataset Info
        self.datasetInfo = getRaise(params, "dataModuleInfo", paramsName)

        # Setup F1
        self.f1AverageStrategy = getRaise(params, "f1AverageStrategy", paramsName)
        self.numClasses = getRaise(params, "numClasses", paramsName)
        self.f1 = F1Score(
            task="multiclass",
            num_classes=self.numClasses,
            average=self.f1AverageStrategy,
        )

        # Get Reconstruction Loss weight
        self.reconstructionLossWeight = getRaise(
            params, "reconstructionLossWeight", paramsName
        )

        # Get global feature usage
        self.useGlobalFeatures = getRaise(params, "useGlobalFeatures", paramsName)

        # Get Learning parameters
        self.learningRate = getRaise(params, "learningRate", paramsName)
        self.weightDecay = getRaise(params, "weightDecay", paramsName)

        # Setup models
        self.modelSetup(params)

    def modelSetup(self, params: dict):
        paramsName = "modelParams in LightningModule.modelSetup"

        # EXTRACT MODEL CONFIGURATIONS
        self.activationFunction = getRaise(params, "activationFunction", paramsName)
        # Time Series
        self.timeSeriesShape = self.datasetInfo["shapes"]["train"]["timeSeriesShape"]
        self.timeSeriesEmbeddingDim = getRaise(
            params, "timeSeriesEmbeddingDim", paramsName
        )
        self.timeSeriesEncoderNumLayers = getRaise(
            params, "timeSeriesEncoderNumLayers", paramsName
        )
        self.timeSeriesDropout = getRaise(params, "timeSeriesDropout", paramsName)

        # Predictor
        self.predictorNumLayers = getRaise(params, "predictorNumLayers", paramsName)
        self.predictorDropout = getRaise(params, "predictorDropout", paramsName)
        self.predictorFixALogit = getRaise(params, "predictorFixALogit", paramsName)
        self.predictorLogitToFix = (
            None
            if not self.predictorFixALogit
            else getRaise(params, "predictorLogitToFix", paramsName)
        )

        # Global Features
        if self.useGlobalFeatures:
            self.numGlobalFeatures = self.datasetInfo["shapes"]["train"][
                "globalFeaturesShape"
            ][-1]
            self.globalFeaturesEmbeddingDim = getRaise(
                params, "globalFeaturesEmbeddingDim", paramsName
            )
            self.globalFeaturesEncoderNumLayers = getRaise(
                params, "globalFeaturesEncoderNumLayers", paramsName
            )
            self.globalFeaturesDropout = getRaise(
                params, "globalFeaturesDropout", paramsName
            )

        else:
            self.globalFeaturesEmbeddingDim = 0

        # SETUP MODELS
        # Setup time series encoder
        timeSeriesDim = self.timeSeriesShape[-1] * self.timeSeriesShape[-2]
        self.timeSeriesEncoder = FeedForwardModel.linearlyInterpolateLayers(
            inputDim=timeSeriesDim,
            outputDim=self.timeSeriesEmbeddingDim,
            nLayers=self.timeSeriesEncoderNumLayers,
            dropoutProb=self.timeSeriesDropout,
            activation=self.activationFunction,
        )

        # Setup classifier
        self.classifier = FeedForwardModel.linearlyInterpolateLayers(
            inputDim=self.timeSeriesEmbeddingDim + self.globalFeaturesEmbeddingDim,
            outputDim=self.numClasses,
            nLayers=self.predictorNumLayers,
            dropoutProb=self.predictorDropout,
            activation=self.activationFunction,
            logitToFix=self.predictorLogitToFix,
        )

        # Setup time series decoder
        self.timeSeriesDecoder = FeedForwardModel.linearlyInterpolateLayers(
            inputDim=self.timeSeriesEmbeddingDim,
            outputDim=timeSeriesDim,
            nLayers=self.timeSeriesEncoderNumLayers,
            dropoutProb=self.timeSeriesDropout,
            activation=self.activationFunction,
        )

        if not self.useGlobalFeatures:
            return

        # Setup global features encoder
        self.globalFeaturesEncoder = FeedForwardModel.linearlyInterpolateLayers(
            inputDim=self.numGlobalFeatures,
            outputDim=self.globalFeaturesEmbeddingDim,
            nLayers=self.globalFeaturesEncoderNumLayers,
            dropoutProb=self.globalFeaturesDropout,
            activation=self.activationFunction,
        )

        # Setup global features decoder
        self.globalFeaturesDecoder = FeedForwardModel.linearlyInterpolateLayers(
            inputDim=self.globalFeaturesEmbeddingDim,
            outputDim=self.numGlobalFeatures,
            nLayers=self.globalFeaturesEncoderNumLayers,
            dropoutProb=self.globalFeaturesDropout,
            activation=self.activationFunction,
        )

    def encode(self, timeSeries, globalFeatures=None, nWindows: int = 1):
        # Encode time Series of shape (batchSize*nWindows, windowSize*nFeatures)
        timeSeriesEncoded = self.timeSeriesEncoder(timeSeries)

        if not self.useGlobalFeatures:
            return timeSeriesEncoded, None, timeSeriesEncoded

        # Encode global features of shape (batchSize, nGlobalFeatures)
        globalFeaturesEncoded = self.globalFeaturesEncoder(globalFeatures)

        # Combine embeddings by repeating the global embedding per window
        globalFeaturesEncodedRepeated = globalFeaturesEncoded.repeat_interleave(
            nWindows, dim=0
        )

        # Create combined embeddings
        embeddings = torch.cat(
            (timeSeriesEncoded, globalFeaturesEncodedRepeated), dim=1
        )

        return timeSeriesEncoded, globalFeaturesEncoded, embeddings

    def predict(self, embeddings):
        # Compute predictions
        predictions = self.classifier(embeddings)  # (batchSize * nWindows, nClasses)
        return predictions

    def decode(self, timeSeriesEncoded, globalFeaturesEncoded=None):
        # Decode time series
        timeSeriesReconstructed = self.timeSeriesDecoder(timeSeriesEncoded)

        if not self.useGlobalFeatures:
            return timeSeriesReconstructed, None

        # Decode global features
        globalFeaturesReconstructed = self.globalFeaturesDecoder(globalFeaturesEncoded)

        return timeSeriesReconstructed, globalFeaturesReconstructed

    def encodeDecode(self, timeSeries, globalFeatures=None, nWindows: int = 1):
        timeSeriesEncoded, globalFeaturesEncoded, embeddings = self.encode(
            timeSeries, globalFeatures, nWindows
        )
        timeSeriesReconstructed, globalFeaturesReconstructed = self.decode(
            timeSeriesEncoded, globalFeaturesEncoded
        )

        return (
            embeddings,
            timeSeriesReconstructed,
            globalFeaturesReconstructed,
        )

    def forward(self, x):
        timeSeries, globalFeatures = x

        # Reshape time Series
        batchSize, nWindows, windowSize, nFeatures = timeSeries.shape
        timeSeries = timeSeries.view(batchSize * nWindows, windowSize * nFeatures)

        # Encode inputs
        timeSeriesEncoded, globalFeaturesEncoded, combinedEmbedding = self.encode(
            timeSeries, globalFeatures, nWindows
        )

        # Compute predictions
        predictions = self.predict(
            combinedEmbedding
        )  # (batchSize * nWindows, nClasses)

        # Aggregate predictions back to original samples
        predictions = predictions.view(
            batchSize, nWindows, -1
        )  # (batchSize, nWindows, nClasses)
        aggregatedPredictions = self.aggregationStrategyForLogits.aggregate(
            predictions
        )  # (batchSize, nClasses)

        # Compute reconstructions
        timeSeriesReconstructed = self.timeSeriesDecoder(timeSeriesEncoded)
        timeSeriesReconstructed = timeSeriesReconstructed.view(
            batchSize, nWindows, windowSize, nFeatures
        )

        if not self.useGlobalFeatures:
            return aggregatedPredictions, timeSeriesReconstructed, None

        globalFeaturesReconstructed = self.globalFeaturesDecoder(globalFeaturesEncoded)

        return (
            aggregatedPredictions,
            timeSeriesReconstructed,
            globalFeaturesReconstructed,
        )

    def training_step(self, batch, batchIdx):
        (timeSeries, globalFeatures), labels = batch

        # Compute Labelled Mask
        labelledMask = labels >= 0
        labels = labels[labelledMask]

        # Reshape Time Series
        batchSize, nWindows, windowSize, nFeatures = timeSeries.shape
        timeSeries = timeSeries.view(batchSize * nWindows, windowSize * nFeatures)

        # Autoencoder Pass
        (
            combinedEmbedding,
            timeSeriesReconstructed,
            globalFeaturesReconstructed,
        ) = self.encodeDecode(timeSeries, globalFeatures, nWindows)

        if labels.size(0) != 0:
            # Prediction Pass
            extendedLabelledMask = labelledMask.repeat_interleave(nWindows)
            labeledEmbeddings = combinedEmbedding[extendedLabelledMask]
            predictions = self.predict(labeledEmbeddings)
            # Aggregate predictions
            predictions = predictions.view(
                -1, nWindows, predictions.size(-1)
            )  # (batchSize_labeled, nWindows, nClasses)
            predictionsLogits = self.aggregationStrategyForLogits.aggregate(predictions)

            # Compute losses
            classificationLoss = self.classificationLoss(predictionsLogits, labels)
            predictedLabels = self.aggregationStrategyForClassification.aggregate(
                predictions
            )
        else:
            classificationLoss = torch.tensor(0.0, device=self.device)
            predictionsLogits = torch.zeros((0, self.numClasses), device=self.device)
            predictedLabels = torch.zeros((0,), dtype=torch.long, device=self.device)

        timeSeries = timeSeries.view_as(timeSeriesReconstructed)
        reconstructionLossTimeSeries = self.reconstructionLoss(
            timeSeriesReconstructed, timeSeries
        )

        if not self.useGlobalFeatures:
            reconstructionLossGlobalFeatures = 0
        else:
            reconstructionLossGlobalFeatures = self.reconstructionLoss(
                globalFeaturesReconstructed, globalFeatures
            )

        reconstructionLoss = (
            reconstructionLossTimeSeries + reconstructionLossGlobalFeatures
        )

        totalLoss = (
            1 - self.reconstructionLossWeight
        ) * classificationLoss + self.reconstructionLossWeight * reconstructionLoss

        # Log losses
        self.log("train/classification_loss", classificationLoss)
        self.log("train/reconstruction_loss_time_series", reconstructionLossTimeSeries)
        self.log("train/reconstruction_loss", reconstructionLoss)
        self.log("train/total_loss", totalLoss, prog_bar=True)

        if labels.size(0) != 0:
            self.log("train/f1_score", self.f1(predictedLabels, labels), prog_bar=True)

        if self.useGlobalFeatures:
            self.log(
                "train/reconstruction_loss_global_features",
                reconstructionLossGlobalFeatures,
            )

        return totalLoss

    def validation_step(self, batch, batchIdx):
        (timeSeries, globalFeatures), labels = batch

        # Compute Labelled Mask
        labelledMask = labels >= 0
        labels = labels[labelledMask]

        # Reshape Time Series
        batchSize, nWindows, windowSize, nFeatures = timeSeries.shape
        timeSeries = timeSeries.view(batchSize * nWindows, windowSize * nFeatures)

        # Autoencoder Pass
        (
            combinedEmbedding,
            timeSeriesReconstructed,
            globalFeaturesReconstructed,
        ) = self.encodeDecode(timeSeries, globalFeatures, nWindows)

        if labels.size(0) != 0:
            # Prediction Pass
            extendedLabelledMask = labelledMask.repeat_interleave(nWindows)
            labeledEmbeddings = combinedEmbedding[extendedLabelledMask]
            predictions = self.predict(labeledEmbeddings)

            # Aggregate predictions
            predictions = predictions.view(-1, nWindows, predictions.size(-1))
            predictionsLogits = self.aggregationStrategyForLogits.aggregate(predictions)

            # Compute losses
            classificationLoss = self.classificationLoss(predictionsLogits, labels)
            predictedLabels = self.aggregationStrategyForClassification.aggregate(
                predictions
            )
        else:
            classificationLoss = torch.tensor(0.0, device=self.device)
            predictionsLogits = torch.zeros((0, self.numClasses), device=self.device)
            predictedLabels = torch.zeros((0,), dtype=torch.long, device=self.device)

        timeSeries = timeSeries.view_as(timeSeriesReconstructed)
        reconstructionLossTimeSeries = self.reconstructionLoss(
            timeSeriesReconstructed, timeSeries
        )

        if not self.useGlobalFeatures:
            reconstructionLossGlobalFeatures = 0
        else:
            reconstructionLossGlobalFeatures = self.reconstructionLoss(
                globalFeaturesReconstructed, globalFeatures
            )

        reconstructionLoss = (
            reconstructionLossTimeSeries + reconstructionLossGlobalFeatures
        )

        totalLoss = (
            1 - self.reconstructionLossWeight
        ) * classificationLoss + self.reconstructionLossWeight * reconstructionLoss

        # Log losses
        self.log("val/classification_loss", classificationLoss)
        self.log("val/reconstruction_loss_time_series", reconstructionLossTimeSeries)
        self.log("val/reconstruction_loss", reconstructionLoss)
        self.log("val/total_loss", totalLoss, prog_bar=True)

        if labels.size(0) != 0:
            self.log("val/f1_score", self.f1(predictedLabels, labels), prog_bar=True)

        if self.useGlobalFeatures:
            self.log(
                "val/reconstruction_loss_global_features",
                reconstructionLossGlobalFeatures,
            )

        return totalLoss

    def configure_optimizers(self):  # type: ignore
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.learningRate, weight_decay=self.weightDecay
        )

        # Return a tuple of optimizer and scheduler
        return optimizer
