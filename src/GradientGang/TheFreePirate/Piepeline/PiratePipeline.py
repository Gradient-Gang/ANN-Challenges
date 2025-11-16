from ..DataHandling.PirateDataModule import PirateDataModule
from ..DataHandling.DataAugmentation.TensorAugmenter import (
    TensorAugmenter,
    ScaleAugmenter,
    JitterAugmenter,
    OffsetAugmenter,
    TimeWarpAugmenter,
    SequentialAugmenter,
    WindowingAugmenter,
)
from ..Utils import getRaise


class PiratePipeline:
    def __init__(self, dataParams: dict, *args, **kwargs):
        self.dataModule = PirateDataModule(**dataParams)

    def setupTrial(self, params: dict):
        self.dataSetup(**params)

    def evaluateParameters(self, params: dict) -> float:

        return 0.0

    def dataSetup(self, dataParams: dict, *args, **kwargs):
        dictionaryName = "dataParams in PiratePipeline.dataSetup"
        numFold = getRaise(dataParams, "numFolds", dictionaryName)
        includeTestInFolds = getRaise(dataParams, "includeTestInFolds", dictionaryName)
        augmentTestSet = getRaise(dataParams, "augmentTestSet", dictionaryName)

        # Set up augmentations
        nCopies = getRaise(dataParams, "augmentation-nCopies", dictionaryName)
        keepOriginal = getRaise(dataParams, "augmentation-keepOriginal", dictionaryName)
        scaleRange = getRaise(dataParams, "augmentation-scaleRange", dictionaryName)
        jitterStdDev = getRaise(dataParams, "augmentation-jitterStdDev", dictionaryName)
        offsetRange = getRaise(dataParams, "augmentation-offsetRange", dictionaryName)
        maxWarpFraction = getRaise(
            dataParams, "augmentation-maxWarpFraction", dictionaryName
        )

        # Set up Sequential Augmenter
        augmenters: list[TensorAugmenter] = [
            ScaleAugmenter(rangeScale=scaleRange),
            JitterAugmenter(jitterStdDev=jitterStdDev),
            OffsetAugmenter(rangeOffset=offsetRange),
            TimeWarpAugmenter(maxWarpFraction=maxWarpFraction),
        ]

        augmenter = SequentialAugmenter(
            nCopies=nCopies,
            keepOriginal=keepOriginal,
            augmenters=augmenters,
        )

        # Set up Windowing
        windowSize = getRaise(dataParams, "augmentation-windowSize", dictionaryName)
        windowStride = getRaise(dataParams, "augmentation-windowStride", dictionaryName)

        windowingAugmenter = WindowingAugmenter(
            windowSize=windowSize,
            stride=windowStride,
        )

        # Setup K-Folds with augmentations
        self.dataModule.setupKFolds(
            nFolds=numFold,
            dataAugmenter=augmenter,
            windowingAugmenter=windowingAugmenter,
            includeTestInFolds=includeTestInFolds,
            augmentTestSet=augmentTestSet,
        )
