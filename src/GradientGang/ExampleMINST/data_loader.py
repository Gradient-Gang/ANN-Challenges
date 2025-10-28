import torch
import torch.nn.functional as F
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

# PyTorch Lightning
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger     # http://localhost:6006/
from pytorch_lightning.profilers import PyTorchProfiler
# Metrics
import torchmetrics
from torchmetrics import Metric
# Callbacks
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, Callback


# Custom Accuracy Metric
class MyAccuracy(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("correct", default=torch.tensor(0),
                       dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds, target):
        preds = torch.argmax(preds, dim=1)
        assert preds.shape == target.shape
        self.correct += (preds == target).sum()
        self.total += target.numel()

    def compute(self):
        return self.correct.float() / self.total

# Callbacks


class MyPrintCallback(Callback):
    def on_train_start(self, trainer, pl_module):
        print("Training is starting!")

    def on_train_end(self, trainer, pl_module):
        print("Training has ended!")

# Data Loader


class MyDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size, num_workers):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        datasets.MNIST(self.data_dir, train=True, download=True)
        datasets.MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage):
        entire_dataset = datasets.MNIST(root=self.data_dir,
                                        train=True,
                                        transform=transforms.ToTensor(),
                                        download=False)
        self.train_ds, self.val_ds = random_split(
            entire_dataset, [48000, 12000])
        self.test_ds = datasets.MNIST(root=self.data_dir,
                                      train=False,
                                      transform=transforms.ToTensor(),
                                      download=False)

    def train_dataloader(self):
        return DataLoader(self.train_ds,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_ds,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_ds,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers)

# Neural Network


class NN(pl.LightningModule):
    def __init__(self, input_size, num_classes, hidden_size=50):
        super().__init__()
        # LAYERS
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, num_classes)

        # LOSS FUNCTION & METRICS
        self.loss_fn = nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes)
        self.f1_score = torchmetrics.F1Score(
            task="multiclass", num_classes=num_classes)
        self.my_accuracy = MyAccuracy()

    # MODEL
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = self.layer2(x)
        return x

    # common_step
    def _common_step(self, batch, batch_idx):
        x, y = batch
        x = x.reshape(x.size(0), -1)
        scores = self.forward(x)
        loss = self.loss_fn(scores, y)
        return loss, scores, y

    # training_step
    def training_step(self, batch, batch_idx):
        x, y = batch
        loss, scores, y = self._common_step(batch, batch_idx)
        accuracy = self.accuracy(scores, y)
        f1_score = self.f1_score(scores, y)
        my_accuracy = self.my_accuracy(scores, y)

        self.log_dict(
            {
                'train_loss': loss,
                'train_accuracy': accuracy,
                'train_f1_score': f1_score,
                'train_my_accuracy': my_accuracy
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True
        )

        if batch_idx % 100 == 0:
            x = x[:8]
            grid = torchvision.utils.make_grid(x.view(-1, 1, 28, 28))
            self.logger.experiment.add_image(
                'train_images', grid, self.global_step)

        return {'loss': loss, 'scores': scores, 'y': y}

    # validation_step
    def validation_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch, batch_idx)
        self.log('val_loss', loss, prog_bar=True)
        return loss

    # test_step
    def test_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch, batch_idx)
        self.log('test_loss', loss, prog_bar=True)
        return loss

    # predict_step
    def predict_step(self, batch, batch_idx):
        x, y = batch
        x = x.reshape(x.size(0), -1)
        scores = self.forward(x)
        preds = torch.argmax(scores, dim=1)
        return preds

    # configure_optimizers
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-3)


# device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type == 'cuda':
    num_workers = torch.cuda.device_count()
else:
    num_workers = 0

# hyperparameters
input_size = 28 * 28
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 3

# logger
logger = TensorBoardLogger("logs_tb",
                           name="mnist_model_1"
                           )
# profiler
profiler = PyTorchProfiler(
    on_trace_ready=torch.profiler.tensorboard_trace_handler(
        "logs_tb/profiler0"),
    schedule=torch.profiler.schedule(
        skip_first=10, wait=1, warmup=1, active=20)
)
# network
model = NN(
    input_size=input_size,
    num_classes=num_classes
)
# dataset
dm = MyDataModule(
    data_dir='dataset/',
    batch_size=batch_size,
    num_workers=0
)
# trainer
trainer = pl.Trainer(
    profiler=profiler,
    logger=logger,
    min_epochs=1,
    max_epochs=num_epochs
)

# run
trainer.fit(model, dm)
trainer.validate(model, dm)
trainer.test(model, dm)
