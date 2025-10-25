import torch
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

# PyTorch Lightning
import pytorch_lightning as pl
# Metrics
import torchmetrics
from torchmetrics import Metric

# Custom Accuracy Metric
class MyAccuracy(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds, target):
        preds = torch.argmax(preds, dim=1)
        assert preds.shape == target.shape
        self.correct += (preds == target).sum()
        self.total += target.numel()

    def compute(self):
        return self.correct.float() / self.total

class NN(pl.LightningModule):
    def __init__(self, input_size, num_classes, hidden_size=50): 
        super().__init__()
        #LAYERS
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, num_classes)

        #LOSS FUNCTION & METRICS
        self.loss_fn = nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.f1_score = torchmetrics.F1Score(task="multiclass", num_classes=num_classes)
        self.my_accuracy = MyAccuracy()

    #MODEL
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
    
    #training_step
    def training_step(self, batch, batch_idx):  
        loss, scores, y = self._common_step(batch, batch_idx)
        accuracy = self.accuracy(scores, y)
        f1_score = self.f1_score(scores, y)
        my_accuracy = self.my_accuracy(scores, y)
        self.log_dict({'train_loss': loss, 'train_accuracy': accuracy, 'train_f1_score': f1_score, 'train_my_accuracy': my_accuracy},
                      on_step=False, on_epoch=True, prog_bar=True)
        return {'loss': loss, 'scores': scores, 'y': y}

    #validation_step
    def validation_step(self, batch, batch_idx):  
        loss, scores, y = self._common_step(batch, batch_idx)
        self.log('val_loss', loss, prog_bar=True)
        return loss
    
    #test_step
    def test_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch, batch_idx)
        self.log('test_loss', loss, prog_bar=True)
        return loss
    
    #predict_step
    def predict_step(self, batch, batch_idx):
        x, y = batch
        x = x.reshape(x.size(0), -1)
        scores = self.forward(x)
        preds = torch.argmax(scores, dim=1)
        return preds

    #configure_optimizers
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-3)

#device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#hyperparameters
input_size = 28 * 28
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 10

#dataset
entire_dataset = datasets.MNIST(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)
train_dataset, val_dataset = random_split(entire_dataset, [48000, 12000])
test_dataset = datasets.MNIST(root='dataset/', train=False, transform=transforms.ToTensor(), download=True)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

#network
model = NN(input_size=input_size, num_classes=num_classes).to(device)

#trainer
trainer = pl.Trainer(min_epochs=num_epochs, max_epochs=num_epochs, precision=16)
trainer.fit(model, train_loader, val_loader)
trainer.validate(model, val_loader)
trainer.test(model, test_loader)