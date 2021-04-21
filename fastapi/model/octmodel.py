# ----- model imports -----
import torch
import pytorch_lightning as pl
from pl_bolts.models.self_supervised import SimCLR
from pytorch_lightning.callbacks import Callback
import torch.optim as optim
import torch.nn as nn
from argparse import Namespace

# model
class OCTModel(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        hparams = Namespace(**hparams) if isinstance(hparams, dict) else hparams
        self.save_hyperparameters(hparams)

        # load simCLR pre-trained on imagenet for feature extraction  
        self.base_model = SimCLR.load_from_checkpoint(self.hparams.embeddings_path, 
                                                 strict=False)     
        # Set some params
        self.tune = self.hparams.tune
        # self.accuracy = pl.metrics.Accuracy()
        num_target_classes = self.hparams.n_classes
        
        #freeze
        if self.hparams.freeze_base:
            self.base_model.eval()  

        # Use the pretrained simclr model representations to classify oct. 
        self.classifier = nn.Sequential(
            nn.Linear(self.base_model.encoder.fc.in_features, 
                      self.base_model.encoder.fc.in_features),
            nn.ReLU(),
            nn.Linear(self.base_model.encoder.fc.in_features, 
                      num_target_classes if num_target_classes > 2 else 1),
        )
        
        # Create loss function
        self.loss_fn = torch.nn.CrossEntropyLoss()
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate,) 
        lr_scheduler = {'scheduler': torch.optim.lr_scheduler.OneCycleLR(
                                        optimizer,
                                        max_lr=self.hparams.learning_rate,
                                        steps_per_epoch=int(self.hparams.steps_per_epoch),
                                        epochs=self.hparams.max_epochs,
                                        anneal_strategy="linear",
                                        final_div_factor = 30,
                                        cycle_momentum=False
                                    ),
                        'name': 'learning_rate',
                        'interval':'step',
                        'frequency': 1}
        return [optimizer], [lr_scheduler]
         
    def forward(self, input_data):
        representations = self.base_model(input_data)
        preds = self.classifier(representations)
        # self.reps = representations
        return preds
    
    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        predictions = self.forward(x)
        loss = self.loss_fn(predictions, y)
        self.log('train_loss', loss)
        return {'loss': loss}
    
    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        predictions = self.forward(x)
        loss = self.loss_fn(predictions, y)
        self.log('val_loss', loss)