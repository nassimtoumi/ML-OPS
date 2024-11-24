import os
import ast
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from pytorch_lightning.core.module import LightningModule
from pytorch_lightning import Trainer
from argparse import ArgumentParser
from model import SpeechRecognition
from dataset import Data, collate_fn_padd
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from minio import Minio

# Initialize MinIO client with command-line parameters
def initialize_minio_client(endpoint, access_key, secret_key):
    return Minio(
        endpoint,
        access_key=access_key,
        secret_key=secret_key,
        secure=False
    )

def download_json_files_from_minio(minio_client, bucket):
    # Download train and test JSON files from MinIO to /tmp
    minio_client.fget_object(bucket, "train.json", "/tmp/train.json")
    minio_client.fget_object(bucket, "test.json", "/tmp/test.json")

def upload_checkpoint_to_minio(minio_client, bucket, filepath):
    # Upload checkpoint to MinIO bucket after saving
    checkpoint_name = os.path.basename(filepath)
    minio_client.fput_object(bucket, checkpoint_name, filepath)
    print(f"Checkpoint {checkpoint_name} uploaded to MinIO bucket {bucket}")

class SpeechModule(LightningModule):
    def __init__(self, model, args):
        super(SpeechModule, self).__init__()
        self.model = model
        self.criterion = nn.CTCLoss(blank=28, zero_infinity=True)
        self.args = args

    def forward(self, x, hidden):
        return self.model(x, hidden)

    def configure_optimizers(self):
        self.optimizer = optim.AdamW(self.model.parameters(), self.args.learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.50, patience=6)
        return [self.optimizer], [self.scheduler]

    def step(self, batch):
        spectrograms, labels, input_lengths, label_lengths = batch 
        bs = spectrograms.shape[0]
        hidden = self.model._init_hidden(bs)
        hn, c0 = hidden[0].to(self.device), hidden[1].to(self.device)
        output, _ = self(spectrograms, (hn, c0))
        output = F.log_softmax(output, dim=2)
        loss = self.criterion(output, labels, input_lengths, label_lengths)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.step(batch)
        logs = {'loss': loss, 'lr': self.optimizer.param_groups[0]['lr'] }
        return {'loss': loss, 'log': logs}

    def train_dataloader(self):
        d_params = Data.parameters
        d_params.update(self.args.dparams_override)
        train_dataset = Data(json_path=self.args.train_file, 
                             sample_rate=d_params['sample_rate'], 
                             n_feats=d_params['n_feats'], 
                             specaug_rate=d_params['specaug_rate'], 
                             specaug_policy=d_params['specaug_policy'], 
                             time_mask=d_params['time_mask'], 
                             freq_mask=d_params['freq_mask'],
                             minio_client=self.minio_client,
                             minio_bucket=self.args.minio_bucket)
        return DataLoader(dataset=train_dataset,
                          batch_size=self.args.batch_size,
                          num_workers=self.args.data_workers,
                          pin_memory=True,
                          collate_fn=collate_fn_padd)

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch)
        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.scheduler.step(avg_loss)
        tensorboard_logs = {'val_loss': avg_loss}
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def val_dataloader(self):
        d_params = Data.parameters
        d_params.update(self.args.dparams_override)
        test_dataset = Data(json_path=self.args.valid_file,
                            sample_rate=d_params['sample_rate'], 
                            n_feats=d_params['n_feats'], 
                            specaug_rate=d_params['specaug_rate'], 
                            specaug_policy=d_params['specaug_policy'], 
                            time_mask=d_params['time_mask'], 
                            freq_mask=d_params['freq_mask'], 
                            minio_client=self.minio_client,
                            minio_bucket=self.args.minio_bucket,
                            valid=True)
        return DataLoader(dataset=test_dataset,
                          batch_size=self.args.batch_size,
                          num_workers=self.args.data_workers,
                          collate_fn=collate_fn_padd,
                          pin_memory=True)

def checkpoint_callback(args):
    # Save checkpoints locally and upload to MinIO after each save
    return ModelCheckpoint(
    dirpath=args.checkpoint_dir,
    filename='{epoch:02d}-{val_loss:.2f}',
    monitor='val_loss',
    save_top_k=3,
    mode='min'
    )

def main(args):
    # Initialize MinIO client
    minio_client = initialize_minio_client(args.minio_endpoint, args.minio_access_key, args.minio_secret_key)

    # Download the necessary JSON files from MinIO for training
    download_json_files_from_minio(minio_client, args.minio_bucket)

    h_params = SpeechRecognition.hyper_parameters
    h_params.update(args.hparams_override)
    model = SpeechRecognition(**h_params)

    speech_module = SpeechModule(model, args)
    
    # Assign MinIO client to module for use in DataLoader
    speech_module.minio_client = minio_client

    logger = TensorBoardLogger(args.logdir, name='speech_recognition')
    trainer = Trainer(
        max_epochs=args.epochs,
        gpus=args.gpus,
        num_nodes=args.nodes,
        logger=logger,
        gradient_clip_val=1.0,
        val_check_interval=args.valid_every,
        checkpoint_callback=checkpoint_callback(args),
        resume_from_checkpoint=args.resume_from_checkpoint
    )

    # Train the model
    trainer.fit(speech_module)

    # After training, upload all checkpoints in the checkpoints directory to MinIO
    checkpoint_dir = '/tmp/checkpoints'
    if os.path.isdir(checkpoint_dir):
        for checkpoint_file in os.listdir(checkpoint_dir):
            upload_checkpoint_to_minio(minio_client, args.minio_bucket, os.path.join(checkpoint_dir, checkpoint_file))

if __name__ == "__main__":
    parser = ArgumentParser()
    
    # MinIO parameters
    parser.add_argument('--minio_endpoint', required=True, type=str, help='MinIO server endpoint')
    parser.add_argument('--minio_access_key', required=True, type=str, help='MinIO access key')
    parser.add_argument('--minio_secret_key', required=True, type=str, help='MinIO secret key')
    parser.add_argument('--minio_bucket', required=True, type=str, help='MinIO bucket name')

        # distributed training setup
    parser.add_argument('-n', '--nodes', default=1, type=int, help='number of data loading workers')
    parser.add_argument('-g', '--gpus', default=1, type=int, help='number of gpus per node')
    parser.add_argument('-w', '--data_workers', default=0, type=int,
                        help='n data loading workers, default 0 = main process only')
    parser.add_argument('-db', '--dist_backend', default='ddp', type=str,
                        help='which distributed backend to use. defaul ddp')

    # train and valid
    parser.add_argument('--train_file', default=None, required=True, type=str,
                        help='json file to load training data')
    parser.add_argument('--valid_file', default=None, required=True, type=str,
                        help='json file to load testing data')
    parser.add_argument('--valid_every', default=1000, required=False, type=int,
                        help='valid after every N iteration')

    # dir and path for models and logs
    parser.add_argument('--save_model_path', default=None, required=True, type=str,
                        help='path to save model')
    parser.add_argument('--load_model_from', default=None, required=False, type=str,
                        help='path to load a pretrain model to continue training')
    parser.add_argument('--resume_from_checkpoint', default=None, required=False, type=str,
                        help='check path to resume from')
    parser.add_argument('--logdir', default='tb_logs', required=False, type=str,
                        help='path to save logs')
    
    # Checkpoint directory
    parser.add_argument('--checkpoint_dir', required=False, type=str, help='Directory to save checkpoints')
    # general
    parser.add_argument('--epochs', default=10, type=int, help='number of total epochs to run')
    parser.add_argument('--batch_size', default=64, type=int, help='size of batch')
    parser.add_argument('--learning_rate', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--pct_start', default=0.3, type=float, help='percentage of growth phase in one cycle')
    parser.add_argument('--div_factor', default=100, type=int, help='div factor for one cycle')
    parser.add_argument("--hparams_override", default="{}", type=str, required=False,
		help='override the hyper parameters, should be in form of dict. ie. {"attention_layers": 16 }')
    parser.add_argument("--dparams_override", default="{}", type=str, required=False,
		help='override the data parameters, should be in form of dict. ie. {"sample_rate": 8000 }')

    args = parser.parse_args()
    args.hparams_override = ast.literal_eval(args.hparams_override)
    args.dparams_override = ast.literal_eval(args.dparams_override)

    main(args)