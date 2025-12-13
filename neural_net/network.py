import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch
from typing import List, Any
import pandas as pd
from tqdm import tqdm

class Network_Trainer():

    def __init__(self,
                 learning_rate: float = 1e-3,
                 weight_decay: float = 1e-4, 
                 model: Any = None):
        
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.model.to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), 
                                     lr=self.learning_rate, 
                                     weight_decay=self.weight_decay)
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 
                                                               mode='min',
                                                               patience=5, 
                                                               factor=0.5)

    def forward(self, 
                training_policy: torch.Tensor,
                similar_policy: torch.Tensor,
                dissimilar_policies: torch.Tensor):
        
        batch_size, num_dissimilar, input_dim = dissimilar_policies.shape

        training_input = torch.cat([
            training_policy,
            similar_policy,
            dissimilar_policies.view(batch_size * num_dissimilar, input_dim)
        ], dim=0)

        all_embeddings = self.model.forward(training_input)

        training_emb = all_embeddings[:batch_size]
        similar_emb = all_embeddings[batch_size:2*batch_size]
        dissimilar_embs = all_embeddings[2*batch_size:].view(batch_size, num_dissimilar, -1)

        sim_positive = torch.sum(training_emb * similar_emb, dim=1) / self.model.temperature
        sim_negative = torch.bmm(dissimilar_embs, training_emb.unsqueeze(2)).squeeze(2) / self.model.temperature

        all_similarities = torch.cat([sim_positive.unsqueeze(1), sim_negative], dim=1)

        labels = torch.zeros(batch_size, dtype=torch.long, device=all_similarities.device)
        loss = F.cross_entropy(all_similarities, labels)
        return all_similarities, loss, labels
        
    
    def train_epoch(self, 
                    dataloader: DataLoader, 
                    training: bool = True):
        
        if training:
            self.model.train()
        else:
            self.model.eval()

        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        num_batches = 0

        context = torch.enable_grad() if training else torch.no_grad()
        with context:
            for batch in dataloader:
                query = batch['anchors'].to(self.device)
                sim_pol = batch['sim_pol'].to(self.device)
                candidates = batch['candidates'].to(self.device)
                # labels = batch['labels'].to(self.device)

                # forward pass
                similarities, loss, labels = self.forward(query, sim_pol, candidates)

                # accuracy
                predictions = torch.argmax(similarities, dim=1)

                correct_predictions += (predictions == labels).sum().item()
                total_predictions += labels.size(0)

                # backward pass (only if training)
                if training:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches
        avg_accuracy = correct_predictions / total_predictions
        metrics = {'loss': avg_loss, 'accuracy': avg_accuracy}

        return metrics
    
    def training(self, 
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 n_epochs: int = 50,
                 desc : str = "Training Progress",
                 verbose: bool = False):
       
        best_training_accuracy = 0.0
        best_validation_accuracy = 0.0

        for epoch in tqdm(range(n_epochs), desc=desc):
            train_metrics = self.train_epoch(train_loader, training=True)
            val_metrics = self.train_epoch(val_loader, training=False)
            self.scheduler.step(val_metrics['loss'])

            if train_metrics['accuracy'] > best_training_accuracy:
                best_training_accuracy = train_metrics['accuracy']

            if val_metrics['accuracy'] > best_validation_accuracy:
                best_validation_accuracy = val_metrics['accuracy']

            if verbose:
                if (epoch + 1) % 20 == 0:
                    tqdm.write(f"Epoch {epoch+1}/{n_epochs} - "
                            f"Train Loss: {train_metrics['loss']:.4f}, "
                            f"Train Acc: {train_metrics['accuracy']:.4f} - "
                            f"Val Loss: {val_metrics['loss']:.4f}, "
                            f"Val Acc: {val_metrics['accuracy']:.4f}")

        return best_training_accuracy, best_validation_accuracy