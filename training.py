import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Optional,  Dict
from tqdm import tqdm
from cnn_for_extract_feature import TabularCNNNetwork
from deepfm_for_relationship import DeepFM
from torch.utils.data import Dataset
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split


class HybridCNNDeepFM(nn.Module):
    def __init__(self,tabular_dim: int, 
                #cnn        
                embed_dim = 64,
                conv_channels = 128,
                kernel_size = 3,
                bilinear_rank = 64,
                bilinear_out_dim= 256,
                seq_length= 10,
                cnn_dropout = 0.3,
                # DeepFM params
                num_classes: int = 2,
                deepfm_embed_dim: int = 16,
                deepfm_hidden=None,
                deepfm_dropout: float = 0.2,
                # Training mode
                freeze_cnn: bool = False,):
        super().__init__()
        self.num_classes = num_classes
        self.freeze_cnn = freeze_cnn

        self.cnn = TabularCNNNetwork(tabular_dim=tabular_dim,embed_dim=embed_dim,conv_channels=conv_channels,
            kernel_size=kernel_size,bilinear_rank=bilinear_rank,bilinear_out_dim=bilinear_out_dim,num_classes=num_classes, seq_length=seq_length,dropout=cnn_dropout)

        deepfm_hidden = deepfm_hidden or [256, 128, 64]
        self.deepfm = DeepFM(num_classes=num_classes,categorical_cardinalities=None,num_numerical=0,embed_dim=deepfm_embed_dim,
            deep_hidden=deepfm_hidden,dropout=deepfm_dropout,dense_in_dim=bilinear_out_dim,use_bias=True,)

        if freeze_cnn:
            for p in self.cnn.parameters():
                p.requires_grad = False
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cnn_features = self.cnn.get_embedding(x, detach=self.freeze_cnn)
        logits = self.deepfm(dense_x=cnn_features)
        return logits

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.cnn.get_embedding(x, detach=True)

    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        return self.cnn.get_attention_weights(x)            


from torch.utils.data import Dataset

class IEEEFraudDataset(Dataset):
    """Dataset for IEEE-CIS Fraud Detection (preprocessed CSV files)"""
    def __init__(self, csv_path: str, target_col: str = 'isFraud'):
        print(f"Loading data from {csv_path}...")
        df = pd.read_csv(csv_path)

        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(0.0)
        if target_col in df.columns:
            self.y = torch.FloatTensor(df[target_col].values)
            self.X = torch.FloatTensor(df.drop(columns=[target_col]).values)
            self.has_target = True
        else:
            self.X = torch.FloatTensor(df.values)
            self.y = None
            self.has_target = False

        self.feature_names = (
            df.drop(columns=[target_col]).columns.tolist()
            if self.has_target else df.columns.tolist()
        )

        print(f"✓ Loaded {len(self.X)} samples with {self.X.shape[1]} features")
        if self.has_target:
            fraud_ratio = self.y.mean().item()
            print(f"✓ Fraud ratio: {fraud_ratio*100:.2f}%")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.has_target:
            return self.X[idx], self.y[idx]
        else:
            return self.X[idx]

class Trainer:
    """Training and evaluation for Hybrid CNN-DeepFM model"""
    
    def __init__(
        self,
        model: HybridCNNDeepFM,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        pos_weight: Optional[float] = None,  # For imbalanced data
    ):
        self.model = model.to(device)
        self.device = device
        
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,mode='min',factor=0.5,patience=5,verbose=True)
        
        # Loss function based on num_classes
        if model.num_classes == 2:
            if pos_weight is not None:
                # Handle class imbalance
                pos_weight_tensor = torch.tensor([pos_weight]).to(device)
                self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
                print(f"Using pos_weight={pos_weight:.2f} for imbalanced classes")
            else:
                self.criterion = nn.BCEWithLogitsLoss()
        else:
            self.criterion = nn.CrossEntropyLoss()
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc='Training')
        for batch_x, batch_y in pbar:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            # Forward pass
            logits = self.model(batch_x)
            
            # Compute loss
            if self.model.num_classes == 2:
                logits = torch.clamp(logits, min=-20, max=20)
                loss = self.criterion(logits.view(-1), batch_y.float())

                preds = (torch.sigmoid(logits) > 0.5).long().view(-1)
            else:
                loss = self.criterion(logits, batch_y)
                preds = torch.argmax(logits, dim=1)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Metrics
            total_loss += loss.item()
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.0 * correct / total:.2f}%'
            })
        
        return {
            'loss': total_loss / len(train_loader),
            'accuracy': 100.0 * correct / total
        }
    
    @torch.no_grad()
    def evaluate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Evaluate on validation set"""
        self.model.eval()
        
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            # Forward pass
            logits = self.model(batch_x)
            
            # Compute loss
            if self.model.num_classes == 2:
                logits = torch.clamp(logits, min=-20, max=20)
                loss = self.criterion(logits.view(-1), batch_y.float())

                preds = (torch.sigmoid(logits) > 0.5).long().view(-1)
            else:
                loss = self.criterion(logits, batch_y)
                preds = torch.argmax(logits, dim=1)
            
            # Metrics
            total_loss += loss.item()
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)
        
        return {'loss': total_loss / len(val_loader),'accuracy': 100.0 * correct / total}
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 100,
        early_stopping_patience: int = 10,
        save_path: Optional[str] = None,
    ):
        """Full training loop with early stopping"""
        best_val_loss = float('inf')
        patience_counter = 0
        
        history = {'train_loss': [],'train_acc': [],'val_loss': [],'val_acc': []}
        
        for epoch in range(epochs):
            print(f'\nEpoch {epoch+1}/{epochs}')
            print('-' * 50)
            
            # Train
            train_metrics = self.train_epoch(train_loader)
            
            # Validate
            val_metrics = self.evaluate(val_loader)
            
            # Learning rate scheduling
            self.scheduler.step(val_metrics['loss'])
            
            # Store history
            history['train_loss'].append(train_metrics['loss'])
            history['train_acc'].append(train_metrics['accuracy'])
            history['val_loss'].append(val_metrics['loss'])
            history['val_acc'].append(val_metrics['accuracy'])
            
            print(f"Train Loss: {train_metrics['loss']:.4f} | Train Acc: {train_metrics['accuracy']:.2f}%")
            print(f"Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy']:.2f}%")
            
            # Early stopping
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                patience_counter = 0
                
                # Save best model
                if save_path:
                    torch.save({'epoch': epoch,'model_state_dict': self.model.state_dict(),'optimizer_state_dict': self.optimizer.state_dict(),
                        'val_loss': best_val_loss,'val_acc': val_metrics['accuracy'],}, save_path)
                    print(f"✓ Model saved to {save_path}")
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"\nEarly stopping triggered after {epoch+1} epochs")
                    break
        
        return history
    
    @torch.no_grad()
    def predict(self, test_loader: DataLoader, return_proba: bool = False) -> np.ndarray:
        """Make predictions on test set"""
        self.model.eval()
        
        all_preds = []
        all_probs = []
        
        for batch in tqdm(test_loader, desc='Predicting'):
            if isinstance(batch, (list, tuple)):
                batch_x = batch[0]
            else:
                batch_x = batch

            batch_x = batch_x.to(self.device)
            
            # Forward pass
            logits = self.model(batch_x)
            
            if self.model.num_classes == 2:
                probs = torch.sigmoid(logits).view(-1)
                preds = (probs > 0.5).long()
                all_probs.append(probs.cpu().numpy())
            else:
                probs = F.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)
                all_probs.append(probs.cpu().numpy())
            
            all_preds.append(preds.cpu().numpy())
        
        predictions = np.concatenate(all_preds)
        probabilities = np.concatenate(all_probs)
        
        if return_proba:
            return probabilities
        return predictions
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        print(f"Val Loss: {checkpoint['val_loss']:.4f} | Val Acc: {checkpoint['val_acc']:.2f}%")


def create_synthetic_data(n_samples: int = 10000, n_features: int = 50, n_classes: int = 2):
    """Create synthetic dataset for testing"""
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    
    # Create target based on some pattern
    if n_classes == 2:
        y = ((X[:, :5].sum(axis=1) + np.random.randn(n_samples) * 0.5) > 0).astype(np.int64)
    else:
        y = np.random.randint(0, n_classes, n_samples)
    
    return X, y


if __name__ == "__main__":
    TRAIN_CSV = "data/merge/train_processed.csv"
    TEST_CSV  = "data/merge/test_processed.csv"
    # Set random seed
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Argument parser
    parser = argparse.ArgumentParser(description='Train Hybrid CNN-DeepFM on IEEE-CIS Fraud Detection')
    parser.add_argument('--train_csv', type=str, default='D:/project/data/merge/train_processed.csv', help='Path to training CSV')
    parser.add_argument('--test_csv', type=str, default='D:/project/data/merge/test_processed.csv',  help='Path to test CSV')
    parser.add_argument('--output', type=str, default='submission.csv', help='Output submission file')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--val_split', type=float, default=0.2, help='Validation split ratio')
    parser.add_argument('--use_pos_weight', action='store_true', help='Use pos_weight for imbalanced data')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'predict', 'train_and_predict'], help='Mode: train, predict, or train_and_predict')
    parser.add_argument('--checkpoint', type=str, default=None, help='Checkpoint path for prediction')
    parser.add_argument('--model_save_path', type=str, default='best_fraud_model.pth', help='Path to save best model')
    
    args = parser.parse_args()
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("=" * 60)
    print("IEEE-CIS Fraud Detection: Hybrid CNN-DeepFM")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Mode: {args.mode}")
    print(f"Train CSV: {args.train_csv}")
    print(f"Test CSV: {args.test_csv}")
    print()
    
    if args.mode in ['train', 'train_and_predict']:
        # Load training data
        print("Loading training data...")
        full_dataset = IEEEFraudDataset(args.train_csv, target_col='isFraud')
        
        # Get number of features
        n_features = full_dataset.X.shape[1]
        print(f"Number of features: {n_features}")
        
        # Split into train/val
        train_size = int((1 - args.val_split) * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        print(f"Train samples: {len(train_dataset)}")
        print(f"Val samples: {len(val_dataset)}")
        print()
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True if DEVICE == 'cuda' else False
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True if DEVICE == 'cuda' else False
        )
        
        # Calculate pos_weight for imbalanced data
        pos_weight = None
        if args.use_pos_weight and full_dataset.has_target:
            n_pos = full_dataset.y.sum().item()
            n_neg = len(full_dataset.y) - n_pos
            pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
            print(f"Calculated pos_weight: {pos_weight:.2f} (neg/pos = {n_neg}/{n_pos})")
        
        # Create model
        print("\nBuilding Hybrid CNN-DeepFM model...")
        model = HybridCNNDeepFM(tabular_dim=n_features,embed_dim=64,conv_channels=128,kernel_size=3,
            bilinear_rank=64,
            bilinear_out_dim=256,
            seq_length=10,
            cnn_dropout=0.3,
            num_classes=2,  
            deepfm_embed_dim=16,
            deepfm_hidden=[256, 128, 64],
            deepfm_dropout=0.2,freeze_cnn=False,
        )
        
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total trainable parameters: {total_params:,}")
        print()
        
        # Create trainer
        trainer = Trainer(model=model,device=DEVICE,learning_rate=args.lr,weight_decay=1e-5,pos_weight=pos_weight)
        
        # Train model
        print("Starting training...")
        print("=" * 60)
        
        history = trainer.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=args.epochs,
            early_stopping_patience=15,
            save_path=args.model_save_path,
        )
        
        print("\n" + "=" * 60)
        print("Training completed!")
        print("=" * 60)
        
        # Final evaluation
        print("\nFinal evaluation on validation set:")
        final_metrics = trainer.evaluate(val_loader)
        print(f"Validation Loss: {final_metrics['loss']:.4f}")
        print(f"Validation Accuracy: {final_metrics['accuracy']:.2f}%")
        
        # If train_and_predict mode, continue to prediction
        if args.mode == 'train_and_predict':
            print("\n" + "=" * 60)
            print("Starting prediction on test set...")
            print("=" * 60)
            
            # Load best model
            trainer.load_checkpoint(args.model_save_path)
            
            # Load test data
            print("\nLoading test data...")
            test_dataset = IEEEFraudDataset(args.test_csv, target_col='isFraud')
            
            test_loader = DataLoader(
                test_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True if DEVICE == 'cuda' else False
            )
            
            # Make predictions
            print("\nMaking predictions on test set...")
            predictions_proba = trainer.predict(test_loader, return_proba=True)
            
            # Create submission file
            print(f"\nCreating submission file: {args.output}")
            test_df = pd.read_csv(args.test_csv)
            
            if 'TransactionID' in test_df.columns:
                submission_df = pd.DataFrame({
                    'TransactionID': test_df['TransactionID'],
                    'isFraud': predictions_proba
                })
            else:
                submission_df = pd.DataFrame({
                    'TransactionID': range(len(predictions_proba)),
                    'isFraud': predictions_proba
                })
            
            submission_df.to_csv(args.output, index=False)
            print(f"✓ Submission saved to {args.output}")
            print(f"✓ Number of predictions: {len(submission_df)}")
            print(f"✓ Mean prediction: {predictions_proba.mean():.4f}")
            print(f"✓ Predictions > 0.5: {(predictions_proba > 0.5).sum()} ({100*(predictions_proba > 0.5).sum()/len(predictions_proba):.2f}%)")
    
    elif args.mode == 'predict':
        if args.checkpoint is None:
            raise ValueError("--checkpoint is required for prediction mode")
        
        # Load test data
        print("Loading test data...")
        test_dataset = IEEEFraudDataset(args.test_csv, target_col='isFraud')
        
        n_features = test_dataset.X.shape[1]
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True if DEVICE == 'cuda' else False
        )
        
        # Create model
        print("Building model...")
        model = HybridCNNDeepFM(
            tabular_dim=n_features,
            embed_dim=64,
            conv_channels=128,
            kernel_size=3,
            bilinear_rank=64,
            bilinear_out_dim=256,
            seq_length=10,
            cnn_dropout=0.3,
            num_classes=2,
            deepfm_embed_dim=16,
            deepfm_hidden=[256, 128, 64],
            deepfm_dropout=0.2,
            freeze_cnn=False,
        )
        
        # Create trainer and load checkpoint
        trainer = Trainer(model=model, device=DEVICE)
        trainer.load_checkpoint(args.checkpoint)
        
        # Make predictions
        print("\nMaking predictions...")
        predictions_proba = trainer.predict(test_loader, return_proba=True)
        
        # Create submission file
        print(f"\nCreating submission file: {args.output}")
        
        # Load original test CSV to get TransactionID if available
        test_df = pd.read_csv(args.test_csv)
        
        if 'TransactionID' in test_df.columns:
            submission_df = pd.DataFrame({
                'TransactionID': test_df['TransactionID'],
                'isFraud': predictions_proba
            })
        else:
            # If no TransactionID, create sequential IDs
            submission_df = pd.DataFrame({
                'TransactionID': range(len(predictions_proba)),
                'isFraud': predictions_proba
            })
        
        submission_df.to_csv(args.output, index=False)
        print(f"✓ Submission saved to {args.output}")
        print(f"✓ Number of predictions: {len(submission_df)}")
        print(f"✓ Mean prediction: {predictions_proba.mean():.4f}")
        print(f"✓ Predictions > 0.5: {(predictions_proba > 0.5).sum()} ({100*(predictions_proba > 0.5).sum()/len(predictions_proba):.2f}%)")