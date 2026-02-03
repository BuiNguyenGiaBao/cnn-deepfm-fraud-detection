import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Optional, Dict, Tuple
from tqdm import tqdm
from cnn_for_extract_feature import TabularCNNNetwork
from deepfm_for_relationship import DeepFM
import argparse
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, 
    recall_score, f1_score, confusion_matrix, 
    classification_report, roc_curve, auc
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


class HybridCNNDeepFM(nn.Module):
    """Hybrid CNN-DeepFM model for fraud detection"""
    def __init__(self, tabular_dim: int, 
                # CNN        
                embed_dim=64,
                conv_channels=128,
                kernel_size=3,
                bilinear_rank=64,
                bilinear_out_dim=256,
                seq_length=10,
                cnn_dropout=0.3,
                # DeepFM params
                num_classes: int=2,
                deepfm_embed_dim: int=16,
                deepfm_hidden=None,
                deepfm_dropout: float=0.2,
                # Training mode
                freeze_cnn: bool=False):
        super().__init__()
        self.num_classes = num_classes
        self.freeze_cnn = freeze_cnn

        # CNN Feature Extractor
        self.cnn = TabularCNNNetwork(
            tabular_dim=tabular_dim,
            embed_dim=embed_dim,
            conv_channels=conv_channels,
            kernel_size=kernel_size,
            bilinear_rank=bilinear_rank,
            bilinear_out_dim=bilinear_out_dim,
            num_classes=num_classes,
            seq_length=seq_length,
            dropout=cnn_dropout
        )

        # DeepFM for relationship learning
        deepfm_hidden = deepfm_hidden or [256, 128, 64]
        self.deepfm = DeepFM(
            num_classes=num_classes,
            categorical_cardinalities=None,
            num_numerical=0,
            embed_dim=deepfm_embed_dim,
            deep_hidden=deepfm_hidden,
            dropout=deepfm_dropout,
            dense_in_dim=bilinear_out_dim,
            use_bias=True
        )

        if freeze_cnn:
            for p in self.cnn.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: CNN Feature Extract -> DeepFM Classification"""
        cnn_features = self.cnn.get_embedding(x, detach=self.freeze_cnn)
        logits = self.deepfm(dense_x=cnn_features)
        return logits

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract CNN features only"""
        return self.cnn.get_embedding(x, detach=True)

    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Get attention weights from CNN"""
        return self.cnn.get_attention_weights(x)


class IEEEFraudDataset(Dataset):
    """Dataset for IEEE-CIS Fraud Detection"""
    def __init__(self, csv_path: str, target_col: str='isFraud', normalize: bool=False):
        print(f"Loading data from {csv_path}...")
        df = pd.read_csv(csv_path)

        # Handle infinite and missing values
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(0.0)
        
        # Load features and target
        if target_col in df.columns:
            self.y = torch.FloatTensor(df[target_col].values)
            self.X = torch.FloatTensor(df.drop(columns=[target_col]).values)
            self.has_target = True
        else:
            self.X = torch.FloatTensor(df.values)
            self.y = None
            self.has_target = False

        # Store feature names
        self.feature_names = (
            df.drop(columns=[target_col]).columns.tolist()
            if self.has_target else df.columns.tolist()
        )

        # Normalize features if requested
        if normalize and self.X.shape[0] > 0:
            mean = self.X.mean(dim=0)
            std = self.X.std(dim=0)
            std[std == 0] = 1.0  # Avoid division by zero
            self.X = (self.X - mean) / std

        print(f"✓ Loaded {len(self.X)} samples with {self.X.shape[1]} features")
        if self.has_target:
            fraud_ratio = self.y.mean().item()
            n_fraud = int(self.y.sum().item())
            n_normal = len(self.y) - n_fraud
            print(f"✓ Class distribution: Normal={n_normal}, Fraud={n_fraud}")
            print(f"✓ Fraud ratio: {fraud_ratio*100:.2f}%")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.has_target:
            return self.X[idx], self.y[idx]
        else:
            return self.X[idx]


class FraudDetectionTrainer:
    """Trainer for Fraud Detection Classification"""
    def __init__(
        self,
        model: HybridCNNDeepFM,
        device: str='cuda' if torch.cuda.is_available() else 'cpu',
        learning_rate: float=1e-3,
        weight_decay: float=1e-5,
        pos_weight: Optional[float]=None,
    ):
        self.model = model.to(device)
        self.device = device
        
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        # Loss function with pos_weight for imbalanced data
        if pos_weight is not None:
            pos_weight_tensor = torch.tensor([pos_weight]).to(device)
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
            print(f"✓ Using BCEWithLogitsLoss with pos_weight={pos_weight:.2f}")
        else:
            self.criterion = nn.BCEWithLogitsLoss()
            print(f"✓ Using BCEWithLogitsLoss (balanced)")
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train one epoch"""
        self.model.train()

        total_loss = 0.0
        all_preds = []
        all_probs = []
        all_labels = []

        pbar = tqdm(train_loader, desc='Training')
        for batch_x, batch_y in pbar:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device).float()

            # Forward pass
            logits = self.model(batch_x).view(-1)
            probs = torch.sigmoid(logits)

            # Compute loss
            loss = self.criterion(logits, batch_y)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # Store predictions
            preds = (probs > 0.5).long()
            all_preds.extend(preds.cpu().detach().numpy())
            all_probs.extend(probs.cpu().detach().numpy())
            all_labels.extend(batch_y.cpu().detach().numpy())
            
            total_loss += loss.item()

            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        # Calculate metrics
        all_preds = np.array(all_preds)
        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)
        
        metrics = {
            'loss': total_loss / len(train_loader),
            'accuracy': accuracy_score(all_labels, all_preds),
            'precision': precision_score(all_labels, all_preds, zero_division=0),
            'recall': recall_score(all_labels, all_preds, zero_division=0),
            'f1': f1_score(all_labels, all_preds, zero_division=0),
        }
        
        # Calculate AUC
        try:
            metrics['auc'] = roc_auc_score(all_labels, all_probs)
        except:
            metrics['auc'] = 0.0
                
        return metrics

    @torch.no_grad()
    def evaluate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Evaluate on validation set"""
        self.model.eval()

        total_loss = 0.0
        all_preds = []
        all_probs = []
        all_labels = []

        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device).float()

            logits = self.model(batch_x).view(-1)
            probs = torch.sigmoid(logits)
            loss = self.criterion(logits, batch_y)

            total_loss += loss.item()
            
            # Store predictions
            preds = (probs > 0.5).long()
            all_preds.extend(preds.cpu().detach().numpy())
            all_probs.extend(probs.cpu().detach().numpy())
            all_labels.extend(batch_y.cpu().detach().numpy())

        all_preds = np.array(all_preds)
        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)
        
        metrics = {
            'loss': total_loss / len(val_loader),
            'accuracy': accuracy_score(all_labels, all_preds),
            'precision': precision_score(all_labels, all_preds, zero_division=0),
            'recall': recall_score(all_labels, all_preds, zero_division=0),
            'f1': f1_score(all_labels, all_preds, zero_division=0),
        }
        
        # Calculate AUC
        try:
            metrics['auc'] = roc_auc_score(all_labels, all_probs)
        except:
            metrics['auc'] = 0.0
        
        return metrics

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int=100,
        early_stopping_patience: int=10,
        save_path: Optional[str]=None,
    ):
        """Full training loop with early stopping"""

        best_val_auc = 0.0
        patience_counter = 0

        history = {
            'train_loss': [], 'train_accuracy': [], 'train_precision': [], 
            'train_recall': [], 'train_f1': [], 'train_auc': [],
            'val_loss': [], 'val_accuracy': [], 'val_precision': [], 
            'val_recall': [], 'val_f1': [], 'val_auc': []
        }

        for epoch in range(epochs):
            print(f'\n{"="*70}')
            print(f'Epoch {epoch+1}/{epochs}')
            print(f'{"="*70}')

            # Train and validate
            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader)

            # Learning rate scheduling
            self.scheduler.step(val_metrics['loss'])

            # Save history
            for metric in ['loss', 'accuracy', 'precision', 'recall', 'f1', 'auc']:
                history[f'train_{metric}'].append(train_metrics.get(metric, 0.0))
                history[f'val_{metric}'].append(val_metrics.get(metric, 0.0))

            # Logging
            print(f"\nTrain Metrics:")
            print(f"  Loss: {train_metrics['loss']:.4f} | Acc: {train_metrics['accuracy']:.4f} | "
                  f"Prec: {train_metrics['precision']:.4f} | Rec: {train_metrics['recall']:.4f}")
            print(f"  F1: {train_metrics['f1']:.4f} | AUC: {train_metrics['auc']:.4f}")

            print(f"\nValidation Metrics:")
            print(f"  Loss: {val_metrics['loss']:.4f} | Acc: {val_metrics['accuracy']:.4f} | "
                  f"Prec: {val_metrics['precision']:.4f} | Rec: {val_metrics['recall']:.4f}")
            print(f"  F1: {val_metrics['f1']:.4f} | AUC: {val_metrics['auc']:.4f}")

            # Early stopping based on AUC
            if val_metrics['auc'] > best_val_auc:
                best_val_auc = val_metrics['auc']
                patience_counter = 0

                if save_path:
                    torch.save(
                        {
                            'epoch': epoch,
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'val_loss': val_metrics['loss'],
                            'val_accuracy': val_metrics['accuracy'],
                            'val_precision': val_metrics['precision'],
                            'val_recall': val_metrics['recall'],
                            'val_f1': val_metrics['f1'],
                            'val_auc': val_metrics['auc'],
                        },
                        save_path
                    )
                    print(f"\n✓ Best model saved! (AUC: {best_val_auc:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"\n⛔ Early stopping triggered after {epoch+1} epochs")
                    break

        return history
    
    @torch.no_grad()
    def predict(self, test_loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
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
            logits = self.model(batch_x).view(-1)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).long()
            
            all_preds.append(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
        
        predictions = np.concatenate(all_preds)
        probabilities = np.concatenate(all_probs)
        
        return predictions, probabilities
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        print(f"\n✓ Checkpoint loaded from epoch {checkpoint['epoch']}")
        print(f"  Val Loss: {checkpoint['val_loss']:.4f}")
        print(f"  Val AUC: {checkpoint['val_auc']:.4f}")
        print(f"  Val F1: {checkpoint['val_f1']:.4f}")

    def save_best_metrics(self, val_loader: DataLoader, save_dir: str='./results'):
        """Save detailed evaluation metrics and plots"""
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        self.model.eval()
        all_preds = []
        all_probs = []
        all_labels = []

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(self.device)
                logits = self.model(batch_x).view(-1)
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).long()
                
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())

        all_preds = np.array(all_preds)
        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)

        # Generate classification report
        report = classification_report(all_labels, all_preds, 
                                      target_names=['Normal', 'Fraud'])
        print("\n" + "="*60)
        print("CLASSIFICATION REPORT")
        print("="*60)
        print(report)

        # Save report
        with open(f'{save_dir}/classification_report.txt', 'w') as f:
            f.write(report)

        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Confusion Matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                   xticklabels=['Normal', 'Fraud'],
                   yticklabels=['Normal', 'Fraud'])
        axes[0].set_title('Confusion Matrix')
        axes[0].set_ylabel('True Label')
        axes[0].set_xlabel('Predicted Label')

        # Plot 2: ROC Curve
        fpr, tpr, _ = roc_curve(all_labels, all_probs)
        roc_auc = auc(fpr, tpr)
        axes[1].plot(fpr, tpr, color='darkorange', lw=2, 
                    label=f'ROC curve (AUC = {roc_auc:.4f})')
        axes[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        axes[1].set_xlim([0.0, 1.0])
        axes[1].set_ylim([0.0, 1.05])
        axes[1].set_xlabel('False Positive Rate')
        axes[1].set_ylabel('True Positive Rate')
        axes[1].set_title('ROC Curve')
        axes[1].legend(loc="lower right")

        plt.tight_layout()
        plt.savefig(f'{save_dir}/evaluation_metrics.png', dpi=300, bbox_inches='tight')
        print(f"\n✓ Metrics saved to {save_dir}/")


def plot_training_history(history: Dict, save_dir: str='./results'):
    """Plot training history"""
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    
    metrics = ['loss', 'accuracy', 'precision', 'recall', 'f1', 'auc']
    
    for idx, metric in enumerate(metrics):
        row, col = idx // 3, idx % 3
        
        train_key = f'train_{metric}'
        val_key = f'val_{metric}'
        
        if train_key in history and val_key in history:
            axes[row, col].plot(history[train_key], label=f'Train {metric}', marker='o')
            axes[row, col].plot(history[val_key], label=f'Val {metric}', marker='s')
            axes[row, col].set_title(f'{metric.upper()}')
            axes[row, col].set_xlabel('Epoch')
            axes[row, col].set_ylabel(metric.upper())
            axes[row, col].legend()
            axes[row, col].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/training_history.png', dpi=300, bbox_inches='tight')
    print(f"✓ Training history saved to {save_dir}/training_history.png")


if __name__ == "__main__":
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Argument parser
    parser = argparse.ArgumentParser(
        description='Fraud Detection with Hybrid CNN-DeepFM'
    )
    parser.add_argument('--train_csv', type=str, default='train_processed.csv',
                       help='Path to training CSV')
    parser.add_argument('--test_csv', type=str, default='test_processed.csv',
                       help='Path to test CSV')
    parser.add_argument('--output', type=str, default='submission.csv',
                       help='Output submission file')
    parser.add_argument('--batch_size', type=int, default=256,
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--val_split', type=float, default=0.2,
                       help='Validation split ratio')
    parser.add_argument('--use_pos_weight', action='store_true',
                       help='Use pos_weight for imbalanced data')
    parser.add_argument('--mode', type=str, default='train',
                       choices=['train', 'predict', 'train_and_predict'],
                       help='Mode: train, predict, or train_and_predict')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Checkpoint path for prediction')
    parser.add_argument('--model_save_path', type=str, default='best_fraud_model.pth',
                       help='Path to save best model')
    parser.add_argument('--results_dir', type=str, default='./results',
                       help='Directory to save results')
    
    args = parser.parse_args()
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("\n" + "="*70)
    print("🔍 IEEE-CIS FRAUD DETECTION: Hybrid CNN-DeepFM")
    print("="*70)
    print(f"Device: {DEVICE}")
    print(f"Mode: {args.mode}")
    print(f"Train CSV: {args.train_csv}")
    print(f"Test CSV: {args.test_csv}")
    print("="*70 + "\n")
    
    if args.mode in ['train', 'train_and_predict']:
        # ========== LOAD TRAINING DATA ==========
        print("📂 Loading training data...")
        full_dataset = IEEEFraudDataset(args.train_csv, target_col='isFraud', normalize=True)
        
        n_features = full_dataset.X.shape[1]
        print(f"📊 Number of features: {n_features}\n")
        
        # Split into train/val
        train_size = int((1 - args.val_split) * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        print(f"📈 Train samples: {len(train_dataset)}")
        print(f"📉 Val samples: {len(val_dataset)}\n")
        
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
            print(f"⚖️  Pos Weight: {pos_weight:.2f} (to balance {n_neg} vs {n_pos})\n")
        
        # ========== BUILD MODEL ==========
        print("🏗️  Building Hybrid CNN-DeepFM model...")
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
        
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"✓ Total trainable parameters: {total_params:,}\n")
        
        # ========== TRAINING ==========
        print("🚀 Starting training...\n")
        trainer = FraudDetectionTrainer(
            model=model,
            device=DEVICE,
            learning_rate=args.lr,
            weight_decay=1e-5,
            pos_weight=pos_weight
        )
        
        history = trainer.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=args.epochs,
            early_stopping_patience=15,
            save_path=args.model_save_path,
        )
        
        # Plot training history
        plot_training_history(history, save_dir=args.results_dir)
        
        # Save detailed metrics
        trainer.save_best_metrics(val_loader, save_dir=args.results_dir)
        
        print("\n" + "="*70)
        print("✅ TRAINING COMPLETED!")
        print("="*70 + "\n")
        
        # ========== PREDICTION ==========
        if args.mode == 'train_and_predict':
            print("🔮 Starting prediction on test set...\n")
            
            # Load best model
            trainer.load_checkpoint(args.model_save_path)
            
            # Load test data
            print("📂 Loading test data...")
            test_dataset = IEEEFraudDataset(args.test_csv, target_col='isFraud', normalize=True)
            
            test_loader = DataLoader(
                test_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True if DEVICE == 'cuda' else False
            )
            
            # Make predictions
            print("🎯 Making predictions...")
            predictions, probabilities = trainer.predict(test_loader)
            
            # Create submission file
            print(f"\n💾 Creating submission file: {args.output}")
            test_df = pd.read_csv(args.test_csv)
            
            if 'TransactionID' in test_df.columns:
                submission_df = pd.DataFrame({
                    'TransactionID': test_df['TransactionID'],
                    'isFraud': probabilities
                })
            else:
                submission_df = pd.DataFrame({
                    'TransactionID': range(len(probabilities)),
                    'isFraud': probabilities
                })
            
            submission_df.to_csv(args.output, index=False)
            
            # Print statistics
            print(f"\n{'='*60}")
            print("📊 PREDICTION STATISTICS")
            print(f"{'='*60}")
            print(f"✓ Total predictions: {len(submission_df)}")
            print(f"✓ Mean fraud probability: {probabilities.mean():.4f}")
            print(f"✓ Predictions > 0.5: {(probabilities > 0.5).sum()} ({100*(probabilities > 0.5).sum()/len(probabilities):.2f}%)")
            print(f"✓ Predictions > 0.3: {(probabilities > 0.3).sum()} ({100*(probabilities > 0.3).sum()/len(probabilities):.2f}%)")
            print(f"✓ Predictions > 0.7: {(probabilities > 0.7).sum()} ({100*(probabilities > 0.7).sum()/len(probabilities):.2f}%)")
            print(f"✓ Saved to: {args.output}")
            print(f"{'='*60}\n")
    
    elif args.mode == 'predict':
        if args.checkpoint is None:
            raise ValueError("❌ --checkpoint is required for prediction mode")
        
        # Load test data
        print("📂 Loading test data...")
        test_dataset = IEEEFraudDataset(args.test_csv, target_col='isFraud', normalize=True)
        
        n_features = test_dataset.X.shape[1]
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True if DEVICE == 'cuda' else False
        )
        
        # Create model
        print("🏗️  Building model...")
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
        trainer = FraudDetectionTrainer(model=model, device=DEVICE)
        trainer.load_checkpoint(args.checkpoint)
        
        # Make predictions
        print("\n🎯 Making predictions...")
        predictions, probabilities = trainer.predict(test_loader)
        
        # Create submission file
        print(f"\n💾 Creating submission file: {args.output}")
        test_df = pd.read_csv(args.test_csv)
        
        if 'TransactionID' in test_df.columns:
            submission_df = pd.DataFrame({
                'TransactionID': test_df['TransactionID'],
                'isFraud': probabilities
            })
        else:
            submission_df = pd.DataFrame({
                'TransactionID': range(len(probabilities)),
                'isFraud': probabilities
            })
        
        submission_df.to_csv(args.output, index=False)
        
        # Print statistics
        print(f"\n{'='*60}")
        print("📊 PREDICTION STATISTICS")
        print(f"{'='*60}")
        print(f"✓ Total predictions: {len(submission_df)}")
        print(f"✓ Mean fraud probability: {probabilities.mean():.4f}")
        print(f"✓ Predictions > 0.5: {(probabilities > 0.5).sum()} ({100*(probabilities > 0.5).sum()/len(probabilities):.2f}%)")
        print(f"✓ Predictions > 0.3: {(probabilities > 0.3).sum()} ({100*(probabilities > 0.3).sum()/len(probabilities):.2f}%)")
        print(f"✓ Predictions > 0.7: {(probabilities > 0.7).sum()} ({100*(probabilities > 0.7).sum()/len(probabilities):.2f}%)")
        print(f"✓ Saved to: {args.output}")
        print(f"{'='*60}\n")