"""
Training Module
--------------
Handles model training with:
- Mixed precision training
- Learning rate scheduling
- Early stopping
- Checkpointing
- Metrics tracking
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path
import logging
from tqdm import tqdm
import json
from typing import Dict, Optional
import numpy as np

logger = logging.getLogger(__name__)


class Trainer:
    """Trainer class for nodule segmentation model."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        config: dict,
        checkpoint_dir: Path,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.config = config
        self.checkpoint_dir = Path(checkpoint_dir)
        self.scheduler = scheduler
        
        # Mixed precision training
        self.use_amp = config.get('use_amp', True)
        self.scaler = GradScaler() if self.use_amp else None
        
        # Early stopping
        self.early_stopping_patience = config.get('early_stopping_patience', 15)
        self.early_stopping_counter = 0
        self.best_val_metric = -np.inf if config.get('monitor_mode', 'max') == 'max' else np.inf
        
        # History
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_iou': [],
            'val_iou': [],
            'train_dice': [],
            'val_dice': [],
            'lr': []
        }
        
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Trainer initialized")
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        running_loss = 0.0
        running_iou = 0.0
        running_dice = 0.0
        num_batches = len(self.train_loader)
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1} [Train]")
        
        for batch_idx, (images, masks) in enumerate(pbar):
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if self.use_amp:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, masks)
                
                # Backward pass
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                loss.backward()
                self.optimizer.step()
            
            # Compute metrics
            with torch.no_grad():
                probs = torch.sigmoid(outputs)
                batch_iou = self._compute_batch_iou(probs, masks)
                batch_dice = self._compute_batch_dice(probs, masks)
            
            running_loss += loss.item()
            running_iou += batch_iou
            running_dice += batch_dice
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'iou': f"{batch_iou:.4f}",
                'dice': f"{batch_dice:.4f}"
            })
        
        # Average metrics
        avg_loss = running_loss / num_batches
        avg_iou = running_iou / num_batches
        avg_dice = running_dice / num_batches
        
        return {
            'loss': avg_loss,
            'iou': avg_iou,
            'dice': avg_dice
        }
    
    @torch.no_grad()
    def validate(self, epoch: int) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        
        running_loss = 0.0
        running_iou = 0.0
        running_dice = 0.0
        num_batches = len(self.val_loader)
        
        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch+1} [Val]")
        
        for images, masks in pbar:
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            # Forward pass
            if self.use_amp:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, masks)
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
            
            # Compute metrics
            probs = torch.sigmoid(outputs)
            batch_iou = self._compute_batch_iou(probs, masks)
            batch_dice = self._compute_batch_dice(probs, masks)
            
            running_loss += loss.item()
            running_iou += batch_iou
            running_dice += batch_dice
            
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'iou': f"{batch_iou:.4f}"
            })
        
        # Average metrics
        avg_loss = running_loss / num_batches
        avg_iou = running_iou / num_batches
        avg_dice = running_dice / num_batches
        
        return {
            'loss': avg_loss,
            'iou': avg_iou,
            'dice': avg_dice
        }
    
    def _compute_batch_iou(self, pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
        """Compute IoU for a batch."""
        pred_binary = (pred > threshold).float()
        intersection = (pred_binary * target).sum()
        union = pred_binary.sum() + target.sum() - intersection
        iou = intersection / (union + 1e-8)
        return iou.item()
    
    def _compute_batch_dice(self, pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
        """Compute Dice for a batch."""
        pred_binary = (pred > threshold).float()
        intersection = (pred_binary * target).sum()
        dice = (2.0 * intersection) / (pred_binary.sum() + target.sum() + 1e-8)
        return dice.item()
    
    def save_checkpoint(self, epoch: int, val_metrics: dict, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_metrics': val_metrics,
            'history': self.history,
            'config': self.config
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        # Save last checkpoint
        last_path = self.checkpoint_dir / 'last.pth'
        torch.save(checkpoint, last_path)
        logger.info(f"Saved last checkpoint: {last_path}")
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / 'best.pth'
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best checkpoint: {best_path} (IoU: {val_metrics['iou']:.4f})")
    
    def check_early_stopping(self, val_metric: float) -> bool:
        """
        Check if training should stop early.
        
        Returns:
            True if should stop, False otherwise
        """
        if not self.config.get('early_stopping', True):
            return False
        
        monitor_mode = self.config.get('monitor_mode', 'max')
        min_delta = self.config.get('early_stopping_min_delta', 1e-4)
        
        # Check if validation metric improved
        if monitor_mode == 'max':
            improved = val_metric > (self.best_val_metric + min_delta)
        else:
            improved = val_metric < (self.best_val_metric - min_delta)
        
        if improved:
            self.best_val_metric = val_metric
            self.early_stopping_counter = 0
            return False
        else:
            self.early_stopping_counter += 1
            logger.info(f"Early stopping counter: {self.early_stopping_counter}/{self.early_stopping_patience}")
            
            if self.early_stopping_counter >= self.early_stopping_patience:
                logger.info("Early stopping triggered!")
                return True
            return False
    
    def train(self, num_epochs: int):
        """Full training loop."""
        logger.info(f"Starting training for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            # Train
            train_metrics = self.train_epoch(epoch)
            
            # Validate
            val_metrics = self.validate(epoch)
            
            # Update history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['train_iou'].append(train_metrics['iou'])
            self.history['val_iou'].append(val_metrics['iou'])
            self.history['train_dice'].append(train_metrics['dice'])
            self.history['val_dice'].append(val_metrics['dice'])
            self.history['lr'].append(self.optimizer.param_groups[0]['lr'])
            
            # Log epoch summary
            logger.info(f"Epoch {epoch+1}/{num_epochs} - "
                       f"Train Loss: {train_metrics['loss']:.4f}, Val Loss: {val_metrics['loss']:.4f}, "
                       f"Train IoU: {train_metrics['iou']:.4f}, Val IoU: {val_metrics['iou']:.4f}")
            
            # Learning rate scheduling
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['iou'])
                else:
                    self.scheduler.step()
            
            # Check if best model
            is_best = val_metrics['iou'] > self.best_val_metric
            
            # Save checkpoint
            self.save_checkpoint(epoch, val_metrics, is_best=is_best)
            
            # Check early stopping
            if self.check_early_stopping(val_metrics['iou']):
                logger.info("Training stopped early")
                break
        
        # Save final history
        history_path = self.checkpoint_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        logger.info(f"Saved training history: {history_path}")
        
        logger.info("Training completed!")
