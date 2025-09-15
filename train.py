"""
Training script for MMIN-DGODE with LLM-guided Emotion Recognition
ICASSP 2026 Submission
"""

import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

from model import MMIN_DGODE
from dataset import IEMOCAPDataset, MELDDataset
from utils import setup_seed, get_logger, save_checkpoint, load_checkpoint
from config import get_config

def train_epoch(model, train_loader, optimizer, criterion, device, config):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    total_cls_loss = 0
    total_recon_loss = 0
    total_align_loss = 0
    total_consist_loss = 0
    
    all_preds = []
    all_labels = []
    
    progress_bar = tqdm(train_loader, desc='Training')
    for batch_idx, batch in enumerate(progress_bar):
        # Move batch to device
        audio = batch['audio'].to(device) if batch['audio'] is not None else None
        visual = batch['visual'].to(device) if batch['visual'] is not None else None
        text = batch['text'].to(device) if batch['text'] is not None else None
        labels = batch['labels'].to(device)
        speaker_ids = batch['speaker_ids'].to(device)
        utterance_ids = batch['utterance_ids'].to(device)
        modality_masks = batch['modality_masks'].to(device)  # [B, 3] for a,v,t availability
        
        # Forward pass
        outputs = model(
            audio=audio,
            visual=visual, 
            text=text,
            speaker_ids=speaker_ids,
            utterance_ids=utterance_ids,
            modality_masks=modality_masks,
            labels=labels,  # For consistency loss
            return_losses=True
        )
        
        # Calculate total loss
        loss = outputs['cls_loss']
        total_cls_loss += outputs['cls_loss'].item()
        
        if 'recon_loss' in outputs and outputs['recon_loss'] is not None:
            loss += config.gamma1 * outputs['recon_loss']
            total_recon_loss += outputs['recon_loss'].item()
            
        if 'align_loss' in outputs and outputs['align_loss'] is not None:
            loss += config.gamma2 * outputs['align_loss']
            total_align_loss += outputs['align_loss'].item()
            
        if 'consist_loss' in outputs and outputs['consist_loss'] is not None:
            loss += config.gamma3 * outputs['consist_loss']
            total_consist_loss += outputs['consist_loss'].item()
        
        total_loss += loss.item()
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        optimizer.step()
        
        # Store predictions
        preds = outputs['logits'].argmax(dim=-1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'cls': f'{outputs["cls_loss"].item():.4f}'
        })
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    avg_losses = {
        'total': total_loss / len(train_loader),
        'cls': total_cls_loss / len(train_loader),
        'recon': total_recon_loss / len(train_loader) if total_recon_loss > 0 else 0,
        'align': total_align_loss / len(train_loader) if total_align_loss > 0 else 0,
        'consist': total_consist_loss / len(train_loader) if total_consist_loss > 0 else 0
    }
    
    return avg_losses, accuracy, f1

def validate(model, val_loader, device, config):
    """Validate the model"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Validation'):
            # Move batch to device
            audio = batch['audio'].to(device) if batch['audio'] is not None else None
            visual = batch['visual'].to(device) if batch['visual'] is not None else None
            text = batch['text'].to(device) if batch['text'] is not None else None
            labels = batch['labels'].to(device)
            speaker_ids = batch['speaker_ids'].to(device)
            utterance_ids = batch['utterance_ids'].to(device)
            modality_masks = batch['modality_masks'].to(device)
            
            # Forward pass
            outputs = model(
                audio=audio,
                visual=visual,
                text=text,
                speaker_ids=speaker_ids,
                utterance_ids=utterance_ids,
                modality_masks=modality_masks,
                labels=labels,
                return_losses=True
            )
            
            # Calculate loss
            loss = outputs['cls_loss']
            if 'recon_loss' in outputs and outputs['recon_loss'] is not None:
                loss += config.gamma1 * outputs['recon_loss']
            if 'align_loss' in outputs and outputs['align_loss'] is not None:
                loss += config.gamma2 * outputs['align_loss']
            if 'consist_loss' in outputs and outputs['consist_loss'] is not None:
                loss += config.gamma3 * outputs['consist_loss']
                
            total_loss += loss.item()
            
            # Store predictions
            preds = outputs['logits'].argmax(dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    conf_matrix = confusion_matrix(all_labels, all_preds)
    
    return total_loss / len(val_loader), accuracy, f1, conf_matrix

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='iemocap', choices=['iemocap', 'meld'])
    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    parser.add_argument('--log_dir', type=str, default='./logs')
    parser.add_argument('--wandb_project', type=str, default='mmin-dgode')
    parser.add_argument('--missing_rate', type=float, default=0.3, help='Missing modality rate for training')
    parser.add_argument('--missing_scenario', type=str, default='random', 
                       choices=['random', 'audio_missing', 'visual_missing', 'text_missing', 
                               'av_missing', 'at_missing', 'vt_missing'])
    
    args = parser.parse_args()
    
    # Get config
    config = get_config(args.dataset)
    config.update(vars(args))
    
    # Setup
    setup_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger = get_logger(args.log_dir)
    
    # Initialize wandb
    wandb.init(project=args.wandb_project, config=config)
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Load dataset
    logger.info(f"Loading {args.dataset} dataset...")
    if args.dataset == 'iemocap':
        train_dataset = IEMOCAPDataset(
            data_path=args.data_path,
            split='train',
            missing_rate=args.missing_rate,
            missing_scenario=args.missing_scenario
        )
        val_dataset = IEMOCAPDataset(
            data_path=args.data_path,
            split='val',
            missing_rate=args.missing_rate,
            missing_scenario=args.missing_scenario
        )
    else:  # MELD
        train_dataset = MELDDataset(
            data_path=args.data_path,
            split='train',
            missing_rate=args.missing_rate,
            missing_scenario=args.missing_scenario
        )
        val_dataset = MELDDataset(
            data_path=args.data_path,
            split='val',
            missing_rate=args.missing_rate,
            missing_scenario=args.missing_scenario
        )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Initialize model
    logger.info("Initializing model...")
    model = MMIN_DGODE(config).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Initialize optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-6
    )
    
    # Training loop
    best_val_f1 = 0
    best_epoch = 0
    
    for epoch in range(1, args.epochs + 1):
        logger.info(f"\n{'='*50}")
        logger.info(f"Epoch {epoch}/{args.epochs}")
        logger.info(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Train
        train_losses, train_acc, train_f1 = train_epoch(
            model, train_loader, optimizer, None, device, config
        )
        
        # Validate
        val_loss, val_acc, val_f1, conf_matrix = validate(
            model, val_loader, device, config
        )
        
        # Update scheduler
        scheduler.step()
        
        # Log metrics
        logger.info(f"Train - Loss: {train_losses['total']:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
        logger.info(f"Val - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")
        
        wandb.log({
            'epoch': epoch,
            'train/loss': train_losses['total'],
            'train/cls_loss': train_losses['cls'],
            'train/recon_loss': train_losses['recon'],
            'train/align_loss': train_losses['align'],
            'train/consist_loss': train_losses['consist'],
            'train/accuracy': train_acc,
            'train/f1': train_f1,
            'val/loss': val_loss,
            'val/accuracy': val_acc,
            'val/f1': val_f1,
            'lr': optimizer.param_groups[0]['lr']
        })
        
        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            
            save_checkpoint({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_f1': val_f1,
                'val_acc': val_acc,
                'config': config
            }, os.path.join(args.save_dir, f'best_model_{args.dataset}.pth'))
            
            logger.info(f"Saved best model with F1: {val_f1:.4f}")
            
            # Log confusion matrix
            wandb.log({
                'val/confusion_matrix': wandb.plot.confusion_matrix(
                    probs=None,
                    y_true=list(range(config.num_classes)),
                    preds=list(range(config.num_classes)),
                    class_names=config.emotion_labels
                )
            })
    
    logger.info(f"\nTraining completed!")
    logger.info(f"Best model: Epoch {best_epoch} with F1: {best_val_f1:.4f}")
    
    wandb.finish()

if __name__ == '__main__':
    main()
