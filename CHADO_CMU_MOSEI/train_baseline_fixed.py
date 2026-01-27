import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import sys
from pathlib import Path
from tqdm import tqdm
import json
import numpy as np

sys.path.append('src')
from data_processing.dataset import CMUMOSEIDataset
from models.simple_baseline_fixed import SimpleMultimodalBaselineFixed

def compute_class_weights(dataset):
    """Compute class weights for imbalanced dataset"""
    labels = [sample['label'] for sample in dataset.data]
    unique, counts = np.unique(labels, return_counts=True)
    
    # Inverse frequency weighting
    total = len(labels)
    weights = total / (len(unique) * counts)
    
    print("\nClass distribution and weights:")
    for cls, count, weight in zip(unique, counts, weights):
        print(f"  Class {cls}: {count:5d} samples ({100*count/total:5.1f}%) -> weight: {weight:.3f}")
    
    return torch.FloatTensor(weights)

def train_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc=f'Epoch {epoch}')
    for batch_idx, batch in enumerate(pbar):
        audio = batch['audio'].to(device)
        visual = batch['visual'].to(device)
        text = batch['text'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        
        logits = model(audio, visual, text)
        loss = criterion(logits, labels)
        
        # Check for NaN
        if torch.isnan(loss):
            print(f"\nWarning: NaN loss at batch {batch_idx}, skipping...")
            continue
        
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = logits.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    return total_loss / len(loader), 100. * correct / total

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc='Validating'):
            audio = batch['audio'].to(device)
            visual = batch['visual'].to(device)
            text = batch['text'].to(device)
            labels = batch['label'].to(device)
            
            logits = model(audio, visual, text)
            loss = criterion(logits, labels)
            
            if not torch.isnan(loss):
                total_loss += loss.item()
            
            _, predicted = logits.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Per-class accuracy
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    print("\nPer-class accuracy:")
    emotion_names = ['happy', 'sad', 'anger', 'surprise', 'disgust', 'fear']
    for i in range(6):
        mask = all_labels == i
        if mask.sum() > 0:
            class_acc = (all_preds[mask] == all_labels[mask]).sum() / mask.sum() * 100
            print(f"  {emotion_names[i]:10s} (class {i}): {class_acc:5.2f}%")
    
    return total_loss / len(loader), 100. * correct / total

def main():
    # Config
    device = 'cuda:0'
    batch_size = 32
    epochs = 50
    lr = 5e-5
    weight_decay = 1e-4
    
    print("="*60)
    print("TRAINING FIXED BASELINE (TARGET: 75% ACCURACY)")
    print("="*60)
    
    # Load data
    print("\nLoading datasets...")
    train_dataset = CMUMOSEIDataset('data/processed/train_data.pkl')
    val_dataset = CMUMOSEIDataset('data/processed/val_data.pkl')
    test_dataset = CMUMOSEIDataset('data/processed/test_data.pkl')
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    print(f"✓ Train: {len(train_dataset)} samples")
    print(f"✓ Val: {len(val_dataset)} samples")
    print(f"✓ Test: {len(test_dataset)} samples")
    
    # Compute class weights
    class_weights = compute_class_weights(train_dataset).to(device)
    
    # Model
    print("\nInitializing model...")
    model = SimpleMultimodalBaselineFixed(
        audio_dim=74,
        visual_dim=35,
        text_dim=300,
        hidden_dim=128,
        num_classes=6
    ).to(device)
    
    print(f"✓ Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training setup with weighted loss
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    # Training loop
    best_val_acc = 0
    results = []
    patience_counter = 0
    max_patience = 10
    
    for epoch in range(epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"{'='*60}")
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, epoch+1)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_acc)
        new_lr = optimizer.param_groups[0]['lr']
        
        if new_lr != old_lr:
            print(f"\nLearning rate reduced: {old_lr:.2e} -> {new_lr:.2e}")
        
        print(f"\nTrain Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        results.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc
        })
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'checkpoints/baseline_fixed_best.pth')
            print(f"✓ Best model saved! Val Acc: {best_val_acc:.2f}%")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                print(f"\nEarly stopping after {epoch+1} epochs (no improvement for {max_patience} epochs)")
                break
    
    # Test evaluation
    print(f"\n{'='*60}")
    print("FINAL TEST EVALUATION")
    print(f"{'='*60}")
    
    model.load_state_dict(torch.load('checkpoints/baseline_fixed_best.pth'))
    test_loss, test_acc = validate(model, test_loader, criterion, device)
    
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Acc: {test_acc:.2f}%")
    print(f"Best Val Acc: {best_val_acc:.2f}%")
    
    # Save results
    with open('results/baseline_fixed_results.json', 'w') as f:
        json.dump({
            'results': results,
            'best_val_acc': best_val_acc,
            'test_acc': test_acc
        }, f, indent=2)
    
    print(f"\n{'='*60}")
    print("✓ TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Best Val Acc: {best_val_acc:.2f}%")
    print(f"Test Acc: {test_acc:.2f}%")
    
    if test_acc >= 70:
        print(f"✅ EXCELLENT BASELINE! Ready for CHADO implementation")
        print(f"   CHADO Target: 75%+ (need +{75-test_acc:.1f}% improvement)")
    elif test_acc >= 65:
        print(f"✅ GOOD BASELINE! CHADO should reach 75%+")
    else:
        print(f"⚠️  Baseline lower than expected. Check data quality.")
    print(f"{'='*60}\n")

if __name__ == '__main__':
    main()
