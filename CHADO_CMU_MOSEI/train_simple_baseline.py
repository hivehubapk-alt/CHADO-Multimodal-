import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import sys
from pathlib import Path
from tqdm import tqdm
import json

sys.path.append('src')
from data_processing.dataset import CMUMOSEIDataset
from models.simple_baseline import SimpleMultimodalBaseline

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc='Training')
    for batch in pbar:
        audio = batch['audio'].to(device)
        visual = batch['visual'].to(device)
        text = batch['text'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        
        logits = model(audio, visual, text)
        loss = criterion(logits, labels)
        
        loss.backward()
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
    
    with torch.no_grad():
        for batch in tqdm(loader, desc='Validating'):
            audio = batch['audio'].to(device)
            visual = batch['visual'].to(device)
            text = batch['text'].to(device)
            labels = batch['label'].to(device)
            
            logits = model(audio, visual, text)
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            _, predicted = logits.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
    
    return total_loss / len(loader), 100. * correct / total

def main():
    # Config
    device = 'cuda:0'  # Will use GPU 5 (CUDA_VISIBLE_DEVICES=5)
    batch_size = 32
    epochs = 30
    lr = 1e-4
    
    print("="*60)
    print("TRAINING SIMPLE BASELINE")
    print("="*60)
    
    # Load data
    print("\nLoading datasets...")
    train_dataset = CMUMOSEIDataset('data/processed/train_data.pkl')
    val_dataset = CMUMOSEIDataset('data/processed/val_data.pkl')
    test_dataset = CMUMOSEIDataset('data/processed/test_data.pkl')
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    print(f"✓ Train: {len(train_dataset)} samples")
    print(f"✓ Val: {len(val_dataset)} samples")
    print(f"✓ Test: {len(test_dataset)} samples")
    
    # Model
    print("\nInitializing model...")
    model = SimpleMultimodalBaseline(
        audio_dim=74,
        visual_dim=35,
        text_dim=300,
        hidden_dim=128,
        num_classes=6
    ).to(device)
    
    print(f"✓ Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Training loop
    best_val_acc = 0
    results = []
    
    for epoch in range(epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"{'='*60}")
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        scheduler.step()
        
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
            torch.save(model.state_dict(), 'checkpoints/simple_baseline_best.pth')
            print(f"✓ Best model saved! Val Acc: {best_val_acc:.2f}%")
    
    # Test evaluation
    print(f"\n{'='*60}")
    print("FINAL TEST EVALUATION")
    print(f"{'='*60}")
    
    model.load_state_dict(torch.load('checkpoints/simple_baseline_best.pth'))
    test_loss, test_acc = validate(model, test_loader, criterion, device)
    
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Acc: {test_acc:.2f}%")
    print(f"Best Val Acc: {best_val_acc:.2f}%")
    
    # Save results
    with open('results/simple_baseline_results.json', 'w') as f:
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
    print(f"{'='*60}\n")

if __name__ == '__main__':
    main()
