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
from models.audio_visual_baseline import AudioVisualBaseline

def compute_class_weights(dataset):
    labels = [sample['label'] for sample in dataset.data]
    unique, counts = np.unique(labels, return_counts=True)
    total = len(labels)
    weights = total / (len(unique) * counts)
    
    print("\nClass distribution and weights:")
    emotion_names = ['happy', 'sad', 'anger', 'surprise', 'disgust', 'fear']
    for cls, count, weight in zip(unique, counts, weights):
        print(f"  {emotion_names[cls]:10s}: {count:5d} samples ({100*count/total:5.1f}%) -> weight: {weight:.3f}")
    
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
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        
        logits = model(audio, visual)
        loss = criterion(logits, labels)
        
        if torch.isnan(loss):
            continue
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = logits.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100.*correct/total:.2f}%'})
    
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
            labels = batch['label'].to(device)
            
            logits = model(audio, visual)
            loss = criterion(logits, labels)
            
            if not torch.isnan(loss):
                total_loss += loss.item()
            
            _, predicted = logits.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    print("\nPer-class accuracy:")
    emotion_names = ['happy', 'sad', 'anger', 'surprise', 'disgust', 'fear']
    for i in range(6):
        mask = all_labels == i
        if mask.sum() > 0:
            class_acc = (all_preds[mask] == all_labels[mask]).sum() / mask.sum() * 100
            print(f"  {emotion_names[i]:10s}: {class_acc:5.2f}%")
    
    return total_loss / len(loader), 100. * correct / total

def main():
    device = 'cuda:0'
    batch_size = 64  # Increased since simpler model
    epochs = 100
    lr = 1e-3
    weight_decay = 1e-4
    
    print("="*60)
    print("AUDIO-VISUAL BASELINE (TARGET: 70%+)")
    print("="*60)
    
    print("\nLoading datasets...")
    train_dataset = CMUMOSEIDataset('data/processed/train_data.pkl')
    val_dataset = CMUMOSEIDataset('data/processed/val_data.pkl')
    test_dataset = CMUMOSEIDataset('data/processed/test_data.pkl')
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    print(f"✓ Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    class_weights = compute_class_weights(train_dataset).to(device)
    
    print("\nInitializing model...")
    model = AudioVisualBaseline(
        audio_dim=74,
        visual_dim=35,
        hidden_dim=256,
        num_classes=6
    ).to(device)
    
    print(f"✓ Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    
    best_val_acc = 0
    results = []
    
    for epoch in range(epochs):
        print(f"\n{'='*60}\nEpoch {epoch+1}/{epochs}\n{'='*60}")
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, epoch+1)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()
        
        print(f"\nTrain: {train_loss:.4f} / {train_acc:.2f}% | Val: {val_loss:.4f} / {val_acc:.2f}%")
        
        results.append({'epoch': epoch+1, 'train_acc': train_acc, 'val_acc': val_acc})
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'checkpoints/av_baseline_best.pth')
            print(f"✓ Best: {best_val_acc:.2f}%")
        
        if epoch > 30 and val_acc < best_val_acc - 5:
            print("\nEarly stopping")
            break
    
    print(f"\n{'='*60}\nFINAL TEST\n{'='*60}")
    model.load_state_dict(torch.load('checkpoints/av_baseline_best.pth'))
    test_loss, test_acc = validate(model, test_loader, criterion, device)
    
    print(f"\nTest: {test_acc:.2f}% | Best Val: {best_val_acc:.2f}%")
    
    with open('results/av_baseline_results.json', 'w') as f:
        json.dump({'best_val_acc': best_val_acc, 'test_acc': test_acc, 'results': results}, f)
    
    print(f"\n{'='*60}")
    if test_acc >= 68:
        print(f"✅ GOOD! CHADO should reach 75%+ (need +{75-test_acc:.1f}%)")
    else:
        print(f"⚠️  Lower than expected. CHADO target may be challenging.")
    print(f"{'='*60}\n")

if __name__ == '__main__':
    main()
