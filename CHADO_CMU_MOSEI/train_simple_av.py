import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import sys
from tqdm import tqdm
import numpy as np

sys.path.append('src')
from data_processing.dataset import CMUMOSEIDataset
from models.simple_av_baseline import SimpleAVBaseline

def compute_class_weights(dataset):
    labels = [s['label'] for s in dataset.data]
    unique, counts = np.unique(labels, return_counts=True)
    total = len(labels)
    weights = total / (len(unique) * counts)
    return torch.FloatTensor(weights)

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch in tqdm(loader, desc='Training'):
        audio = batch['audio'].to(device)
        visual = batch['visual'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        logits = model(audio, visual)
        loss = criterion(logits, labels)
        
        if torch.isnan(loss):
            continue
            
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        _, pred = logits.max(1)
        correct += pred.eq(labels).sum().item()
        total += labels.size(0)
    
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
            
            _, pred = logits.max(1)
            correct += pred.eq(labels).sum().item()
            total += labels.size(0)
            
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    emotions = ['happy', 'sad', 'anger', 'surprise', 'disgust', 'fear']
    print("\nPer-class accuracy:")
    for i in range(6):
        mask = all_labels == i
        if mask.sum() > 0:
            acc = (all_preds[mask] == all_labels[mask]).sum() / mask.sum() * 100
            print(f"  {emotions[i]:10s}: {acc:5.2f}%")
    
    return total_loss / len(loader), 100. * correct / total

def main():
    device = 'cuda:0'
    batch_size = 128
    epochs = 100
    
    print("="*60)
    print("SIMPLE AUDIO-VISUAL BASELINE")
    print("="*60)
    
    train_dataset = CMUMOSEIDataset('data/processed/train_data.pkl')
    val_dataset = CMUMOSEIDataset('data/processed/val_data.pkl')
    test_dataset = CMUMOSEIDataset('data/processed/test_data.pkl')
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    print(f"\n✓ Train: {len(train_dataset)}")
    print(f"✓ Val: {len(val_dataset)}")
    print(f"✓ Test: {len(test_dataset)}")
    
    class_weights = compute_class_weights(train_dataset).to(device)
    print(f"\n✓ Class weights: {class_weights.cpu().numpy()}")
    
    model = SimpleAVBaseline(audio_dim=74, visual_dim=35, num_classes=6).to(device)
    print(f"✓ Model params: {sum(p.numel() for p in model.parameters()):,}")
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=7, factor=0.5)
    
    best_val_acc = 0
    patience = 0
    
    for epoch in range(epochs):
        print(f"\n{'='*60}\nEpoch {epoch+1}/{epochs}\n{'='*60}")
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_acc)
        new_lr = optimizer.param_groups[0]['lr']
        
        if new_lr != old_lr:
            print(f"\n✓ LR: {old_lr:.2e} → {new_lr:.2e}")
        
        print(f"\nTrain: {train_loss:.4f} / {train_acc:.2f}%")
        print(f"Val:   {val_loss:.4f} / {val_acc:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'checkpoints/simple_av_best.pth')
            print(f"✓ Best: {best_val_acc:.2f}%")
            patience = 0
        else:
            patience += 1
            if patience >= 15:
                print("\n✓ Early stopping")
                break
    
    print(f"\n{'='*60}\nFINAL TEST\n{'='*60}")
    model.load_state_dict(torch.load('checkpoints/simple_av_best.pth'))
    test_loss, test_acc = validate(model, test_loader, criterion, device)
    
    print(f"\n{'='*60}")
    print(f"Best Val:  {best_val_acc:.2f}%")
    print(f"Test Acc:  {test_acc:.2f}%")
    print(f"{'='*60}")
    
    if test_acc >= 65:
        print(f"✅ ACCEPTABLE! CHADO target: 75% (+{75-test_acc:.1f}%)")
    elif test_acc >= 60:
        print(f"⚠️  LOW. CHADO target will be challenging.")
    else:
        print(f"❌ TOO LOW. Data quality issues.")
    print(f"{'='*60}\n")

if __name__ == '__main__':
    main()
