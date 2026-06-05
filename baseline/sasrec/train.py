import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import time
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
plugin_dir = os.path.dirname(current_dir)

root_dir = os.path.dirname(plugin_dir)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
    
from plugin.MoE.dataset.general_dataset import GeneralDataset
from plugin.MoE.utils.model import SASRec
from training_figure import plot_training_results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='./data/amazon', type=str)
    parser.add_argument('--output_dir', default='./saved_models', type=str)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=50) 
    parser.add_argument('--hidden_size', type=int, default=64)
    parser.add_argument('--num_heads', type=int, default=2)    
    parser.add_argument('--dropout', type=float, default=0.2)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Dataset (Train & Valid)
    print("Loading data...")
    try:
        train_dataset = GeneralDataset(args, stage='train')
        valid_dataset = GeneralDataset(args, stage='valid') 
    except Exception as e:
        print(f"Lỗi load dataset: {e}")
        return

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
    
    item_num = train_dataset.item_num 
    seq_size = getattr(train_dataset, 'max_seq_len', 50) 
    
    print(f"Dataset info: Item_num={item_num}, Seq_size={seq_size}")
    print(f"Train batches: {len(train_loader)} | Valid batches: {len(valid_loader)}")

    model = SASRec(args.hidden_size, item_num, seq_size, args.dropout, device, args.num_heads)
    model.to(device)
    
    criterion = nn.CrossEntropyLoss(ignore_index=0) 
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    os.makedirs(args.output_dir, exist_ok=True)
    
    train_loss_history, val_loss_history, l2_history = [], [], []
    best_val_loss = float('inf') 

    print(f"🚀 Bắt đầu train {args.epochs} epochs...")
    
    for epoch in range(args.epochs):
        # --- PHASE: TRAINING ---
        model.train()
        total_train_loss = 0
        start_time = time.time()
        
        for batch in train_loader:
            seq = batch['seq'].to(device) 
            pos = batch['next'].to(device)
            len_seq = batch['len_seq'].to(device)

            optimizer.zero_grad()
            logits = model(seq, len_seq) 
            loss = criterion(logits, pos)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        train_loss_history.append(avg_train_loss)

        # --- PHASE: VALIDATION ---
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in valid_loader:
                seq = batch['seq'].to(device)
                pos = batch['next'].to(device)
                len_seq = batch['len_seq'].to(device)
                
                logits = model(seq, len_seq)
                loss = criterion(logits, pos)
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(valid_loader)
        val_loss_history.append(avg_val_loss)

        with torch.no_grad():
            l2_norm = torch.norm(model.item_embeddings.weight, p=2).item()
            l2_history.append(l2_norm)
            
        print(f"Epoch {epoch+1:02d}/{args.epochs} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"L2: {l2_norm:.2f} | "
              f"Time: {time.time()-start_time:.1f}s")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model_name = os.path.basename(args.data_dir) + "_best_model.pt"
            save_path = os.path.join(args.output_dir, model_name)
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'item_num': item_num,
                'hidden_size': args.hidden_size,
            }, save_path)

    final_model_name = os.path.basename(args.data_dir) + "_last_model.pt"
    torch.save(model.state_dict(), os.path.join(args.output_dir, final_model_name))
    print(f"Complete Training Progress. Best Val Loss: {best_val_loss:.4f}")
    plot_name = os.path.basename(args.data_dir) + "_training_plot.png"
    plot_path = os.path.join(args.output_dir, plot_name)
    
    plot_training_results(train_loss_history, l2_history, plot_path, val_loss_history=val_loss_history)

if __name__ == '__main__':
    main()