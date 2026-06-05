import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
import argparse

from layer import LightGCN
from utils import sample_bpr_batch, plot_loss
from bprloss import BPRLoss


def parse_args():
    parser = argparse.ArgumentParser(description="Training LightGCN (He et al., SIGIR 2020)")

    parser.add_argument('--data_file', type=str, default='processed_graph_data.pt',
                        help='Path to the input graph data (.pt)')
    parser.add_argument('--export_file', type=str, default='gcn_embeddings.pt',
                        help='Path to save the output embeddings (.pt)')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='Number of Epochs (paper used 1000)')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='Batch size (paper used 1024 or 2048)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning Rate (default 0.001, Adam optimizer)')
    parser.add_argument('--reg', type=float, default=1e-4,
                        help='L2 Regularization weight lambda (default 1e-4)')
    parser.add_argument('--emb_dim', type=int, default=64,
                        help='Embedding dimension (default 64)')
    parser.add_argument('--num_layers', type=int, default=3,
                        help='Number of GCN layers (paper tested K=1,2,3,4; default K=3)')
    parser.add_argument('--dataset', type=str, default=None, help='Dataset Name')

    # System
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'],
                        help='Training device')

    return parser.parse_args()


def main():
    args = parse_args()

    if args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    print(f"Training on: {device}")
    print(f"Settings: Epochs={args.epochs}, Batch={args.batch_size}, "
          f"LR={args.lr}, Dim={args.emb_dim}, Layers={args.num_layers}, Reg={args.reg}")
    print("Model: LightGCN (He et al., SIGIR 2020)")

    try:
        data = torch.load(args.data_file)
    except FileNotFoundError:
        print(f"Error: '{args.data_file}'")
        return

    train_dict = data['train_dict']
    node_item_mapping = data['node_item_mapping']
    A_hat = data['adj_norm']
    num_users = data['num_users']
    num_nodes = data['num_nodes']
    num_items = num_nodes - num_users

    print(f"Graph info: {num_users} users, {num_items} items, {num_nodes} total nodes")

    model = LightGCN(
        num_users=num_users,
        num_items=num_items,
        embedding_dim=args.emb_dim,
        num_layers=args.num_layers
    ).to(device)

    A_hat = A_hat.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = BPRLoss(reg_weight=args.reg)

    print(f"\n--- Start Training (LightGCN, K={args.num_layers} layers) ---")
    model.train()
    start_time = time.time()

    loss_history = []
    reg_history = []

    for epoch in range(args.epochs):
        optimizer.zero_grad()

       
        final_embs, initial_embs = model(A_hat)

        users_idx, pos_idx, neg_idx = sample_bpr_batch(
            train_dict, num_users, num_nodes, args.batch_size
        )

        users_idx = users_idx.to(device)
        pos_idx = pos_idx.to(device)
        neg_idx = neg_idx.to(device)

        u_final = final_embs[users_idx]
        i_pos_final = final_embs[pos_idx]
        i_neg_final = final_embs[neg_idx]

        u_0 = initial_embs[users_idx]
        i_pos_0 = initial_embs[pos_idx]
        i_neg_0 = initial_embs[neg_idx]

        loss, bpr, reg = criterion(
            users_idx, pos_idx, neg_idx,
            u_final, i_pos_final, i_neg_final,
            u_0, i_pos_0, i_neg_0
        )

        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())
        reg_history.append(reg.item())

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:04d}/{args.epochs} | "
                  f"Total Loss: {loss.item():.4f} | "
                  f"BPR: {bpr.item():.4f} | "
                  f"Reg: {reg.item():.6f}")

    print(f"\nTraining finished in {time.time() - start_time:.2f}s")

    plot_loss(loss_history, reg_history, args.dataset)
    print("Saved Loss Figure.")

    # Export embeddings
    model.eval()
    with torch.no_grad():
        final_node_embeddings, _ = model(A_hat)

        final_dict = {}
        for idx, original_id in enumerate(node_item_mapping):
            final_dict[original_id] = final_node_embeddings[idx].cpu()

        torch.save(final_dict, args.export_file)
        print(f"--> Saved embeddings to '{args.export_file}'")
        print(f"--> Embedding matrix shape: {final_node_embeddings.shape}")


if __name__ == "__main__":
    main()