# ─── scripts/train_lightgcn.py ────────────────────────────────────────────
import os
import pandas as pd
import torch
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from tqdm.auto import tqdm, trange
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

LIKE_TH = 1
EPOCHS = 30
EMB_DIM = 64
N_LAYERS = 2
BATCH_SIZE = 4096
LEARNING_RATE = 7e-4
WEIGHT_DECAY = 3.2e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ratings_raw = pd.read_csv("data/Ratings.csv")
ratings_raw = ratings_raw[ratings_raw["Book-Rating"] >= LIKE_TH]

def train_test_split(df):
    df['rank_latest'] = df.groupby(['User-ID'])['Book-Rating'].rank(method='first', ascending=False)
    train_df = df[df['rank_latest'] != 1]
    test_df = df[df['rank_latest'] == 1]
    return train_df.drop(columns='rank_latest'), test_df.drop(columns='rank_latest')

train_df, test_df = train_test_split(ratings_raw)
user2idx = {u: i for i, u in enumerate(train_df["User-ID"].unique())}
book2idx = {b: i for i, b in enumerate(train_df["ISBN"].unique())}
train_df["user_idx"] = train_df["User-ID"].map(user2idx)
train_df["book_idx"] = train_df["ISBN"].map(book2idx)
test_df["user_idx"] = test_df["User-ID"].map(user2idx)
test_df["book_idx"] = test_df["ISBN"].map(book2idx)
test_df = test_df.dropna().astype({"user_idx": int, "book_idx": int})
test_df = test_df.sample(n=min(len(test_df), 2000), random_state=42)
n_users, n_books = len(user2idx), len(book2idx)

def get_graph(train_df, n_users, n_books):
    user_ids = train_df['user_idx'].values
    item_ids = train_df['book_idx'].values
    R = sp.coo_matrix((np.ones(len(user_ids)), (user_ids, item_ids)), shape=(n_users, n_books), dtype=np.float32)
    
    # Build symmetric adjacency matrix directly in coo
    rows = np.concatenate([R.row, R.col + n_users])
    cols = np.concatenate([R.col + n_users, R.row])
    data = np.ones(len(rows), dtype=np.float32)
    
    adj = sp.coo_matrix((data, (rows, cols)), shape=(n_users + n_books, n_users + n_books))
    
    rowsum = np.array(adj.sum(axis=1)).flatten()
    d_inv_sqrt = np.power(rowsum, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    norm_adj = d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt
    norm_adj = norm_adj.tocoo()
    
    i = torch.LongTensor(np.vstack((norm_adj.row, norm_adj.col)))
    v = torch.FloatTensor(norm_adj.data)
    return torch.sparse_coo_tensor(i, v, norm_adj.shape).to(DEVICE)

graph = get_graph(train_df, n_users, n_books)

class LightGCNDataset(Dataset):
    def __init__(self, df, n_books):
        self.users = df["user_idx"].values
        self.items = df["book_idx"].values
        self.n_books = n_books
        self.user_item_set = set(zip(self.users, self.items))

    def __len__(self): return len(self.users)

    def __getitem__(self, idx):
        user, pos_item = self.users[idx], self.items[idx]
        neg_item = np.random.randint(self.n_books)
        while (user, neg_item) in self.user_item_set:
            neg_item = np.random.randint(self.n_books)
        return torch.tensor(user), torch.tensor(pos_item), torch.tensor(neg_item)

train_dl = DataLoader(LightGCNDataset(train_df, n_books), batch_size=BATCH_SIZE, shuffle=True)

class LightGCN(nn.Module):
    def __init__(self, n_users, n_items, emb_dim, n_layers, graph):
        super().__init__()
        self.n_users, self.n_items = n_users, n_items
        self.emb_dim, self.n_layers = emb_dim, n_layers
        self.graph = graph
        self.user_embedding = nn.Embedding(n_users, emb_dim)
        self.item_embedding = nn.Embedding(n_items, emb_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def compute_embeddings(self):
        x = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        embs = [x]
        for _ in range(self.n_layers):
            x = torch.sparse.mm(self.graph, x)
            embs.append(x)
        final = torch.mean(torch.stack(embs, dim=0), dim=0)
        return torch.split(final, [self.n_users, self.n_items])

    def forward(self, users, pos, neg):
        u_emb, i_emb = self.compute_embeddings()
        return (u_emb[users] * i_emb[pos]).sum(1), (u_emb[users] * i_emb[neg]).sum(1)

def bpr_loss(pos, neg):
    return -torch.log(torch.sigmoid(pos - neg) + 1e-9).mean()

def evaluate(model):
    model.eval()
    u_emb, i_emb = model.compute_embeddings()
    hist = train_df.groupby("user_idx")["book_idx"].apply(set).to_dict()
    total, hits, ndcg_sum = 0, 0, 0
    with torch.no_grad():
        for _, row in test_df.iterrows():
            u, pos = int(row["user_idx"]), int(row["book_idx"])
            seen = hist.get(u, set())
            candidates = list(set(np.random.choice(n_books, 100, replace=False)) - seen)[:99] + [pos]
            scores = torch.matmul(u_emb[u], i_emb[candidates].T)
            rank = (scores >= scores[-1]).sum().item()
            if rank <= 10:
                hits += 1
                ndcg_sum += 1 / np.log2(rank + 1)
            total += 1
    return hits / total, ndcg_sum / total

model = LightGCN(n_users, n_books, EMB_DIM, N_LAYERS, graph).to(DEVICE)
opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

loss_hist, recall_hist, ndcg_hist = [], [], []
for epoch in trange(EPOCHS):
    model.train()
    epoch_loss = 0
    for users, pos, neg in tqdm(train_dl, leave=False):
        users, pos, neg = users.to(DEVICE), pos.to(DEVICE), neg.to(DEVICE)
        opt.zero_grad()
        ps, ns = model(users, pos, neg)
        loss = bpr_loss(ps, ns)
        reg = (
            model.user_embedding(users).norm(2).pow(2) +
            model.item_embedding(pos).norm(2).pow(2) +
            model.item_embedding(neg).norm(2).pow(2)
        ) / len(users)
        loss += WEIGHT_DECAY * reg
        loss.backward(); opt.step()
        epoch_loss += loss.item()

    recall, ndcg = evaluate(model)
    loss_hist.append(epoch_loss / len(train_dl))
    recall_hist.append(recall)
    ndcg_hist.append(ndcg)
    print(f"Epoch {epoch+1:02d} | Loss: {loss_hist[-1]:.4f} | Recall@10: {recall:.4f} | NDCG@10: {ndcg:.4f}")

# ─── Save artifacts ───────────────────────────────────────────────────
os.makedirs("artifacts", exist_ok=True)
pd.DataFrame({'loss': loss_hist}).to_csv("artifacts/loss.csv", index=False)
pd.DataFrame({'recall': recall_hist}).to_csv("artifacts/recall.csv", index=False)
pd.DataFrame({'ndcg': ndcg_hist}).to_csv("artifacts/ndcg.csv", index=False)
torch.save(model.state_dict(), "models/lightgcn.pt")

# Save embeddings
users_emb, items_emb = model.compute_embeddings()
torch.save({
    "user_embedding": users_emb.cpu(),
    "item_embedding": items_emb.cpu(),
    "user2idx": user2idx,
    "book2idx": book2idx,
    "idx2book": {v: k for k, v in book2idx.items()}
}, "artifacts/embeddings.pt")

# ─── 11 │ Save Model + Metrics + Embeddings ─────────────────────────────
import os, json

os.makedirs("models", exist_ok=True)
os.makedirs("artifacts", exist_ok=True)

# Save model
torch.save(model.state_dict(), "models/lightgcn_model.pt")

# Save embeddings
user_emb, item_emb = model.compute_embeddings()
torch.save({
    "user_embedding": user_emb.cpu(),
    "item_embedding": item_emb.cpu(),
    "user2idx": user2idx,
    "book2idx": book2idx,
    "idx2book": {v: k for k, v in book2idx.items()}
}, "artifacts/embeddings.pt")

# Save metrics JSON
with open("artifacts/metrics.json", "w") as f:
    json.dump({
        "recall@10": recall_hist[-1],
        "ndcg@10": ndcg_hist[-1],
        "final_loss": loss_hist[-1]
    }, f, indent=2)

# Save performance plot
plt.savefig("artifacts/metrics_plot.png")
