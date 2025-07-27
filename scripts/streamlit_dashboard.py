import streamlit as st
import torch
import pandas as pd
import numpy as np

import torch.serialization  # make sure this is imported

@st.cache_data
def load_embeddings():
    with torch.serialization.safe_globals([np.core.multiarray.scalar]):
        data = torch.load("artifacts/embeddings.pt", map_location="cpu", weights_only=False)
    return (
        data["user_embedding"],
        data["item_embedding"],
        data["user2idx"],
        data["book2idx"],
        data["idx2book"]
    )


# Load book metadata
@st.cache_data
def load_books():
    df = pd.read_csv("data/Books.csv")
    return df.set_index("ISBN").to_dict("index")

# Utility: top-N
def get_top_n(query_vec, all_vecs, top_n=10, exclude_idx=None):
    scores = torch.matmul(query_vec, all_vecs.T).squeeze()
    if exclude_idx is not None:
        scores[exclude_idx] = -1e9
    top_scores, top_indices = torch.topk(scores, top_n)
    return top_indices.numpy(), top_scores.numpy()

# Display book cards
def show_books(indices, idx2book, book_info):
    for idx in indices:
        isbn = idx2book.get(idx)
        info = book_info.get(isbn, {})
        title = info.get("Book-Title", "Unknown Title")
        author = info.get("Book-Author", "Unknown Author")
        img_url = info.get("Image-URL-L", "")
        st.markdown(f"**{title}**  \nAuthor: {author}")
        if img_url:
            st.image(img_url, width=120)

# Load everything
user_emb, item_emb, user2idx, book2idx, idx2book = load_embeddings()
book_info = load_books()

# UI
st.set_page_config(page_title="LightGCN Recommender", layout="wide")
st.title("📚 LightGCN Book Recommender")

mode = st.radio("Choose Mode", ["User-Based Recommendation", "Item-Based Similarity"])

if mode == "User-Based Recommendation":
    user_id = st.selectbox("Select a User ID", sorted(user2idx.keys()))
    if st.button("Recommend Books"):
        user_idx = user2idx[user_id]
        seen_books = []  # You can load this from train_df if needed
        user_vec = user_emb[user_idx]
        top_idxs, _ = get_top_n(user_vec, item_emb, top_n=10, exclude_idx=seen_books)
        st.subheader("📖 Recommended Books")
        show_books(top_idxs, idx2book, book_info)

else:
    book_isbn = st.selectbox("Select a Book (ISBN)", sorted(book2idx.keys()))
    if st.button("Find Similar Books"):
        book_idx = book2idx[book_isbn]
        book_vec = item_emb[book_idx]
        top_idxs, _ = get_top_n(book_vec, item_emb, top_n=10, exclude_idx=book_idx)
        st.subheader("🔍 Similar Books")
        show_books(top_idxs, idx2book, book_info)
