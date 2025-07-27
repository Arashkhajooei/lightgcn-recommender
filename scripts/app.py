# scripts/app.py
import streamlit as st
import torch
import pandas as pd

st.set_page_config(page_title="📚 LightGCN Recommender", layout="centered")
st.title("📚 Book Recommender (LightGCN)")

@st.cache_data
def load_book_metadata():
    df = pd.read_csv("data/Books.csv")
    return df.set_index("ISBN").to_dict("index")

book_info = load_book_metadata()


def display_book(idx, idx2book, book_info):
    isbn = idx2book.get(idx)
    info = book_info.get(isbn, {})
    title = info.get("Book-Title", "Unknown Title")
    author = info.get("Book-Author", "Unknown Author")
    img_url = info.get("Image-URL-L", "")
    
    st.markdown(f"**{title}**  \nAuthor: {author}")
    if img_url and isinstance(img_url, str):
        st.image(img_url, width=120)
    st.markdown("---")


# Load embeddings
@st.cache_resource
def load_embeddings():
    data = torch.load("artifacts/embeddings.pt", map_location="cpu", weights_only=False)
    return data["user_embedding"], data["item_embedding"], data["user2idx"], data["idx2book"]

user_emb, item_emb, user2idx, idx2book = load_embeddings()

# UI input
user_id_input = st.selectbox("Select a User ID", sorted(user2idx.keys()))


if user_id_input:
    if user_id_input not in user2idx:
        st.warning("⚠️ User not found in training data.")
    else:
        user_idx = user2idx[user_id_input]
        scores = torch.matmul(user_emb[user_idx], item_emb.T)
        top_k = torch.topk(scores, 10).indices.numpy()

        st.subheader("📖 Top 10 Book Recommendations:")
        for i, idx in enumerate(top_k, 1):
            st.markdown(f"### {i}.")
            display_book(idx, idx2book, book_info)

