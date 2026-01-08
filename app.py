# Requirements for this file: Open terminal and run this command below:
# pip install streamlit chromadb sentence-transformers pandas

import streamlit as st
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
import os

# --- Configs ---
st.set_page_config(
    page_title="Manga AI Recommender",
    page_icon="https://www.flaticon.com/free-icons/manga",
    layout="wide"
)

# Load data and model 
@st.cache_resource
def load_resources():
    
    # Load dataframe to get cover image and details
    try:
        df = pd.read_pickle('processed_manga.pkl')
        df['mal_id'] = df['mal_id'].astype(str) # string
        if df['mal_id'].duplicated().any():
            # Drop duplicates
            df = df.drop_duplicates(subset=['mal_id'], keep='first')
        manga_dict = df.set_index('mal_id').to_dict('index') # Fast lookup
    except Exception as e:
        st.error(f"Loi xu ly du lieu: {e}")
        return None, None, None, None

    # Load chromaDB
    try:
        chroma_client = chromadb.PersistentClient(path="./manga_chroma_db")
        collection = chroma_client.get_collection("manga_collection")
    except Exception as e:
        st.error(f"Loi ket noi voi co so du lieu: {e}")
        return None, None, None, None

    # Load AI model
    # Change model here if needed
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
    except Exception as e:
        st.error(f"Loi load model AI: {e}")
        return None, None, None, None
    
    return df, manga_dict, collection, model

df, manga_dict, collection, model = load_resources()

# Sidebar UI
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/c/c3/Python-logo-notext.svg", width=50)
    st.title("Bo loc")
    
    # Hybrid Ranking
    st.markdown("### Trong so tim kiem")
    semantic_weight = st.slider("Do sat nghia (Noi dung)", 0.0, 1.0, 0.7, 0.1)
    popularity_weight = st.slider("Do pho bien (Popularity)", 0.0, 1.0, 0.3, 0.1)
    
    st.info(f"Cong thuc: {semantic_weight:.1f} * Do sat nghia + {popularity_weight:.1f} * Do pho bien")
    
    num_results = st.number_input("So luong ket qua", min_value=1, max_value=20, value=5)

# Main UI
st.title("Manga AI Recommender System")
st.markdown("""
He thong goi y manga thong minh su dung **RAG (Retrieval-Augmented Generation)**.
Nhap bat cu thu gi: Mo ta cot truyen, the loai, tac gia, ... ma ban muon doc. Hoac tham chi la tam trang cua ban ngay hom nay ?
""")

# Search bar
query = st.text_input("Nhap noi dung ban muon tim (Vi du: Assasination Classroom, Horror, Ninja, Pirates,...)", "")

# Data processing
if st.button("Tim kiem") and query:
    if not collection or not model:
        st.error("He thong chua san sang. Vui long kiem tra lai du lieu.")
    else:
        with st.spinner(f"Dang tim kiem truyen phu hop voi ban..."):
            # Vectorize questions
            query_vec = model.encode(query).tolist()
            
            # ChromaDB query, get the top 30 mangas that meet the requirements
            results = collection.query(
                query_embeddings=[query_vec],
                n_results=30
            )
            
            # Calculate hybrid ranking and get info
            final_results = []
            ids = results['ids'][0]
            metadatas = results['metadatas'][0]
            distances = results['distances'][0]
            
            for i in range(len(ids)):
                m_id = ids[i]
                
                # Calculate similarity
                semantic_score = 1 - distances[i]
                
                # Get popularity score from metadata
                pop_score = metadatas[i].get('popularity_score', 0)
                
                # Calculate hybrid score)
                final_score = (semantic_score * semantic_weight) + (pop_score * popularity_weight)
                
                # Get info
                info = manga_dict.get(m_id, {})
                image_url = info.get('main_picture', 'https://via.placeholder.com/150')
                
                final_results.append({
                    "id": m_id,
                    "title": metadatas[i]['title'],
                    "score": final_score,
                    "semantic_score": semantic_score,
                    "genres": metadatas[i]['genres'],
                    "synopsis": info.get('synopsis', 'No synopsis available.'),
                    "url": metadatas[i]['url'],
                    "image": image_url,
                    "mal_score": metadatas[i]['score']
                })
            
            # Rearrange results
            final_results.sort(key=lambda x: x['score'], reverse=True)
            
            # Get top N results
            display_results = final_results[:num_results]
            
            # Output
            st.success(f"Tim thay {len(display_results)} truyen phu hop voi tieu chi cua ban.")
            
            for item in display_results:
                with st.container():
                    col1, col2 = st.columns([1, 4])
                    
                    with col1:
                        # Show cover image
                        try:
                            img_src = item['image']
                            if isinstance(img_src, dict):
                                img_src = img_src.get('large', img_src.get('medium', ''))
                            st.image(img_src, use_container_width=True)
                        except:
                            st.write("No Image")
                            
                    with col2:
                        st.subheader(f"[{item['title']}]({item['url']})")
                        st.caption(f"MAL Score: {item['mal_score']} | Match Score: {item['score']:.4f}")
                        
                        # Show matching progress
                        st.progress(min(item['score'], 1.0))
                        
                        st.write(f"**Genre:** {item['genres']}")
                        
                        with st.expander("Doc tom tat cot truyen"):
                            st.write(item['synopsis'])
                    
                    st.divider()

