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
        if not os.path.exists('processed_manga.pkl'):
            st.error("⚠️ Thiếu file 'processed_manga.pkl'.")
            return None, None, None, None

        df = pd.read_pickle('processed_manga.pkl')
        df['mal_id'] = df['mal_id'].astype(str)
        if df['mal_id'].duplicated().any():
            df = df.drop_duplicates(subset=['mal_id'], keep='first')
        manga_dict = df.set_index('mal_id').to_dict('index')
    except Exception as e:
        st.error(f"Lỗi Dataframe: {e}")
        return None, None, None, None

    # Load ChromaDB 
    try:
        # Multilingual
        if os.path.exists("./manga_chroma_db_vn"):
            print("--- Đang dùng Database TIẾNG VIỆT ---")
            chroma_client = chromadb.PersistentClient(path="./manga_chroma_db_vn")
            collection = chroma_client.get_collection("manga_collection_vn")
            model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
            st.toast("Đang sử dụng hệ thống Tiếng Việt (MPNet)", icon="🇻🇳")
            
        # Ènglish only
        elif os.path.exists("./manga_chroma_db"):
            print("--- Đang dùng Database TIẾNG ANH ---")
            chroma_client = chromadb.PersistentClient(path="./manga_chroma_db")
            collection = chroma_client.get_collection("manga_collection")
            
            model = SentenceTransformer('all-MiniLM-L6-v2') 
            st.toast("Đang sử dụng hệ thống Tiếng Anh (MiniLM)", icon="🇬🇧")
            
        else:
            st.error("Không tìm thấy thư mục Database.")
            return None, None, None, None

    except Exception as e:
        st.error(f"Lỗi khởi tạo AI: {e}")
        st.warning("Hãy xóa thư mục manga_chroma_db_... và chạy lại file tạo model.")
        return None, None, None, None
    
    return df, manga_dict, collection, model

# STREAMLIT UI

df, manga_dict, collection, model = load_resources()

# Sidebar UI
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/c/c3/Python-logo-notext.svg", width=50)
    st.title("Bộ lọc")
    
    # Hybrid Ranking
    st.markdown("### Trọng số tìm kiếm")
    semantic_weight = st.slider("Độ sát nghĩa (Nội dung)", 0.0, 1.0, 0.7, 0.1)
    popularity_weight = st.slider("Độ phổ biến", 0.0, 1.0, 0.3, 0.1)
    
    st.info(f"Công thức: {semantic_weight:.1f} * Độ sát nghĩa + {popularity_weight:.1f} * Độ phổ biến ")
    
    num_results = st.number_input("Số lượng kết quả", min_value=1, max_value=20, value=5)

# Main UI
st.title("Manga AI Recommender System")
st.markdown("""
Hệ thống gợi ý manga thông minh sử dụng **RAG (Retrieval-Augmented Generation)**.
Nhập bất cứ thứ gì: Mô tả cốt truyện, thể loại, tác giả, ... mà bạn muốn đọc. Hoặc tâm trạng của bạn hôm nay, nếu bạn muốn ?
Tôi sẽ cố gắng tìm những bộ truyện phù hợp nhất với bạn. 
""")

# Search bar
query = st.text_input("Nhập nội dung bạn muốn tìm (Ví du: Assasination Classroom, Horror, Ninja, Pirates,...)", "")

# Data processing
if st.button("Tìm kiếm") and query:
    if not collection or not model:
        st.error("Hệ thống chưa sẵn sàng (Lỗi load resources).")
    else:
        with st.spinner(f"Đang suy luận..."):
            try:
                # Vectorize query
                query_vec = model.encode(query).tolist()
                
                # ChromaDB query 
                results = collection.query(
                    query_embeddings=[query_vec],
                    n_results=30
                )
            except Exception as e:
                st.error(f"Lỗi truy vấn: {e}")
                st.info("Mẹo: Có thể Database và Model bị lệch kích thước vector. Hãy xóa folder DB đi và chạy lại file Model.")
                st.stop()
            
            final_results = []
            
            # Check if no results available
            if not results['ids'] or not results['ids'][0]:
                st.warning("Không tìm thấy truyện nào phù hợp.")
            else:
                ids = results['ids'][0]
                metadatas = results['metadatas'][0]
                distances = results['distances'][0]
                
                for i in range(len(ids)):
                    m_id = ids[i]
                    
                    # Get metadata
                    meta = metadatas[i]
                    
                    title = meta.get('title', 'Unknown Title')
                    genres = meta.get('genres', meta.get('genres_str', 'Unknown')) # Thử cả 2 tên key
                    mal_score = meta.get('score', 0)
                    url = meta.get('url', '#')
                    pop_score = meta.get('popularity_score', 0)
                    
                    # Calculate score
                    semantic_score = 1 - distances[i]
                    final_score = (semantic_score * semantic_weight) + (pop_score * popularity_weight)
                    
                    # Get extra info from DataFrame
                    info = manga_dict.get(m_id, {})
                    synopsis = info.get('synopsis', 'No synopsis available.')
                    
                    # Get image
                    img_src = info.get('main_picture', 'https://via.placeholder.com/150')
                    if isinstance(img_src, dict):
                         img_src = img_src.get('large', img_src.get('medium', ''))
                    
                    final_results.append({
                        "title": title,
                        "score": final_score,
                        "genres": genres,
                        "synopsis": synopsis,
                        "url": url,
                        "image": img_src,
                        "mal_score": mal_score
                    })
                
                # Output
                final_results.sort(key=lambda x: x['score'], reverse=True)
                display_results = final_results[:num_results]
                
                st.success(f"Tìm thấy {len(display_results)} truyện phù hợp!")
                
                for item in display_results:
                    with st.container():
                        col1, col2 = st.columns([1, 4])
                        with col1:
                             st.image(item['image'], use_container_width=True)
                        with col2:
                            st.subheader(f"[{item['title']}]({item['url']})")
                            st.caption(f"MAL: {item['mal_score']} | Match: {item['score']:.2f}")
                            st.write(f"**Thể loại:** {item['genres']}")
                            with st.expander("Tóm tắt"):
                                st.write(item['synopsis'])
                        st.divider()