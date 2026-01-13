import streamlit as st
from inference import AnimeInference

# Page Configuration
st.set_page_config(
    page_title="Anime Recommendation System",
    layout="wide"
)

# Custom CSS for UI styling
st.markdown("""
<style>
    .cover-img {
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: transform 0.2s;
    }
    .cover-img:hover {
        transform: scale(1.05);
    }
    a {
        text-decoration: none !important;
        color: inherit !important;
    }
    .movie-title {
        font-weight: bold;
        font-size: 16px;
        margin-top: 5px;
        display: block;
        white-space: nowrap; 
        overflow: hidden;
        text-overflow: ellipsis; 
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_engine():
    return AnimeInference()

engine = load_engine()

# Header Section
st.title("Anime Recommendation System")
st.caption("Enter the titles you have watched to discover new recommendations based on content similarity")

# Input Section
st.subheader("Step 1 Enter Your History")

selected_animes = st.multiselect(
    "Search and select titles:", 
    options=engine.indices.index.tolist(),
    placeholder="Example: Naruto, Berserk, Monster..."
)

user_history = []

if selected_animes:
    st.write("---")
    st.write("Rate your interest in these titles")
    
    for i, anime_title in enumerate(selected_animes):
        info = engine.get_anime_info(anime_title)
        
        if info is not None:
            c1, c2 = st.columns([1, 4])
            
            with c1:
                img_url = info.get('main_picture', 'https://placehold.co/100x150')
                st.image(img_url, width=100)
                
            with c2:
                link = info.get('url', '#')
                st.markdown(f"### [{anime_title}]({link})")
                
                genres = info.get('genres', info.get('genre_names', 'Unknown'))
                themes = info.get('theme', info.get('themes', ''))
                st.caption(f"Tags: {genres} | {themes}")
                
                rating = st.select_slider(
                    f"How do you rate {anime_title}?",
                    options=["1 - Bad", "2 - Average", "3 - Good", "4 - Masterpiece"],
                    value="3 - Good",
                    key=f"rate_{i}"
                )
                user_history.append((anime_title, rating))
            st.divider()

# Recommendation Section
if len(user_history) > 0:
    if st.button("Generate Recommendations", type="primary", use_container_width=True):
        st.write("---")
        st.subheader("Step 2 Recommendations For You")
        
        with st.spinner("Analyzing content similarities"):
            recommendations = engine.recommend_by_history(user_history, n=12)
        
        cols = st.columns(4)
        
        for idx, (_, row) in enumerate(recommendations.iterrows()):
            with cols[idx % 4]:
                img = row.get('main_picture', 'https://placehold.co/225x315')
                title = row['title']
                url = row.get('url', '#')
                
                st.markdown(
                    f"""
                    <div style="text-align: center;">
                        <a href="{url}" target="_blank">
                            <img src="{img}" class="cover-img" style="width: 100%; aspect-ratio: 2/3; object-fit: cover;">
                        </a>
                        <a href="{url}" target="_blank" class="movie-title">{title}</a>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
                
                with st.expander("Show Details"):
                    st.caption(f"Genres: {row.get('genre_names', row.get('genres', ''))}")
                    st.caption(f"Synopsis: {str(row.get('synopsis', ''))[:150]}...")
                
                st.write("") 
else:
    st.info("Please select at least one title to get started")