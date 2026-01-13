import pandas as pd
import pickle
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Configuration
ANIME_PATH = 'final.csv' 
MODEL_DIR = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)

def step_1_process_anime_data():
    print("Step 1 Loading Pre Processed Data")
    
    df = pd.read_csv(ANIME_PATH)
    
    # Process Score Tag
    # Converts numerical score to text tags for vectorization
    def get_score_tag(s):
        try:
            s = float(s)
            if s >= 8.0: return 'quality_masterpiece '
            elif s >= 7.5: return 'quality_excellent '
            elif s >= 7.0: return 'quality_good '
            else: return 'quality_average '
        except: return ''
    
    # Check for score column
    score_col = 'score' if 'score' in df.columns else 'rating'
    if score_col in df.columns:
        df['score_tag'] = df[score_col].apply(get_score_tag)
    else:
        df['score_tag'] = ''

    # Handle Missing Values for key columns
    # We assume data is already clean but fillna protects against errors
    cols_to_check = ['genre_names', 'themes', 'demographics', 'synopsis_processed']
    for col in cols_to_check:
        if col in df.columns:
            df[col] = df[col].fillna('')

    # Create Weighted Soup
    # We assume the columns in CSV are already cleaned strings
    def create_weighted_soup(x):
        # Genres x3
        g = (str(x.get('genre_names', '')) + ' ') * 3
        
        # Themes x2
        t = (str(x.get('themes', '')) + ' ') * 2
        
        # Demographics x2
        # Try finding 'demographic' or 'demographics' column
        d_val = x.get('demographic', x.get('demographics', ''))
        d = (str(d_val) + ' ') * 2
        
        # Score x2
        s = (str(x.get('score_tag', '')) + ' ') * 2
        
        # Synopsis x1
        # Prioritize 'synopsis_processed', fallback to 'synopsis'
        syn = str(x.get('synopsis_processed', x.get('synopsis', '')))
        
        return f"{g}{t}{d}{s}{syn}"

    print("   Creating Weighted Soup")
    df['soup'] = df.apply(create_weighted_soup, axis=1)
    
    # Save processed dataframe
    with open(f'{MODEL_DIR}/anime_data.pkl', 'wb') as f:
        pickle.dump(df, f)
    
    return df

def step_2_train_content_model(df):
    print("Step 2 Training Content Model")
    
    try:
        # Using TfidfVectorizer
        # min_df 2 ignores terms that appear in only 1 document
        # max_features 10000 limits vocabulary size to save RAM
        tfidf = TfidfVectorizer(
            stop_words='english', 
            min_df=2, 
            max_features=10000,
            ngram_range=(1, 2)
        )
        
        # Create Vector Matrix
        # Using float32 to save RAM
        print("   Vectorizing soup...")
        vector_matrix = tfidf.fit_transform(df['soup']).astype(np.float32)

        print(f"   Calculating Cosine Similarity for {df.shape[0]} items...")
        cosine_sim = cosine_similarity(vector_matrix).astype(np.float32)

        # Create Index Mapping
        #indices = pd.Series(df.index, index=df['title']).drop_duplicates()
        indices = pd.Series(df.index, index=df['title'])

        
        # Save Model
        with open(f'{MODEL_DIR}/content_model.pkl', 'wb') as f:
            pickle.dump((cosine_sim, indices), f)
            
        print("   Model saved successfully")
        
    except MemoryError:
        print("Error Memory Limit Exceeded. Try reducing max_features.")
    except Exception as e:
        print(f"Error {e}")

# Main Execution
if __name__ == "__main__":
    print("Starting Training Pipeline")
    
    try:
        df_processed = step_1_process_anime_data()
        step_2_train_content_model(df_processed)
        print("Pipeline Completed")
        
    except Exception as e:
        print(f"Pipeline Failed {e}")