import pickle
import pandas as pd
import numpy as np
import difflib
import ast

class AnimeInference:
    def __init__(self, model_dir='models'):
        try:
            # Load data and models
            self.anime_df = pickle.load(open(f'{model_dir}/anime_data.pkl', 'rb'))
            self.cosine_sim, self.indices = pickle.load(open(f'{model_dir}/content_model.pkl', 'rb'))
            
            # Rating weight map
            self.weight_map = {
                "4 - Masterpiece": 2.5,
                "3 - Good": 1.0,
                "2 - Average": 0.5,
                "1 - Bad": -1.0 
            }
            
            # Helper function to extract image URL from the nested dictionary
            def extract_img(x):
                try:
                    if isinstance(x, str):
                        x = ast.literal_eval(x)
                    return x.get('jpg', {}).get('image_url', 'https://placehold.co/225x315?text=No+Image')
                except:
                    return 'https://placehold.co/225x315?text=No+Image'

            # Process image URLs if not already processed
            if 'main_picture' not in self.anime_df.columns:
                if 'images' in self.anime_df.columns:
                    self.anime_df['main_picture'] = self.anime_df['images'].apply(extract_img)
                else:
                    self.anime_df['main_picture'] = 'https://placehold.co/225x315?text=No+Image'
            
            # Generate external links if missing
            if 'url' not in self.anime_df.columns:
                self.anime_df['url'] = self.anime_df['title'].apply(
                    lambda x: f"https://www.google.com/search?q={str(x).replace(' ', '+')}+anime"
                )

        except Exception as e:
            print(f"Error loading model: {e}")

    def get_fuzzy_title(self, query):
        # Find closest matching title
        all_titles = self.indices.index.tolist()
        matches = difflib.get_close_matches(query, all_titles, n=1, cutoff=0.5)
        return matches[0] if matches else None


    def get_anime_info(self, title):
        # Get details for a specific title
        try:
            idx = self.indices[title]
            if isinstance(idx, (pd.Series, pd.Index, np.ndarray)): 
                idx = idx[0]
            return self.anime_df.iloc[idx]
        except:
            return None

    def recommend_by_history(self, history_list, n=12):
        # history_list: [('Title', 'Rating'), ...]
        final_sim_scores = np.zeros(self.cosine_sim.shape[0], dtype=np.float32)
        watched_indices = []

        for title, level in history_list:
            correct_title = self.get_fuzzy_title(title)
            if correct_title:
                idx = self.indices[correct_title]
                if isinstance(idx, (pd.Series, pd.Index, np.ndarray)): 
                    idx = idx[0]
                
                watched_indices.append(idx)
                weight = self.weight_map.get(level, 1.0)
                final_sim_scores += self.cosine_sim[idx] * weight

        # Filter out already watched items
        for idx in watched_indices:
            final_sim_scores[idx] = -999 

        # Get Top N recommendations
        top_indices = np.argsort(final_sim_scores)[::-1][:n]
        return self.anime_df.iloc[top_indices].copy()