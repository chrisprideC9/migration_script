# matching_strategies.py

import streamlit as st
import pandas as pd
import numpy as np
from rapidfuzz import process, fuzz
from sklearn.metrics.pairwise import cosine_similarity

def perform_exact_matching(old_df, new_df, status='Regular'):
    """Exact path matching."""
    exact_matches = pd.merge(
        old_df, new_df, 
        left_on='Path', right_on='Path', 
        how='inner',
        suffixes=('_old', '_new')
    )
    
    if not exact_matches.empty:
        exact_matches = exact_matches[['Address_old', 'Address_new', 'Path', 'Title', 'Meta Description', 'H1_old']].copy()
        exact_matches['Match_Type'] = 'Exact Path'
        exact_matches['Confidence_Score'] = 100  # fully confident
        exact_matches = exact_matches.rename(columns={
            'Title': 'Title_old',
            'Meta Description': 'Meta Description_old'
        })
        st.write(f"🔍 Found **{exact_matches.shape[0]}** exact path matches for {status} URLs.")
        
        return exact_matches[['Address_old','Address_new','Match_Type','Confidence_Score','Title_old','Meta Description_old','H1_old']]
    else:
        st.write(f"🔍 Found **0** exact path matches for {status} URLs.")
        return pd.DataFrame()

def perform_fuzzy_matching(old_df, new_df, threshold=90, status='Regular'):
    """Fuzzy path matching using RapidFuzz."""
    path_similarity_matches = []
    matched_new_paths = set()
    
    for old_path in old_df['Path']:
        match, score, _ = process.extractOne(old_path, new_df['Path'].tolist(), scorer=fuzz.ratio)
        if score >= threshold and match not in matched_new_paths:
            old_row = old_df[old_df['Path'] == old_path].iloc[0]
            new_row = new_df[new_df['Path'] == match].iloc[0]
            
            path_similarity_matches.append({
                'Address_old': old_row['Address_old'],
                'Address_new': new_row['Address_new'],
                'Match_Type': 'Similar Path',
                'Confidence_Score': 50,
                'Title_old': old_row['Title'],
                'Meta Description_old': old_row['Meta Description'],
                'H1_old': old_row['H1']
            })
            matched_new_paths.add(match)
    
    if path_similarity_matches:
        path_similarity_df = pd.DataFrame(path_similarity_matches)
        st.write(f"🔍 Found **{path_similarity_df.shape[0]}** path similarity matches for {status} URLs with threshold ≥ {threshold}%.")
        return path_similarity_df[['Address_old','Address_new','Match_Type','Confidence_Score','Title_old','Meta Description_old','H1_old']]
    else:
        st.write(f"🔍 Found **0** path similarity matches for {status} URLs with threshold ≥ {threshold}%.")
        return pd.DataFrame()

def perform_vector_matching(old_df, new_df, status='Regular'):
    """Vector-based similarity matching using cosine similarity."""
    if old_df.empty or new_df.empty:
        st.write(f"⚠️ No data available for vector-based matching for {status} URLs.")
        return pd.DataFrame()
    
    with st.spinner(f"🔄 Performing vector similarity matching for {status} URLs..."):
        # Ensure embeddings are valid
        if any(old_df['Embedding'].apply(lambda x: len(x) == 0)) or any(new_df['Embedding'].apply(lambda x: len(x) == 0)):
            st.error(f"❌ Some embeddings are missing or invalid for {status} URLs.")
            return pd.DataFrame()
        
        old_embeddings = np.stack(old_df['Embedding'].values)
        new_embeddings = np.stack(new_df['Embedding'].values)
    
        # Compute cosine similarity
        similarity_matrix = cosine_similarity(old_embeddings, new_embeddings)
    
        # For each old URL, find the new URL with highest similarity
        top_n = 1
        top_matches_indices = np.argsort(-similarity_matrix, axis=1)[:, :top_n]
    
        vector_matches = []
        matched_new_urls_vector = set()
        
        for i, old_url in enumerate(old_df['Address_old']):
            for j in top_matches_indices[i]:
                new_url = new_df.iloc[j]['Address_new']
                similarity = similarity_matrix[i][j]
                if new_url not in matched_new_urls_vector:
                    # Assign confidence
                    if similarity >= 0.9:
                        confidence_score = 90
                    elif similarity >= 0.75:
                        confidence_score = 60
                    else:
                        confidence_score = 30
                    
                    old_row = old_df.iloc[i]
                    vector_matches.append({
                        'Address_old': old_row['Address_old'],
                        'Address_new': new_url,
                        'Match_Type': 'Vector Similarity',
                        'Confidence_Score': confidence_score,
                        'Title_old': old_row['Title'],
                        'Meta Description_old': old_row['Meta Description'],
                        'H1_old': old_row['H1']
                    })
                    matched_new_urls_vector.add(new_url)
        
        if vector_matches:
            vector_matches_df = pd.DataFrame(vector_matches)
            st.write(f"🔍 Found **{vector_matches_df.shape[0]}** vector similarity matches for {status} URLs.")
            return vector_matches_df[['Address_old','Address_new','Match_Type','Confidence_Score','Title_old','Meta Description_old','H1_old']]
        else:
            st.write(f"🔍 Found **0** vector similarity matches for {status} URLs.")
            return pd.DataFrame()
