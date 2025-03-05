# ai_matching.py

import os
import openai
import streamlit as st
import pandas as pd
from rapidfuzz import process, fuzz

# If you haven't already set the API key in main.py or elsewhere:
# openai.api_key = os.getenv("OPENAI_API_KEY")

def perform_ai_matching(unmatched_df, new_df, model="gpt-3.5-turbo", max_new_urls=50):
    """
    Uses OpenAI's API to pick the best matching new URL for each unmatched old URL.
    
    Arguments:
      - unmatched_df : DataFrame of unmatched old URLs with columns like:
            ['Address_old','Title_old','Meta Description_old','H1_old', ...]
      - new_df       : DataFrame of new URLs (['Address_new','H1', 'Title', etc.])
      - model        : The OpenAI model to use (e.g., 'gpt-3.5-turbo' or 'gpt-4').
      - max_new_urls : Maximum # of new URLs to pass in a single prompt (for token/cost control).
    
    Returns:
      A DataFrame of AI-based matches with columns:
        ['Address_old','Address_new','Match_Type','Confidence_Score','Title_old','Meta Description_old','H1_old']
    """
    if unmatched_df.empty:
        return pd.DataFrame()
    if new_df.empty:
        return pd.DataFrame()

    ai_matches = []
    # Possibly chunk or limit new_df if large:
    all_new_candidates = new_df[['Address_new','H1']].values.tolist()
    limited_new_candidates = all_new_candidates[:max_new_urls]

    for _, row in unmatched_df.iterrows():
        old_url = row['Address_old']
        old_title = row.get('Title_old', '')
        old_desc = row.get('Meta Description_old', '')
        old_h1   = row.get('H1_old', '')

        # Build prompt
        prompt = f"""
I have an old URL that needs a redirect. Here is its context:

- Old URL: {old_url}
- Title: {old_title}
- Meta Description: {old_desc}
- H1: {old_h1}

I have these potential new URLs to choose from:
"""
        for candidate in limited_new_candidates:
            new_url = candidate[0]
            new_h1  = candidate[1]
            prompt += f"\n- {new_url} | H1: {new_h1}"

        prompt += """
Please pick **one** best matching new URL from the list above, based on
semantic relevance and content. Give me the URL only, not an explanation.
If none match, say "NONE".
"""

        try:
            # NEW SYNTAX for openai>=1.0.0
            response = openai.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are an assistant that helps match old and new URLs for SEO."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
            )
            # Also note the new way to grab the text:
            best_match_url = response.choices[0].message.content.strip()

            if best_match_url.upper() == "NONE":
                ai_matches.append({
                    'Address_old': old_url,
                    'Address_new': 'NOT FOUND (AI)',
                    'Match_Type': 'AI Match',
                    'Confidence_Score': 0,
                    'Title_old': old_title,
                    'Meta Description_old': old_desc,
                    'H1_old': old_h1
                })
            else:
                # Optionally verify the returned URL is in new_df
                candidate_check = new_df[new_df['Address_new'].str.lower() == best_match_url.lower()]
                if not candidate_check.empty:
                    ai_matches.append({
                        'Address_old': old_url,
                        'Address_new': best_match_url,
                        'Match_Type': 'AI Match',
                        'Confidence_Score': 70,
                        'Title_old': old_title,
                        'Meta Description_old': old_desc,
                        'H1_old': old_h1
                    })
                else:
                    ai_matches.append({
                        'Address_old': old_url,
                        'Address_new': best_match_url + " (AI Not Verified)",
                        'Match_Type': 'AI Match',
                        'Confidence_Score': 50,
                        'Title_old': old_title,
                        'Meta Description_old': old_desc,
                        'H1_old': old_h1
                    })

        except Exception as e:
            st.warning(f"OpenAI API error for URL: {old_url}\n{e}")
            ai_matches.append({
                'Address_old': old_url,
                'Address_new': 'NOT FOUND (API Error)',
                'Match_Type': 'AI Match',
                'Confidence_Score': 0,
                'Title_old': old_title,
                'Meta Description_old': old_desc,
                'H1_old': old_h1
            })

    return pd.DataFrame(ai_matches)
