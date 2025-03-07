# main.py

import streamlit as st
import pandas as pd
import numpy as np
import zipfile
import openai
import os

from dotenv import load_dotenv
from matching_strategies import (
    perform_exact_matching,
    perform_fuzzy_matching,
    perform_vector_matching,
)
from ai_matching import perform_ai_matching
from utils import (
    load_model,
    extract_path,
    read_csv_from_zip,
    string_to_embedding,
)

# Load environment variables (e.g., OPENAI_API_KEY from .env)
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize the SentenceTransformer model
model = load_model()

st.title("📄 URL Matching Tool for Website Migration")

st.markdown("""
This tool assists in matching URLs for website migration by comparing old and new URLs using:
- Path-based matching
- Fuzzy matching
- Vector similarity
- H1 header matching
- **AI-based GPT matching** (final fallback)

It also cross-references data from Ahrefs and optionally Google Search Console to prioritize important URLs for redirects.
""")

# Step 1: File Upload
st.header("1. Upload a Zipped File Containing All Required CSVs")

uploaded_zip = st.file_uploader(
    "Upload a ZIP file containing top_pages.csv, old_vectors.csv, new_vectors.csv (impressions.csv is optional)",
    type=["zip"]
)

if uploaded_zip:
    with st.spinner("🔄 Processing the uploaded ZIP file..."):
        try:
            with zipfile.ZipFile(uploaded_zip) as z:
                required_files = ['top_pages.csv', 'old_vectors.csv', 'new_vectors.csv']
                optional_files = ['impressions.csv']
                zip_file_names = z.namelist()
                
                # Check for missing required files
                missing_files = [file for file in required_files if file not in zip_file_names]
                if missing_files:
                    st.error(f"❌ The following required CSV files are missing in the ZIP: {missing_files}")
                    st.stop()
                
                # Read required CSVs
                top_pages = read_csv_from_zip(z, 'top_pages.csv')
                old_vectors = read_csv_from_zip(z, 'old_vectors.csv')
                new_vectors = read_csv_from_zip(z, 'new_vectors.csv')
                
                # Check for and read optional files
                has_impressions = 'impressions.csv' in zip_file_names
                if has_impressions:
                    impressions = read_csv_from_zip(z, 'impressions.csv')
                    st.info("✅ impressions.csv was found and loaded.")
                else:
                    impressions = pd.DataFrame(columns=['Top pages', 'Clicks', 'Impressions'])
                    st.info("ℹ️ impressions.csv was not found. GSC data will not be included in priority scoring.")
            
            st.success("✅ All required CSV files have been successfully extracted and loaded.")
        except zipfile.BadZipFile:
            st.error("❌ The uploaded file is not a valid ZIP archive.")
            st.stop()
        except Exception as e:
            st.error(f"❌ An error occurred while processing the ZIP file: {e}")
            st.stop()
    
    # Step 2: Data Processing
    st.header("2. Processing Extracted Data")
    
    try:
        # Display columns of top_pages.csv
        st.write("### Columns in top_pages.csv")
        st.write(top_pages.columns.tolist())
        
        required_columns = [
            'URL', 'Current traffic', 'Current traffic value', 'Current # of keywords',
            'Current top keyword', 'Current top keyword: Country',
            'Current top keyword: Volume', 'Current top keyword: Position'
        ]
        
        missing_columns_top_pages = [col for col in required_columns if col not in top_pages.columns]
        if missing_columns_top_pages:
            st.error(f"❌ The following required columns are missing in top_pages.csv: {missing_columns_top_pages}")
            st.stop()
        
        # Select and rename columns for top_pages
        top_pages = top_pages[required_columns]
        top_pages = top_pages.rename(columns={
            'Current traffic': 'Traffic',
            'Current traffic value': 'Traffic value',
            'Current # of keywords': 'Keywords',
            'Current top keyword': 'Top keyword',
            'Current top keyword: Country': 'Top keyword: Country',
            'Current top keyword: Volume': 'Top keyword: Volume',
            'Current top keyword: Position': 'Top keyword: Position'
        })
        
        st.success("✅ top_pages.csv processed successfully.")
    except Exception as e:
        st.error(f"❌ Error processing top_pages.csv: {e}")
        st.stop()
    
    try:
        # Process old_vectors
        expected_columns_old = [
            'Address', 'Status Code', 'Title 1', 'Meta Description 1', 'H1-1', 'embeds 1'
        ]
        missing_old = [col for col in expected_columns_old if col not in old_vectors.columns]
        if missing_old:
            st.error(f"❌ Missing columns in old_vectors.csv: {missing_old}")
            st.stop()
        
        old_vectors = old_vectors[expected_columns_old].rename(columns={
            'Address': 'Address_old',
            'Title 1': 'Title',
            'Meta Description 1': 'Meta Description',
            'H1-1': 'H1'
        })
        
        # Keep only status code 200
        old_vectors = old_vectors[old_vectors['Status Code'] == 200].reset_index(drop=True)

        # Convert embeddings
        old_vectors['Embedding'] = old_vectors['embeds 1'].apply(string_to_embedding)
        old_vectors.drop(columns=['embeds 1'], inplace=True)
        
        # Remove rows with empty embeddings
        initial_count = old_vectors.shape[0]
        old_vectors = old_vectors[old_vectors['Embedding'].apply(lambda x: len(x) > 0)].reset_index(drop=True)
        removed_count = initial_count - old_vectors.shape[0]
        if removed_count > 0:
            st.warning(f"⚠️ Removed **{removed_count}** rows from old_vectors.csv due to empty or invalid embeddings.")
        
        st.success("✅ old_vectors.csv processed successfully.")
        
        # Process new_vectors
        expected_columns_new = ['Address', 'H1-1', 'embeds 1']
        missing_new = [col for col in expected_columns_new if col not in new_vectors.columns]
        if missing_new:
            st.error(f"❌ Missing columns in new_vectors.csv: {missing_new}")
            st.stop()
        
        new_vectors = new_vectors[expected_columns_new].rename(columns={
            'Address': 'Address_new',
            'H1-1': 'H1'
        })
        
        new_vectors['Embedding'] = new_vectors['embeds 1'].apply(string_to_embedding)
        new_vectors.drop(columns=['embeds 1'], inplace=True)

        # Remove rows with empty embeddings
        initial_count = new_vectors.shape[0]
        new_vectors = new_vectors[new_vectors['Embedding'].apply(lambda x: len(x) > 0)].reset_index(drop=True)
        removed_count = initial_count - new_vectors.shape[0]
        if removed_count > 0:
            st.warning(f"⚠️ Removed **{removed_count}** rows from new_vectors.csv due to empty or invalid embeddings.")
        
        st.success("✅ new_vectors.csv processed successfully.")
    except Exception as e:
        st.error(f"❌ Error processing vector CSVs: {e}")
        st.stop()
    
    # Step 3: Filter URLs with Status Code 200
    st.header("3. Filtering URLs with Status Code 200")
    
    try:
        st.write(f"📄 **Old URLs with Status Code 200:** {old_vectors.shape[0]}")
        st.write(f"🌐 **Total New URLs:** {new_vectors.shape[0]}")
        
        if old_vectors.empty:
            st.error("❌ After filtering, old_vectors.csv has no URLs with Status Code 200. Please check your data.")
            st.stop()
        else:
            st.success("✅ Successfully filtered old URLs with Status Code 200.")
    except Exception as e:
        st.error(f"❌ Error during status code filtering: {e}")
        st.stop()
    
    # Step 4: Extract Paths from URLs
    st.header("4. Extracting URL Paths")
    try:
        old_vectors['Path'] = old_vectors['Address_old'].apply(extract_path)
        new_vectors['Path'] = new_vectors['Address_new'].apply(extract_path)
        st.success("✅ Extracted URL paths successfully.")
    except Exception as e:
        st.error(f"❌ Error extracting URL paths: {e}")
        st.stop()
    
    # Step 5: URL Matching (Exact/Fuzzy/Vector)
    st.header("5. Matching URLs")
    
    regular_old_vectors = old_vectors.reset_index(drop=True)
    st.write(f"🟢 **Regular URLs (200):** {regular_old_vectors.shape[0]}")

    matches = []  # will store DataFrames of matches

    # Exact Path Matching
    exact_regular = perform_exact_matching(regular_old_vectors, new_vectors, status='Regular')
    if not exact_regular.empty:
        matches.append(exact_regular)
    
    # Fuzzy Path Matching
    matching_similarity_threshold = st.slider(
        "Set Matching Similarity Threshold (%)",
        80, 100, 90
    )
    fuzzy_regular = perform_fuzzy_matching(
        regular_old_vectors, new_vectors, 
        threshold=matching_similarity_threshold, 
        status='Regular'
    )
    if not fuzzy_regular.empty:
        matches.append(fuzzy_regular)
    
    # Vector-Based Matching
    vector_regular = perform_vector_matching(regular_old_vectors, new_vectors, status='Regular')
    if not vector_regular.empty:
        matches.append(vector_regular)
    
    # Combine All Matches
    if len(matches) > 0:
        matches_df = pd.concat(matches, ignore_index=True)
    else:
        matches_df = pd.DataFrame()

    # Step 6: Handling Unmatched URLs
    st.header("6. Handling Unmatched URLs")
    
    if not regular_old_vectors.empty:
        all_old_urls = set(regular_old_vectors['Address_old'])
        matched_old_urls = set(matches_df['Address_old']) if not matches_df.empty else set()
        
        unmatched_old_urls = all_old_urls - matched_old_urls
        if unmatched_old_urls:
            st.write(f"🔍 **{len(unmatched_old_urls)}** Old URLs did not find a match in the new site.")
            
            unmatched_df = regular_old_vectors[regular_old_vectors['Address_old'].isin(unmatched_old_urls)].copy()
            unmatched_df = unmatched_df[['Address_old','Title','Meta Description','H1']].copy()
            unmatched_df['Address_new'] = 'NOT FOUND'
            unmatched_df['Match_Type'] = 'No Match'
            unmatched_df['Confidence_Score'] = 0
            unmatched_df = unmatched_df.rename(columns={
                'Title': 'Title_old',
                'Meta Description': 'Meta Description_old',
                'H1': 'H1_old'
            })
            
            unmatched_df = unmatched_df[[
                'Address_old','Address_new','Match_Type','Confidence_Score',
                'Title_old','Meta Description_old','H1_old'
            ]]
            
            st.info("ℹ️ These URLs have been marked as NOT FOUND and may require manual review or redirects.")
        else:
            unmatched_df = pd.DataFrame()
            st.success("✅ All old URLs have been successfully matched.")
    else:
        unmatched_df = pd.DataFrame()
        st.info("ℹ️ No old URLs to process.")

    # Step 7: H1-Based Matching for Remaining Unmatched URLs
    st.header("7. H1-Based Matching for Remaining Unmatched URLs")
    
    if not unmatched_df.empty:
        with st.spinner("🔄 Performing H1-based matching for unmatched URLs..."):
            try:
                available_new_vectors = new_vectors.copy()
                # keep new URLs with valid H1
                available_new_vectors = available_new_vectors[
                    available_new_vectors['H1'].notnull() & (available_new_vectors['H1'].str.strip() != '')
                ].reset_index(drop=True)
                
                unmatched_old_vectors = unmatched_df.copy()
                unmatched_old_vectors = unmatched_old_vectors[
                    unmatched_old_vectors['H1_old'].notnull() & (unmatched_old_vectors['H1_old'].str.strip() != '')
                ].reset_index(drop=True)
                
                if unmatched_old_vectors.empty:
                    st.info("ℹ️ No unmatched old URLs with valid H1 for H1-based matching.")
                elif available_new_vectors.empty:
                    st.info("ℹ️ No new URLs with valid H1 for H1-based matching.")
                else:
                    # inline function for H1 matching
                    def perform_h1_matching(old_df, new_df, threshold=90):
                        import pandas as pd
                        from rapidfuzz import process, fuzz
                        
                        h1_matches = []
                        
                        for _, old_row in old_df.iterrows():
                            old_h1 = old_row['H1_old']
                            # Exact H1 match
                            exact_matches = new_df[new_df['H1'] == old_h1]
                            if not exact_matches.empty:
                                new_row = exact_matches.iloc[0]
                                h1_matches.append({
                                    'Address_old': old_row['Address_old'],
                                    'Address_new': new_row['Address_new'],
                                    'Match_Type': 'Exact H1',
                                    'Confidence_Score': 80,
                                    'Title_old': old_row['Title_old'],
                                    'Meta Description_old': old_row['Meta Description_old'],
                                    'H1_old': old_h1
                                })
                            else:
                                # fuzzy H1
                                match, score, _ = process.extractOne(
                                    old_h1, new_df['H1'].tolist(), scorer=fuzz.token_sort_ratio
                                )
                                if score >= threshold:
                                    new_row = new_df[new_df['H1'] == match].iloc[0]
                                    h1_matches.append({
                                        'Address_old': old_row['Address_old'],
                                        'Address_new': new_row['Address_new'],
                                        'Match_Type': 'Similar H1',
                                        'Confidence_Score': 60,
                                        'Title_old': old_row['Title_old'],
                                        'Meta Description_old': old_row['Meta Description_old'],
                                        'H1_old': old_h1
                                    })
                        if h1_matches:
                            return pd.DataFrame(h1_matches)
                        else:
                            return pd.DataFrame()
                    
                    h1_similarity_threshold = st.slider(
                        "Set H1 Matching Similarity Threshold (%)",
                        70, 100, 90,
                        key='h1_threshold'
                    )
                    
                    h1_matches_df = perform_h1_matching(
                        unmatched_old_vectors, available_new_vectors, 
                        threshold=h1_similarity_threshold
                    )
                    if not h1_matches_df.empty:
                        st.write(
                            f"🔍 Found **{h1_matches_df.shape[0]}** H1-based matches with threshold ≥ {h1_similarity_threshold}%."
                        )
                        h1_matches_df = h1_matches_df[[
                            'Address_old','Address_new','Match_Type','Confidence_Score',
                            'Title_old','Meta Description_old','H1_old'
                        ]]
                        
                        # Append to matches
                        if not matches_df.empty:
                            matches_df = pd.concat([matches_df, h1_matches_df], ignore_index=True)
                        else:
                            matches_df = h1_matches_df.copy()
                        
                        # Remove H1-matched old URLs from unmatched_df
                        matched_old_urls_h1 = set(h1_matches_df['Address_old'])
                        unmatched_df = unmatched_df[~unmatched_df['Address_old'].isin(matched_old_urls_h1)].reset_index(drop=True)
                        
                        st.success("✅ H1-based matching completed successfully.")
                    else:
                        st.info("ℹ️ No H1-based matches found.")
            except Exception as e:
                st.error(f"❌ Error during H1-based matching: {e}")
                st.stop()
    else:
        st.info("ℹ️ No unmatched URLs available for H1-based matching.")

    # Step 8: AI-Based Matching for Remaining Unmatched URLs (final fallback)
    st.header("8. AI-Based Matching for Remaining Unmatched URLs")
    if not unmatched_df.empty:
        st.write("Use OpenAI's GPT model to suggest matches for any URLs still unmatched.")
        # Instead of a checkbox, we use a button:
        if st.button("Start AI-Based Matching"):
            with st.spinner("⏳ Doing AI shit"):
                ai_matches_df = perform_ai_matching(unmatched_df, new_vectors, model="gpt-3.5-turbo")
                
                if not ai_matches_df.empty:
                    st.success(f"AI-based matching found {ai_matches_df.shape[0]} potential matches.")
                    
                    # Merge AI results with main matches
                    if not matches_df.empty:
                        matches_df = pd.concat([matches_df, ai_matches_df], ignore_index=True)
                    else:
                        matches_df = ai_matches_df.copy()
                    
                    # Remove AI-matched old URLs from unmatched_df
                    matched_from_ai = set(ai_matches_df['Address_old'])
                    unmatched_df = unmatched_df[~unmatched_df['Address_old'].isin(matched_from_ai)].reset_index(drop=True)
                else:
                    st.warning("No AI-based matches found.")
    else:
        st.info("ℹ️ No unmatched URLs remaining for AI-based matching.")

    # Step 9: Cross-Checking with Ahrefs and Search Console
    st.header("9. Cross-Checking with Ahrefs and Search Console Data")
    
    if not matches_df.empty or not unmatched_df.empty:
        with st.spinner("🔗 Merging with Ahrefs and Search Console data..."):
            try:
                # Standardize URL formats
                if not matches_df.empty:
                    matches_df['Address_old'] = matches_df['Address_old'].str.lower().str.rstrip('/')
                    matches_df['Address_new'] = matches_df['Address_new'].str.lower().str.rstrip('/')
                if not unmatched_df.empty:
                    unmatched_df['Address_old'] = unmatched_df['Address_old'].str.lower().str.rstrip('/')
                    unmatched_df['Address_new'] = unmatched_df['Address_new'].str.lower().str.rstrip('/')
                
                top_pages['URL'] = top_pages['URL'].str.lower().str.rstrip('/')
                
                # Handle impressions data
                if has_impressions:
                    impressions['Top pages'] = impressions['Top pages'].str.lower().str.rstrip('/')
                
                # Merge old URLs with top_pages
                if not matches_df.empty:
                    matches_df = matches_df.merge(top_pages, left_on='Address_old', right_on='URL', how='left')
                    # Merge new URLs
                    matches_df = matches_df.merge(top_pages, left_on='Address_new', right_on='URL', how='left', suffixes=('', '_new'))
                    
                    # Merge GSC impressions if available
                    if has_impressions:
                        matches_df = matches_df.merge(impressions, left_on='Address_old', right_on='Top pages', how='left', suffixes=('', '_impressions'))
                    else:
                        # Add empty columns for GSC data
                        matches_df['Clicks'] = 0
                        matches_df['Impressions'] = 0
                    
                    # Handle numeric data
                    numeric_columns = ['Traffic','Traffic value','Keywords','Clicks','Impressions']
                    for col in numeric_columns:
                        if col in matches_df.columns:
                            matches_df[col] = pd.to_numeric(matches_df[col], errors='coerce').fillna(0)
                        else:
                            matches_df[col] = 0
                    
                    # Calculate priority score based on available data
                    if has_impressions:
                        matches_df['Priority_Score'] = (
                            matches_df['Traffic'] * 0.4 +
                            matches_df['Traffic value'] * 0.3 +
                            matches_df['Keywords'] * 0.2 +
                            matches_df['Impressions'] * 0.1
                        )
                    else:
                        # Adjust weights if no impression data
                        matches_df['Priority_Score'] = (
                            matches_df['Traffic'] * 0.5 +
                            matches_df['Traffic value'] * 0.3 +
                            matches_df['Keywords'] * 0.2
                        )
                    matches_df = matches_df.sort_values(by=['Priority_Score'], ascending=False).reset_index(drop=True)
                
                if not unmatched_df.empty:
                    unmatched_df = unmatched_df.merge(top_pages, left_on='Address_old', right_on='URL', how='left')
                    unmatched_df = unmatched_df.merge(top_pages, left_on='Address_new', right_on='URL', how='left', suffixes=('', '_new'))
                    
                    if has_impressions:
                        unmatched_df = unmatched_df.merge(impressions, left_on='Address_old', right_on='Top pages', how='left', suffixes=('', '_impressions'))
                    else:
                        unmatched_df['Clicks'] = 0
                        unmatched_df['Impressions'] = 0
                    
                    numeric_columns = ['Traffic','Traffic value','Keywords','Clicks','Impressions']
                    for col in numeric_columns:
                        if col in unmatched_df.columns:
                            unmatched_df[col] = pd.to_numeric(unmatched_df[col], errors='coerce').fillna(0)
                        else:
                            unmatched_df[col] = 0
                    
                    if has_impressions:
                        unmatched_df['Priority_Score'] = (
                            unmatched_df['Traffic'] * 0.4 +
                            unmatched_df['Traffic value'] * 0.3 +
                            unmatched_df['Keywords'] * 0.2 +
                            unmatched_df['Impressions'] * 0.1
                        )
                    else:
                        unmatched_df['Priority_Score'] = (
                            unmatched_df['Traffic'] * 0.5 +
                            unmatched_df['Traffic value'] * 0.3 +
                            unmatched_df['Keywords'] * 0.2
                        )
                    unmatched_df = unmatched_df.sort_values(by=['Priority_Score'], ascending=False).reset_index(drop=True)
                
                if has_impressions:
                    st.success("✅ Cross-checking with Ahrefs and GSC data completed.")
                else:
                    st.success("✅ Cross-checking with Ahrefs data completed. No GSC data was used.")
            except Exception as e:
                st.error(f"❌ Error during cross-checking: {e}")
                st.stop()
        
        # Step 10: Displaying Results
        st.header("10. Matched and Unmatched URLs for Redirects")
        
        st.write("### 📝 Matched URLs with Priority Scores and Confidence Levels")
        
        matched_display_cols = [
            'Address_old','Address_new','Match_Type','Confidence_Score','Title_old','Meta Description_old','H1_old',
            'Traffic','Traffic value','Keywords'
        ]
        
        # Add impressions columns if they exist
        if has_impressions:
            matched_display_cols.extend(['Clicks', 'Impressions'])
        
        matched_display_cols.append('Priority_Score')
        
        if not matches_df.empty:
            display_matches_df = matches_df[matched_display_cols].fillna('N/A')
        else:
            display_matches_df = pd.DataFrame()
        
        unmatched_display_cols = matched_display_cols.copy()  # Use the same columns as matched
        
        if not unmatched_df.empty:
            display_unmatched_df = unmatched_df[unmatched_display_cols].fillna('N/A')
        else:
            display_unmatched_df = pd.DataFrame()
        
        with st.expander("🔍 View Matched URLs (Status Code 200)"):
            if not display_matches_df.empty:
                show_most_confident_regular = st.checkbox("🔍 Show only the most confident match per Old URL", key='regular_confidence')
                if show_most_confident_regular:
                    # Highest Confidence_Score per old URL
                    regular_filtered = display_matches_df.loc[
                        display_matches_df.groupby('Address_old')['Confidence_Score'].idxmax()
                    ].reset_index(drop=True)
                    st.write("### 📈 Only the most confident match for each old URL.")
                else:
                    regular_filtered = display_matches_df
                    st.write("### 📊 Displaying all matches for each old URL.")
                
                st.dataframe(
                    regular_filtered.style
                        .format({'Confidence_Score': "{:.0f}", 'Priority_Score': "{:.2f}"})
                        .set_properties(**{'text-align': 'left'})
                )
            else:
                st.info("ℹ️ No Matched URLs to display.")
        
        with st.expander("🔍 View Unmatched URLs (Status Code 200)"):
            if not display_unmatched_df.empty:
                st.write("### 🛑 URLs Marked as NOT FOUND")
                st.dataframe(
                    display_unmatched_df.style
                        .format({'Confidence_Score': "{:.0f}", 'Priority_Score': "{:.2f}"})
                        .set_properties(**{'text-align': 'left'})
                )
            else:
                st.info("ℹ️ No Unmatched URLs to display.")
        
        # Download Buttons
        def convert_df_to_csv(df):
            return df.to_csv(index=False).encode('utf-8')
        
        # Most confident matches
        if not display_matches_df.empty:
            most_confident_df = display_matches_df.loc[
                display_matches_df.groupby('Address_old')['Confidence_Score'].idxmax()
            ].reset_index(drop=True)
        else:
            most_confident_df = pd.DataFrame()
        
        all_matches_df = display_matches_df.copy()
        
        unmatched_csv = convert_df_to_csv(display_unmatched_df) if not display_unmatched_df.empty else None
        csv_redirect = convert_df_to_csv(most_confident_df) if not most_confident_df.empty else None
        csv_all = convert_df_to_csv(all_matches_df) if not all_matches_df.empty else None
        
        col1, col2, col3 = st.columns(3)
        with col1:
            if csv_redirect:
                st.download_button(
                    label="📥 Download Matched Redirects (Most Confident) as CSV",
                    data=csv_redirect,
                    file_name='matched_redirect_download.csv',
                    mime='text/csv',
                )
            else:
                st.info("ℹ️ No data available for Matched Redirects Download.")
        
        with col2:
            if csv_all:
                st.download_button(
                    label="📥 Download All Matched URLs as CSV",
                    data=csv_all,
                    file_name='all_matched_urls.csv',
                    mime='text/csv',
                )
            else:
                st.info("ℹ️ No data available for All Matches Download.")
        
        with col3:
            if unmatched_csv:
                st.download_button(
                    label="📥 Download Unmatched URLs as CSV",
                    data=unmatched_csv,
                    file_name='unmatched_urls.csv',
                    mime='text/csv',
                )
            else:
                st.info("ℹ️ No data available for Unmatched URLs Download.")
        
        st.success("🎉 URL matching process completed successfully!")
else:
    st.info("🛠️ Please upload a ZIP file containing the required CSV files to begin the URL matching process.")

st.markdown("---")
st.markdown("© 2025 Calibre Nine | [GitHub Repository](https://github.com/chrisprideC9)")