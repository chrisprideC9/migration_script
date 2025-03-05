import streamlit as st
import pandas as pd
import os
import openai

def suggest_redirect( url_404, site_urls, model="gpt-3.5-turbo" ):
    """
    Sends a single ChatCompletion request to GPT with:
    - A system message about what the assistant does
    - A user message that includes the 404 URL and the entire site URL list
    Returns GPT's chosen redirect as a string.
    """
    system_prompt = (
        "You are an expert at mapping 404 URLs to the best available redirect. "
        "You will be given a 404 URL and a list of possible site URLs. "
        "Your job is to pick the SINGLE site URL from the list that best matches "
        "the intent or content of the 404 URL."
    )
    
    # The user prompt includes the 404 URL and the entire site list
    user_prompt = (
        f"404 URL: {url_404}\n\n"
        f"Here is the list of valid site URLs:\n"
        + "\n".join(site_urls) +
        "\n\nPlease return ONLY the single best matching URL from the above list. "
        "No extra explanation."
    )
    
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0  # lower temperature for more deterministic output
        )
        best_redirect = response["choices"][0]["message"]["content"].strip()
        return best_redirect
    except Exception as e:
        st.warning(f"OpenAI API error for 404 URL '{url_404}': {str(e)}")
        return ""

def main():
    st.title("404 Redirect Suggester (Context-Only via GPT)")
    st.write("""
        This app uses OpenAI's ChatCompletion endpoint (**no embeddings**) 
        to find a suitable redirect for each 404 URL by providing GPT 
        with the entire list of site URLs each time.
        
        **Warning**: This can be very token-intensive if you have a large site URL list. 
        For smaller lists, it's fine.
    """)

    # Make sure your API key is in the environment
    if "OPENAI_API_KEY" not in os.environ:
        st.error("No OPENAI_API_KEY found in environment. Please set it before running.")
        st.stop()
    
    openai.api_key = os.environ["OPENAI_API_KEY"]
    
    # File Uploaders
    file_404 = st.file_uploader("Upload CSV with 404 URLs (one column)", type=["csv"])
    file_site = st.file_uploader("Upload CSV with Site URLs (one column)", type=["csv"])
    
    if file_404 and file_site:
        df_404 = pd.read_csv(file_404)
        df_site = pd.read_csv(file_site)
        
        # Basic validation
        if df_404.shape[1] < 1 or df_site.shape[1] < 1:
            st.error("Each CSV must have at least one column of URLs.")
            return
        
        st.write("**404 URLs (preview):**")
        st.dataframe(df_404.head())
        st.write("**Site URLs (preview):**")
        st.dataframe(df_site.head())
        
        model = st.selectbox("Choose model", ["gpt-3.5-turbo", "gpt-4"])
        
        if st.button("Find Redirects"):
            site_urls = df_site.iloc[:,0].dropna().tolist()
            results = []
            
            with st.spinner("Asking GPT for each 404 URL..."):
                for _, row in df_404.iterrows():
                    url_404 = str(row.iloc[0])
                    best_redirect = suggest_redirect(url_404, site_urls, model=model)
                    results.append({
                        "404_URL": url_404,
                        "Suggested_Redirect": best_redirect
                    })
            
            results_df = pd.DataFrame(results)
            st.success("Done! Here are the suggestions:")
            st.dataframe(results_df)
            
            # Download button
            csv_data = results_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv_data,
                file_name="redirect_suggestions.csv",
                mime="text/csv"
            )

if __name__ == "__main__":
    main()
