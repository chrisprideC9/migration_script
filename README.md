# 📄 URL Matching Tool for Website Migration

A Streamlit-based tool to assist in matching URLs for website migration by using multiple matching strategies, including AI-based matching with OpenAI's GPT models.

## 🌟 Features

- **Multiple matching strategies:**
  - Path-based exact matching
  - Fuzzy path matching with configurable thresholds
  - Vector similarity matching using sentence embeddings
  - H1 header matching for content-based comparison
  - AI-based GPT matching as a final fallback

- **Integration with SEO data:**
  - Cross-references data from Ahrefs and optionally Google Search Console
  - Prioritizes important URLs based on traffic, keywords, and impressions (if available)
  - Generates priority scores to focus on high-value redirects

- **User-friendly interface:**
  - Step-by-step workflow with clear instructions
  - Interactive dashboards and visualizations
  - Downloadable CSV reports for matched and unmatched URLs

## 🔧 Installation

1. Clone this repository:
   ```
   git clone https://github.com/chrisprideC9/url-matching-tool.git
   cd url-matching-tool
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the root directory with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

## 📋 Required Data Files

The tool requires the following CSV files (compressed in a single ZIP file):

1. **top_pages.csv** - Ahrefs data with columns:
   - URL
   - Current traffic
   - Current traffic value
   - Current # of keywords
   - Current top keyword
   - Current top keyword: Country
   - Current top keyword: Volume
   - Current top keyword: Position

2. **old_vectors.csv** - Data for the old website URLs with columns:
   - Address
   - Status Code
   - Title 1
   - Meta Description 1
   - H1-1
   - embeds 1 (vector embeddings)

3. **new_vectors.csv** - Data for the new website URLs with columns:
   - Address
   - H1-1
   - embeds 1 (vector embeddings)

4. **impressions.csv** (Optional) - Google Search Console data
   - If not included, the tool will work without GSC data, adjusting its priority scoring algorithm accordingly

## 🚀 Usage

1. Start the Streamlit app:
   ```
   streamlit run main.py
   ```

2. Upload a ZIP file containing all required CSV files (impressions.csv is optional).

3. Follow the step-by-step workflow in the application:
   - Processing data files
   - Filtering URLs
   - Extracting paths
   - Performing URL matching using various strategies
   - Setting similarity thresholds for fuzzy and H1 matching
   - Running AI-based matching for unmatched URLs
   - Cross-checking with Ahrefs and Search Console data (if available)
   - Reviewing and downloading results

## 🧠 Matching Process

The tool performs matching in the following order:

1. **Exact Path Matching** - Finds URLs with identical paths
2. **Fuzzy Path Matching** - Finds URLs with similar paths using string similarity
3. **Vector Similarity** - Uses sentence embeddings to find semantically similar URLs
4. **H1-Based Matching** - Compares H1 headers for remaining unmatched URLs
5. **AI-Based Matching** - Uses OpenAI's GPT models to suggest matches for difficult cases

## 📊 Results and Reports

The tool generates three downloadable CSV reports:

1. **Matched Redirects (Most Confident)** - The most confident match for each old URL
2. **All Matched URLs** - All potential matches for old URLs
3. **Unmatched URLs** - URLs that couldn't be matched with any strategy

Each URL is assigned a:
- **Confidence Score** - How reliable the match is
- **Priority Score** - Based on traffic, keywords, and impressions (if GSC data is available)

## ⚙️ Advanced Configuration

You can customize the tool by:

- Adjusting matching similarity thresholds
- Changing the OpenAI model used for AI matching 
- Modifying the priority score calculation weights

## 📝 Project Structure

- `main.py` - The main Streamlit application
- `ai_matching.py` - Implementation of AI-based matching using OpenAI
- `matching_strategies.py` - Implementation of exact, fuzzy, and vector matching
- `utils.py` - Utility functions for data processing and embedding
- `.env` - Environment variables (not checked into version control)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License.

## 👨‍💻 Author

Created by Chris Pride at Calibre Nine.