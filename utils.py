# utils.py

import os
import zipfile
import io
import pandas as pd
from urllib.parse import urlparse
from sentence_transformers import SentenceTransformer
import numpy as np
import streamlit as st

@st.cache_resource
def load_model():
    """Load the SentenceTransformer model with Streamlit caching."""
    return SentenceTransformer('all-MiniLM-L6-v2')

def extract_path(url):
    """Extract path from a URL, stripping trailing slash."""
    try:
        return urlparse(url).path.rstrip('/')
    except:
        return ''

def read_csv_from_zip(z, file_name):
    """Reads a CSV file from an already opened ZipFile object."""
    with z.open(file_name) as f:
        return pd.read_csv(f)

def string_to_embedding(s):
    """
    Convert a string '[x1, x2, ...]' to a numpy array.
    If invalid or empty, returns an empty array.
    """
    if isinstance(s, str):
        arr = np.fromstring(s.strip('[]'), sep=',')
        return arr if arr.size > 0 else np.array([])
    return np.array([])
