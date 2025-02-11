# app.py

import streamlit as st
import pandas as pd
import numpy as np
import os
import io
import time
from datetime import datetime
import base64
import re
import pickle
import concurrent.futures  # Import for parallel processing
from typing import List
import plotly.express as px

# Import necessary libraries for embeddings, clustering, and summarization
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from bertopic import BERTopic
from hdbscan import HDBSCAN
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# For summarization
from langchain.chains import LLMChain
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

device = 'cpu'

def init_nltk_resources():
    # try:
    #     nltk.data.find('corpora/stopwords')
    # except LookupError:
    #     nltk.download('stopwords')
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab')

init_nltk_resources()

st.set_page_config(page_title="SNAP", layout="wide")

st.sidebar.title("Data Selection")
dataset_option = st.sidebar.selectbox('Select Dataset', ('PRMS 2022+2023 QAed', 'Upload my dataset'))

@st.cache_data
def load_default_dataset(default_dataset_path):
    if os.path.exists(default_dataset_path):
        df = pd.read_excel(default_dataset_path)
        return df
    else:
        st.error("Default dataset not found. Please ensure the file exists in the 'input' directory.")
        return None

@st.cache_data
def load_uploaded_dataset(uploaded_file):
    df = pd.read_excel(uploaded_file)
    return df

def generate_embeddings(texts, model):
    with st.spinner('Calculating embeddings...'):
        embeddings = model.encode(texts, show_progress_bar=True, device=device)
    return embeddings

@st.cache_resource
def get_embedding_model():
    model = SentenceTransformer('all-MiniLM-L6-v2').to(device)
    return model

def load_or_compute_embeddings(df, using_default_dataset, uploaded_file_name=None, text_columns=None):
    if text_columns is None or len(text_columns) == 0:
        return None, None  # No columns chosen, cannot compute embeddings

    embeddings_dir = os.path.dirname(__file__)
    # Create a hash or unique key from selected columns for caching
    cols_key = "_".join(sorted(text_columns))
    if using_default_dataset:
        embeddings_file = os.path.join(embeddings_dir, f'PRMS_2022_2023_QAed_{cols_key}.pkl')
    else:
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = os.path.splitext(uploaded_file_name)[0] if uploaded_file_name else "custom_dataset"
        embeddings_file = os.path.join(embeddings_dir, f"{base_name}_{cols_key}_{timestamp_str}.pkl")

    # Prepare texts by concatenating selected columns
    df_fill = df.fillna("")
    texts = df_fill[text_columns].astype(str).agg(' '.join, axis=1).tolist()

    # Check session_state cache
    if 'embeddings' in st.session_state and 'embeddings_file' in st.session_state and 'last_text_columns' in st.session_state:
        # If columns are the same and file name matches, reuse
        if st.session_state['last_text_columns'] == text_columns and len(st.session_state['embeddings']) == len(texts):
            return st.session_state['embeddings'], st.session_state['embeddings_file']

    # If embeddings file exists on disk
    if os.path.exists(embeddings_file):
        with open(embeddings_file, 'rb') as f:
            embeddings = pickle.load(f)
        if len(embeddings) == len(texts):
            st.write("Loading pre-calculated embeddings...")
            st.session_state['embeddings'] = embeddings
            st.session_state['embeddings_file'] = embeddings_file
            st.session_state['last_text_columns'] = text_columns
            return embeddings, embeddings_file
        else:
            st.write("Pre-calculated embeddings do not match current data. Regenerating...")

    # Compute embeddings
    st.write("Generating embeddings...")
    model = get_embedding_model()
    embeddings = generate_embeddings(texts, model)
    with open(embeddings_file, 'wb') as f:
        pickle.dump(embeddings, f)
    st.session_state['embeddings'] = embeddings
    st.session_state['embeddings_file'] = embeddings_file
    st.session_state['last_text_columns'] = text_columns
    return embeddings, embeddings_file

def reset_filters():
    st.session_state['selected_regions'] = []
    st.session_state['selected_countries'] = []
    st.session_state['selected_centers'] = []
    st.session_state['selected_impact_area'] = []
    st.session_state['selected_sdg_targets'] = []
    st.session_state['additional_filters'] = {}
    st.session_state['selected_additional_filters'] = {}

if dataset_option == 'PRMS 2022+2023 QAed':
    default_dataset_path = os.path.join(os.path.dirname(__file__), 'input', 'export_data_table_results_20240312_160222CET.xlsx')
    df = load_default_dataset(default_dataset_path)
    if df is not None:
        st.session_state['df'] = df.copy()
        st.session_state['using_default_dataset'] = True
        st.write("Using default dataset:")

        # Load filter options for standard filters if columns exist
        filters_dir = os.path.join(os.path.dirname(__file__), 'filters')

        df_cols = df.columns.tolist()

        # Safe check if columns exist
        def safe_read_filter_options(file_name, col_name):
            path = os.path.join(filters_dir, file_name)
            if os.path.exists(path) and col_name in df_cols:
                with open(path, 'r') as f:
                    opts = [line.strip() for line in f.readlines()]
                return opts
            else:
                return []

        regions_options_raw = safe_read_filter_options('regions.txt', 'Regions')
        countries_options_raw = safe_read_filter_options('countries.txt', 'Countries')
        centers_options_raw = safe_read_filter_options('centers.txt', 'Primary center')
        impact_area_options_raw = safe_read_filter_options('impact_area.txt', 'Impact Area Target')
        sdg_target_options_raw = safe_read_filter_options('sdg_target.txt', 'SDG targets')

        def get_counts(df, column, options):
            if column not in df.columns:
                return options, {o:o for o in options}
            counts = df[column].value_counts()
            mapped = []
            mapping_dict = {}
            for opt in options:
                c = counts[opt] if opt in counts else 0
                mapped.append(f"{opt} ({c})")
                mapping_dict[f"{opt} ({c})"] = opt
            return mapped, mapping_dict

        if 'selected_regions' not in st.session_state:
            st.session_state['selected_regions'] = []
        if 'selected_countries' not in st.session_state:
            st.session_state['selected_countries'] = []
        if 'selected_centers' not in st.session_state:
            st.session_state['selected_centers'] = []
        if 'selected_impact_area' not in st.session_state:
            st.session_state['selected_impact_area'] = []
        if 'selected_sdg_targets' not in st.session_state:
            st.session_state['selected_sdg_targets'] = []
        if 'selected_additional_filters' not in st.session_state:
            st.session_state['selected_additional_filters'] = {}

        # Maps for standard filters
        if regions_options_raw:
            regions_options_mapped, regions_map = get_counts(df, 'Regions', regions_options_raw)
        else:
            regions_options_mapped, regions_map = [], {}

        if countries_options_raw:
            countries_options_mapped, countries_map = get_counts(df, 'Countries', countries_options_raw)
        else:
            countries_options_mapped, countries_map = [], {}

        if centers_options_raw:
            centers_options_mapped, centers_map = get_counts(df, 'Primary center', centers_options_raw)
        else:
            centers_options_mapped, centers_map = [], {}

        if impact_area_options_raw:
            impact_area_options_mapped, impact_area_map = get_counts(df, 'Impact Area Target', impact_area_options_raw)
        else:
            impact_area_options_mapped, impact_area_map = [], {}

        if sdg_target_options_raw:
            sdg_target_options_mapped, sdg_target_map = get_counts(df, 'SDG targets', sdg_target_options_raw)
        else:
            sdg_target_options_mapped, sdg_target_map = [], {}

        with st.sidebar:
            st.write("**Filters**")
            if st.button("Reset Filters"):
                reset_filters()

        # Standard filters UI
        col1, col2 = st.columns(2)
        with col1:
            if regions_options_mapped:
                sel_reg = st.multiselect("Regions", regions_options_mapped, default=st.session_state['selected_regions'])
                st.session_state['selected_regions'] = sel_reg
            else:
                sel_reg = []

        with col2:
            if countries_options_mapped:
                sel_cnt = st.multiselect("Countries", countries_options_mapped, default=st.session_state['selected_countries'])
                st.session_state['selected_countries'] = sel_cnt
            else:
                sel_cnt = []

        col3, col4 = st.columns(2)
        with col3:
            if centers_options_mapped:
                sel_ctr = st.multiselect("Primary Center", centers_options_mapped, default=st.session_state['selected_centers'])
                st.session_state['selected_centers'] = sel_ctr
            else:
                sel_ctr = []

        with col4:
            if impact_area_options_mapped:
                sel_ia = st.multiselect("Impact Area Target(s)", impact_area_options_mapped, default=st.session_state['selected_impact_area'])
                st.session_state['selected_impact_area'] = sel_ia
            else:
                sel_ia = []

        col5 = st.columns(1)
        with col5[0]:
            if sdg_target_options_mapped:
                sel_sdg = st.multiselect("SDG Target(s)", sdg_target_options_mapped, default=st.session_state['selected_sdg_targets'])
                st.session_state['selected_sdg_targets'] = sel_sdg
            else:
                sel_sdg = []

        # Additional filter columns
        st.write("**Additional Filter Columns**")
        all_columns = df.columns.tolist()
        # Remove standard filter columns to avoid duplication
        standard_filter_cols = ['Regions', 'Countries', 'Primary center', 'Impact Area Target', 'SDG targets']
        # This is just a set to ensure no duplicates.
        add_filter_candidates = [c for c in all_columns if c not in standard_filter_cols]

        selected_additional_cols = st.multiselect("Select additional columns to filter by:", add_filter_candidates, default=st.session_state.get('additional_filters_selected', []))
        st.session_state['additional_filters_selected'] = selected_additional_cols

        # For each chosen additional filter column, show a multiselect of unique values
        for col_name in selected_additional_cols:
            # If not already in session state, initialize
            if f'selected_filter_{col_name}' not in st.session_state:
                st.session_state[f'selected_filter_{col_name}'] = []
            unique_vals = df[col_name].dropna().unique().tolist()
            # Sort for consistency
            unique_vals = sorted(unique_vals)
            selected_vals = st.multiselect(f"Filter by {col_name}", options=unique_vals, default=st.session_state[f'selected_filter_{col_name}'])
            st.session_state[f'selected_filter_{col_name}'] = selected_vals

        # Text columns selection
        st.write("**Select Text Columns for Embedding**")
        text_columns_selected = st.multiselect("Text Columns:", all_columns, default=['Title','Description'] if 'Title' in df.columns and 'Description' in df.columns else [])
        st.session_state['text_columns'] = text_columns_selected

        # Apply filters to create filtered_df
        filtered_df = df.copy()

        # Apply standard filters if columns exist
        if 'Regions' in df_cols and st.session_state['selected_regions']:
            reg_values = [regions_map[v] for v in st.session_state['selected_regions']]
            filtered_df = filtered_df[filtered_df['Regions'].isin(reg_values)]
        if 'Countries' in df_cols and st.session_state['selected_countries']:
            cnt_values = [countries_map[v] for v in st.session_state['selected_countries']]
            filtered_df = filtered_df[filtered_df['Countries'].isin(cnt_values)]
        if 'Primary center' in df_cols and st.session_state['selected_centers']:
            ctr_values = [centers_map[v] for v in st.session_state['selected_centers']]
            filtered_df = filtered_df[filtered_df['Primary center'].isin(ctr_values)]
        if 'Impact Area Target' in df_cols and st.session_state['selected_impact_area']:
            ia_values = [impact_area_map[v] for v in st.session_state['selected_impact_area']]
            filtered_df = filtered_df[filtered_df['Impact Area Target'].isin(ia_values)]
        if 'SDG targets' in df_cols and st.session_state['selected_sdg_targets']:
            sdg_values = [sdg_target_map[v] for v in st.session_state['selected_sdg_targets']]
            filtered_df = filtered_df[filtered_df['SDG targets'].isin(sdg_values)]

        # Apply additional filters
        for col_name in selected_additional_cols:
            selected_vals = st.session_state[f'selected_filter_{col_name}']
            if selected_vals:
                filtered_df = filtered_df[filtered_df[col_name].isin(selected_vals)]

        st.session_state['filtered_df'] = filtered_df

        st.write("Filtered Data Preview:")
        st.write(filtered_df.head())

        # Add total count of results
        st.write(f"Total number of results: {len(filtered_df)}")

        # Provide download button for filtered data
        output = io.BytesIO()
        writer = pd.ExcelWriter(output, engine='openpyxl')
        filtered_df.to_excel(writer, index=False)
        writer.close()
        processed_data = output.getvalue()

        st.download_button(
            label="Download Filtered Data",
            data=processed_data,
            file_name='filtered_data.xlsx',
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
    else:
        st.warning("Please ensure the default dataset exists in the 'input' directory.")
else:
    uploaded_file = st.sidebar.file_uploader("Upload your Excel file", type=["xlsx"])
    if uploaded_file is not None:
        df = load_uploaded_dataset(uploaded_file)
        if df is not None:
            st.session_state['df'] = df.copy()
            st.session_state['using_default_dataset'] = False
            st.session_state['uploaded_file_name'] = uploaded_file.name
            st.write("Data preview:")
            st.write(df.head())
            # No standard filters defined for user dataset, but user can pick text cols and additional filters
            df_cols = df.columns.tolist()

            # Additional filters logic for uploaded dataset
            st.write("**Select Text Columns for Embedding**")
            text_columns_selected = st.multiselect("Text Columns:", df_cols, default=df_cols[:1] if df_cols else [])
            st.session_state['text_columns'] = text_columns_selected

            st.write("**Select Additional Filter Columns**")
            selected_additional_cols = st.multiselect("Additional Filter Columns:", df_cols, default=[])
            st.session_state['additional_filters_selected'] = selected_additional_cols

            for col_name in selected_additional_cols:
                if f'selected_filter_{col_name}' not in st.session_state:
                    st.session_state[f'selected_filter_{col_name}'] = []
                unique_vals = df[col_name].dropna().unique().tolist()
                unique_vals = sorted(unique_vals)
                selected_vals = st.multiselect(f"Filter by {col_name}", options=unique_vals, default=st.session_state[f'selected_filter_{col_name}'])
                st.session_state[f'selected_filter_{col_name}'] = selected_vals

            filtered_df = df.copy()
            for col_name in selected_additional_cols:
                selected_vals = st.session_state[f'selected_filter_{col_name}']
                if selected_vals:
                    filtered_df = filtered_df[filtered_df[col_name].isin(selected_vals)]

            st.session_state['filtered_df'] = filtered_df
            st.write("Filtered Data Preview:")
            st.write(filtered_df.head())

            # Add total count and download button
            st.write(f"Total number of results: {len(filtered_df)}")

            output = io.BytesIO()
            writer = pd.ExcelWriter(output, engine='openpyxl')
            filtered_df.to_excel(writer, index=False)
            writer.close()
            processed_data = output.getvalue()

            st.download_button(
                label="Download Filtered Data",
                data=processed_data,
                file_name='filtered_data.xlsx',
                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            )
        else:
            st.warning("Failed to load the uploaded dataset.")
    else:
        st.warning("Please upload an Excel file to proceed.")

    st.write(f"Total number of results: {len(filtered_df)}")

# Create tabs (adding the new "Internal Validation" tab)
tab1, tab2, tab3, tab_help, tab_internal = st.tabs(["Semantic Search", "Clustering", "Summarization", "Help", "Internal Validation"])

# Help Tab
with tab_help:
    st.header("Help")
    st.markdown("""
    ### About SNAP

    SNAP allows you to explore, filter, search, cluster, and summarize textual datasets.
    Key features:
    - Load default or custom datasets.
    - Choose which columns serve as textual input for embeddings and semantic search.
    - Filter data using both predefined filters (in the default dataset) and custom columns in any dataset.
    - Perform semantic search with positive, negative, and required keywords.
    - Cluster documents using BERTopic.
    - Summarize the filtered or clustered sets of documents using OpenAI-based LLM.

    ### How to Use

    1. **Data Selection**: Use the sidebar to choose the default dataset or upload your own.
    2. **Filtering**:
       - For the default dataset, standard filters (Region, Country, etc.) are provided.
       - For all datasets, you can select which columns to use as additional filters.
       - After setting filters, the filtered data is shown below.
    3. **Select Text Columns**:
       - In both default and uploaded datasets, choose which columns contain text you want to use for embeddings.
       - If multiple columns are selected, they are concatenated into a single text per row.
    4. **Semantic Search**:
       - Go to the "Semantic Search" tab.
       - Enter your main query.
       - Optionally specify negative keywords (to exclude documents) and include keywords (documents must contain these).
       - Adjust the similarity threshold and click "Search".
       - Results are displayed in a single table.
    5. **Clustering**:
       - In the "Clustering" tab, choose whether to cluster the full filtered dataset or the semantic search results.
       - Adjust `min_cluster_size` as desired.
       - Run clustering to get topics and visualizations.
    6. **Summarization**:
       - If you have clusters, you can select which clusters to summarize.
       - Choose temperature and max_tokens for the LLM.
       - Generate summaries for the entire selection and per cluster.

    ### Troubleshooting

    - If no embeddings are computed, ensure you've selected at least one text column.
    - If no results appear, try adjusting filters or threshold, or removing negative/required keywords.
    - Check the console for errors if something goes wrong during clustering or summarization.

    ### Additional Tips

    - Experiment with different text columns to see how it affects semantic search and clustering.
    - Use additional filters to narrow down large datasets.
    - Summarization tokens and temperature control how creative or concise the summary is.
    """)

# Semantic Search Tab
with tab1:
    st.header("Semantic Search")
    if 'filtered_df' in st.session_state and st.session_state['filtered_df'] is not None:
        if not st.session_state['filtered_df'].empty:
            df = st.session_state['df']  # full dataset
            filtered_df = st.session_state['filtered_df']

            text_columns = st.session_state.get('text_columns', [])
            if not text_columns:
                st.warning("No text columns selected. Please select at least one column for text embedding in the main view.")
            else:
                # Compute embeddings if not done
                if ('embeddings' not in st.session_state) or (st.session_state.get('last_text_columns') != text_columns):
                    embeddings, embeddings_file = load_or_compute_embeddings(df, st.session_state.get('using_default_dataset', False), st.session_state.get('uploaded_file_name', None), text_columns)
                else:
                    embeddings = st.session_state['embeddings']

                if embeddings is not None:
                    query = st.text_input("Enter your search query:")
                    negative_keywords = st.text_input("Exclude documents containing these words (comma-separated):")
                    include_keywords = st.text_input("Include only documents containing these words (comma-separated):")
                    similarity_threshold = st.slider("Similarity threshold", 0.0, 1.0, 0.35)

                    if st.button("Search"):
                        if query.strip():
                            with st.spinner("Performing Semantic Search..."):
                                model = get_embedding_model()
                                # Prepare texts from filtered_df based on chosen text columns
                                df_fill = filtered_df.fillna("")
                                search_texts = df_fill[text_columns].agg(' '.join, axis=1).tolist()
                                query_embedding = model.encode([query], device=device)
                                filtered_indices = filtered_df.index
                                filtered_embeddings = embeddings[filtered_indices]
                                similarities = cosine_similarity(query_embedding, filtered_embeddings)
                                above_threshold_indices = np.where(similarities[0] > similarity_threshold)[0]

                                if len(above_threshold_indices) == 0:
                                    st.warning("No results found above the similarity threshold.")
                                    # Clear previous results if any
                                    if 'search_results' in st.session_state:
                                        del st.session_state['search_results']
                                    if 'search_results_processed_data' in st.session_state:
                                        del st.session_state['search_results_processed_data']
                                else:
                                    selected_indices = filtered_indices[above_threshold_indices]
                                    results = filtered_df.loc[selected_indices].copy()
                                    results['similarity_score'] = similarities[0][above_threshold_indices]
                                    results = results.sort_values(by='similarity_score', ascending=False)

                                    # Apply negative keyword filter
                                    if negative_keywords.strip():
                                        neg_words = [w.strip().lower() for w in negative_keywords.split(',') if w.strip()]
                                        if neg_words:
                                            results = results[results.apply(lambda row: all(w not in (' '.join(row.astype(str)).lower()) for w in neg_words), axis=1)]

                                    # Apply include keywords filter
                                    if include_keywords.strip():
                                        inc_words = [w.strip().lower() for w in include_keywords.split(',') if w.strip()]
                                        if inc_words:
                                            results = results[results.apply(lambda row: all(w in (' '.join(row.astype(str)).lower()) for w in inc_words), axis=1)]

                                    if results.empty:
                                        st.warning("No results found after applying negative/include keyword filters.")
                                        # Clear previous results if any
                                        if 'search_results' in st.session_state:
                                            del st.session_state['search_results']
                                        if 'search_results_processed_data' in st.session_state:
                                            del st.session_state['search_results_processed_data']
                                    else:
                                        # Store results in session state (DO NOT display them here to avoid duplicate)
                                        st.session_state['search_results'] = results.copy()
                                        # Generate processed_data for download
                                        output = io.BytesIO()
                                        writer = pd.ExcelWriter(output, engine='openpyxl')
                                        results.to_excel(writer, index=False)
                                        writer.close()
                                        processed_data = output.getvalue()
                                        st.session_state['search_results_processed_data'] = processed_data  # Store in session_state
                        else:
                            st.warning("Please enter a query to search.")

                    # Now, outside the 'if st.button("Search")', display results if available
                    if 'search_results' in st.session_state and not st.session_state['search_results'].empty:
                        st.write("Search Results:")
                        results = st.session_state['search_results']
                        columns_to_display = [c for c in results.columns if c not in ['similarity_score']] + ['similarity_score']
                        st.write(results[columns_to_display])

                        # Display total number of results
                        st.write(f"Total number of results: {len(results)}")

                        # Retrieve processed_data from session_state
                        processed_data = st.session_state.get('search_results_processed_data', None)
                        if processed_data is None:
                            # Regenerate processed_data if not available
                            output = io.BytesIO()
                            writer = pd.ExcelWriter(output, engine='openpyxl')
                            results.to_excel(writer, index=False)
                            writer.close()
                            processed_data = output.getvalue()
                            st.session_state['search_results_processed_data'] = processed_data

                        # Download results as Excel
                        st.download_button(
                            label="Download Full Results",
                            data=processed_data,
                            file_name='search_results.xlsx',
                            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                            key='download_search_results'
                        )
                    else:
                        st.info("No search results to display. Please enter a query and click 'Search'.")

                else:
                    st.warning("No embeddings available because no text columns were chosen.")
        else:
            st.warning("The filtered dataset is empty. Please adjust your filters.")
    else:
        st.warning("Please select a dataset to proceed and select text columns.")

# Clustering Tab
# Clustering Tab
# Clustering Tab
with tab2:
    st.header("Clustering")
    if 'filtered_df' in st.session_state and st.session_state['filtered_df'] is not None:
        if not st.session_state['filtered_df'].empty:
            clustering_option = st.radio("Select data for clustering:", ('Full Dataset', 'Semantic Search Results'))
            min_cluster_size_val = st.slider("Min Cluster Size", 2, 50, 5, help="Minimum size of each cluster in HDBSCAN.")

            if clustering_option == 'Semantic Search Results':
                if st.session_state.get('search_results') is not None and not st.session_state['search_results'].empty:
                    df_to_cluster = st.session_state['search_results']
                else:
                    st.warning("No search results found. Please perform a semantic search first.")
                    df_to_cluster = None
            else:
                df_to_cluster = st.session_state['filtered_df']

            if df_to_cluster is not None and not df_to_cluster.empty:
                text_columns = st.session_state.get('text_columns', [])
                if not text_columns:
                    st.warning("No text columns selected. Please select text columns to embed before clustering.")
                else:
                    # Ensure embeddings are computed
                    if 'embeddings' not in st.session_state or st.session_state.get('last_text_columns') != text_columns:
                        df_full = st.session_state['df']
                        embeddings, embeddings_file = load_or_compute_embeddings(df_full, st.session_state.get('using_default_dataset', False), st.session_state.get('uploaded_file_name', None), text_columns)
                    else:
                        embeddings = st.session_state['embeddings']

                    if embeddings is not None:
                        dfc = df_to_cluster.copy()
                        dfc_fill = dfc.fillna("")
                        dfc['text'] = dfc_fill[text_columns].agg(' '.join, axis=1)

                        if len(dfc['text']) == 0:
                            st.warning("No text data available for clustering.")
                        else:
                            texts_cleaned = dfc['text'].tolist()

                            selected_indices = dfc.index
                            embeddings_clustering = embeddings[selected_indices]

                            if st.button("Run Clustering"):
                                with st.spinner("Performing clustering..."):
                                    sentence_model = get_embedding_model()
                                    hdbscan_model = HDBSCAN(min_cluster_size=min_cluster_size_val, metric='euclidean', cluster_selection_method='eom')
                                    topic_model = BERTopic(embedding_model=sentence_model, hdbscan_model=hdbscan_model)
                                    try:
                                        topics, _ = topic_model.fit_transform(texts_cleaned, embeddings=embeddings_clustering)
                                        dfc['Topic'] = topics
                                        st.session_state['topic_model'] = topic_model
                                        if clustering_option == 'Semantic Search Results':
                                            st.session_state['clustered_data'] = dfc.copy()
                                        else:
                                            st.session_state['filtered_df'].loc[dfc.index, 'Topic'] = dfc['Topic']

                                        st.write("Clustering Results:")
                                        columns_to_display = [col for col in dfc.columns if col not in ['text']]
                                        st.write(dfc[columns_to_display])

                                        st.write("Visualizing Topics...")

                                        st.subheader("Intertopic Distance Map")
                                        fig1 = topic_model.visualize_topics()
                                        st.plotly_chart(fig1)

                                        st.subheader("Topic Document Visualization")
                                        fig2 = topic_model.visualize_documents(texts_cleaned, embeddings=embeddings_clustering)
                                        st.plotly_chart(fig2)

                                        st.subheader("Topic Hierarchy Visualization")
                                        fig3 = topic_model.visualize_hierarchy()
                                        st.plotly_chart(fig3)

                                        # Attempt to compute hierarchical topics
                                        st.write("Computing Hierarchical Topics...")
                                        hierarchy = topic_model.hierarchical_topics(texts_cleaned)
                                        # Store hierarchy in session_state so it can be accessed in the Internal Validation tab
                                        st.session_state['hierarchy'] = hierarchy if hierarchy is not None else pd.DataFrame()

                                        # Treemap Visualization of Hierarchical Topics
                                        st.subheader("Hierarchical Topic Treemap")
                                        hierarchy = st.session_state.get('hierarchy', pd.DataFrame())
                                        if hierarchy is not None and not hierarchy.empty:
                                            # Assuming 'Topics' is already a list
                                            parent_dict = {row.Parent_Name: row for _, row in hierarchy.iterrows()}

                                            # Identify top-level parent: assume highest Parent_ID is top
                                            root_row = hierarchy.iloc[hierarchy['Parent_ID'].argmax()]
                                            root_name = root_row.Parent_Name
                                            # If Topics is a list of topic IDs:
                                            all_topics = root_row['Topics']
                                            root_size = len(all_topics)

                                            treemap_nodes = [{"names": "All Topics", "parents": "", "values": root_size}]

                                            def build_nodes(name, parent_name):
                                                """Recursively build treemap nodes from the hierarchy."""
                                                if name in parent_dict:
                                                    # This is a parent node
                                                    row = parent_dict[name]
                                                    node_topics = row['Topics']  # Already a list
                                                    node_size = len(node_topics)

                                                    treemap_nodes.append({
                                                        "names": name,
                                                        "parents": parent_name,
                                                        "values": node_size
                                                    })

                                                    # Get children
                                                    left_child = row['Child_Left_Name']
                                                    right_child = row['Child_Right_Name']

                                                    # Recurse on children
                                                    build_nodes(left_child, name)
                                                    build_nodes(right_child, name)

                                                else:
                                                    # This is a leaf node
                                                    # We assume a leaf node corresponds to a single original topic
                                                    treemap_nodes.append({
                                                        "names": name,
                                                        "parents": parent_name,
                                                        "values": 1
                                                    })

                                            # Build the tree from the root parent
                                            build_nodes(root_name, "All Topics")

                                            # Convert to DataFrame and plot treemap
                                            treemap_df = pd.DataFrame(treemap_nodes)
                                            fig_treemap = px.treemap(treemap_df, names='names', parents='parents', values='values')
                                            fig_treemap.update_traces(root_color="lightgrey")
                                            fig_treemap.update_layout(margin=dict(t=50, l=25, r=25, b=25))
                                            st.plotly_chart(fig_treemap)

                                        else:
                                            st.warning("No hierarchical topic information available for Treemap.")

                                        # Clusters per Year Visualization
                                        st.subheader("Clusters per Year Visualization")
                                        required_columns = ['Year', 'Topic', 'Result code']
                                        missing_columns = [col for col in required_columns if col not in dfc.columns]
                                        if missing_columns:
                                            st.warning(f"The following required columns are missing from the dataset: {', '.join(missing_columns)}")
                                        else:
                                            if not np.issubdtype(dfc['Year'].dtype, np.number):
                                                dfc['Year'] = pd.to_numeric(dfc['Year'], errors='coerce').astype('Int64')
                                            dfc_clean = dfc.dropna(subset=['Year', 'Topic', 'Result code'])
                                            unique_years = sorted(dfc_clean['Year'].unique())
                                            #st.write(f"Data contains the following years: {unique_years}")

                                            if dfc_clean.empty:
                                                st.warning("No valid data available for 'Year', 'Topic', and 'Result code' to generate the visualization.")
                                            else:
                                                df_agg = dfc_clean.groupby(['Year', 'Topic']).agg({'Result code': 'count'}).reset_index().rename(columns={'Result code': 'Count'})
                                                df_agg = df_agg.sort_values(by='Count', ascending=False)
                                                st.write("Aggregated Data for Visualization:")
                                                st.dataframe(df_agg)

                                                unique_topics = df_agg['Topic'].unique()
                                                num_topics = len(unique_topics)
                                                angles = np.linspace(0, 2 * np.pi, num_topics, endpoint=False)
                                                radius = 1
                                                positions = {topic: (np.cos(angle)*radius, np.sin(angle)*radius) for topic, angle in zip(unique_topics, angles)}
                                                df_agg['x_pos'] = df_agg['Topic'].map(lambda t: positions[t][0])
                                                df_agg['y_pos'] = df_agg['Topic'].map(lambda t: positions[t][1])

                                                fig = px.scatter(
                                                    df_agg,
                                                    x="x_pos",
                                                    y="y_pos",
                                                    size="Count",
                                                    color="Topic",
                                                    animation_frame="Year",
                                                    animation_group="Topic",
                                                    hover_name="Topic",
                                                    size_max=60,
                                                    title="Clusters per Year - Animated Bubble Chart",
                                                    labels={"x_pos": "", "y_pos": "", "Count": "Result Code Count"},
                                                    category_orders={"Year": unique_years},
                                                    template="plotly_white"
                                                )

                                                fig.update_xaxes(visible=False)
                                                fig.update_yaxes(visible=False)
                                                fig.update_layout(showlegend=True, title_x=0.5)
                                                fig.layout.updatemenus = [
                                                    dict(
                                                        type="buttons",
                                                        buttons=[
                                                            dict(label="Play",
                                                                 method="animate",
                                                                 args=[None, {"frame": {"duration": 1000, "redraw": True},
                                                                              "fromcurrent": True, "transition": {"duration": 500, "easing": "quadratic-in-out"}}]),
                                                            dict(label="Pause",
                                                                 method="animate",
                                                                 args=[[None], {"frame": {"duration": 0, "redraw": False},
                                                                                "mode": "immediate",
                                                                                "transition": {"duration": 0}}])
                                                        ],
                                                        direction="left",
                                                        pad={"r": 10, "t": 87},
                                                        showactive=False,
                                                        x=0.1,
                                                        xanchor="right",
                                                        y=0,
                                                        yanchor="top"
                                                    )
                                                ]
                                                fig.update_traces(marker=dict(line=dict(width=2, color='DarkSlateGrey')), selector=dict(mode='markers'))
                                                st.plotly_chart(fig, use_container_width=True)

                                    except Exception as e:
                                        st.error(f"An error occurred during clustering: {e}")
                                        # Store exception details in session state to view in internal validation tab
                                        st.session_state['clustering_error'] = str(e)

                    else:
                        st.warning("No embeddings available. Please select text columns and ensure embeddings are computed.")
        else:
            st.warning("No data available for clustering.")
    else:
        st.warning("Please select a dataset to proceed and select text columns.")




# Summarization Tab
with tab3:
    st.header("Summarization")
    if 'filtered_df' in st.session_state and st.session_state['filtered_df'] is not None:
        if not st.session_state['filtered_df'].empty:
            # Determine which df to use for summarization
            if 'clustered_data' in st.session_state and not st.session_state['clustered_data'].empty:
                df_summ = st.session_state['clustered_data']
            else:
                df_summ = st.session_state['filtered_df']

            if df_summ is not None and not df_summ.empty:
                text_columns = st.session_state.get('text_columns', [])
                if not text_columns:
                    st.warning("No text columns selected. Please select text columns before summarization.")
                else:
                    if 'Topic' in df_summ.columns and 'topic_model' in st.session_state:
                        topic_model = st.session_state['topic_model']
                        topics = df_summ['Topic'].unique()
                        # Prepare text column if not already
                        df_summ_fill = df_summ.fillna("")
                        df_summ['text'] = df_summ_fill[text_columns].agg(' '.join, axis=1)

                        # Get cluster info: count, top keywords
                        cluster_info = []
                        for t in topics:
                            cluster_docs = df_summ[df_summ['Topic'] == t]
                            count = len(cluster_docs)
                            top_words = topic_model.get_topic(t)
                            if top_words:
                                top_keywords = ", ".join([w[0] for w in top_words[:5]])
                            else:
                                top_keywords = "N/A"
                            cluster_info.append((t, count, top_keywords))
                        cluster_df = pd.DataFrame(cluster_info, columns=["Topic", "Count", "Top Keywords"])

                        st.write("Available Clusters for Summarization:")
                        st.dataframe(cluster_df)

                        selected_topics = st.multiselect("Select clusters to summarize", cluster_df["Topic"].tolist())

                        temperature = st.slider("Summarization Temperature", 0.0, 1.0, 0.7)
                        max_tokens = st.slider("Max Tokens for Summarization", 100, 3000, 1000)

                        if st.button("Generate Summaries"):
                            system_prompt = """
            You are an expert summarizer skilled in creating concise and relevant summaries.
            You will be given text and an objective context. Please produce a clear, cohesive, and thematically relevant summary.
            Make sure the summary is understandable without requiring the entire original text. Focus on key points, insights, or
            patterns that emerge from the text.
            """

                            openai_api_key = os.environ.get('OPENAI_API_KEY')
                            if not openai_api_key:
                                st.error("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
                            else:
                                llm = ChatOpenAI(api_key=openai_api_key, model_name='gpt-4o', temperature=temperature, max_tokens=max_tokens)

                                if selected_topics:
                                    df_to_summarize = df_summ[df_summ['Topic'].isin(selected_topics)]
                                else:
                                    df_to_summarize = df_summ

                                all_texts = df_to_summarize['text'].tolist()
                                combined_text = " ".join(all_texts)
                                if combined_text.strip() == "":
                                    st.warning("No text data available for summarization.")
                                else:
                                    user_prompt = f"**Text to summarize**: {combined_text}"
                                    system_message = SystemMessagePromptTemplate.from_template(system_prompt)
                                    human_message = HumanMessagePromptTemplate.from_template("{user_prompt}")
                                    chat_prompt = ChatPromptTemplate.from_messages([system_message, human_message])
                                    with st.spinner("Generating high-level summary..."):
                                        chain = LLMChain(llm=llm, prompt=chat_prompt)
                                        response = chain.run(user_prompt=user_prompt)
                                    high_level_summary = response.strip()
                                    st.write("### High-Level Summary:")
                                    st.write(high_level_summary)

                                    # Summaries per cluster
                                    if selected_topics:
                                        summaries = []
                                        grouped_list = list(df_to_summarize.groupby('Topic'))
                                        grouped_list = [g for g in grouped_list if g[0] in selected_topics]

                                        total_topics = len(grouped_list)
                                        if total_topics == 0:
                                            st.warning("No topics found for summarization after selection.")
                                        else:
                                            progress_bar = st.progress(0)

                                            def generate_summary_per_topic(topic_group_tuple):
                                                topic, group = topic_group_tuple
                                                all_text = " ".join(group['text'].tolist())
                                                user_prompt = f"**Text to summarize**: {all_text}"
                                                system_message = SystemMessagePromptTemplate.from_template(system_prompt)
                                                human_message = HumanMessagePromptTemplate.from_template("{user_prompt}")
                                                chat_prompt = ChatPromptTemplate.from_messages([system_message, human_message])
                                                chain = LLMChain(llm=llm, prompt=chat_prompt)
                                                response = chain.run(user_prompt=user_prompt)
                                                summary = response.strip()
                                                return {'Topic': topic, 'Summary': summary}

                                            with st.spinner("Summarizing each selected cluster..."):
                                                with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                                                    futures = {executor.submit(generate_summary_per_topic, item): item[0] for item in grouped_list}
                                                    for idx, future in enumerate(concurrent.futures.as_completed(futures)):
                                                        result = future.result()
                                                        summaries.append(result)
                                                        progress_bar.progress((idx + 1) / total_topics)
                                            progress_bar.empty()

                                            if summaries:
                                                summary_df = pd.DataFrame(summaries)
                                                st.write("### Summaries per Cluster:")
                                                st.write(summary_df)
                                                csv = summary_df.to_csv(index=False)
                                                b64 = base64.b64encode(csv.encode()).decode()
                                                href = f'<a href="data:file/csv;base64,{b64}" download="summaries.csv">Download Summaries CSV</a>'
                                                st.markdown(href, unsafe_allow_html=True)
                    else:
                        st.warning("Please perform clustering first to generate topics.")
            else:
                st.warning("No data available for summarization.")
        else:
            st.warning("The filtered dataset is empty. Please adjust your filters.")
    else:
        st.warning("Please select a dataset and select text columns to proceed.")

# Internal Validation Tab
with tab_internal:
    st.header("Internal Validation & Debugging")

    # Display hierarchical topics DataFrame if available
    hierarchy = st.session_state.get('hierarchy', pd.DataFrame())
    st.subheader("Hierarchical Topics DataFrame")
    if not hierarchy.empty:
        st.dataframe(hierarchy)
    else:
        st.write("No hierarchical topics data available.")

    # Display clustering error details if available
    error_msg = st.session_state.get('clustering_error', None)
    if error_msg:
        st.subheader("Clustering Error Details")
        st.write(error_msg)

    # Display other internal variables
    st.subheader("Internal Variables")
    st.write("Min Cluster Size:", st.session_state.get('min_cluster_size_val', 'Not set'))
    st.write("Text Columns:", st.session_state.get('text_columns', 'Not set'))
    st.write("Using Default Dataset:", st.session_state.get('using_default_dataset', 'Not set'))
    st.write("Last Text Columns Used for Embeddings:", st.session_state.get('last_text_columns', 'Not set'))

    # If needed, display filtered_df, clustered_data, embeddings shape, etc.
    filtered_df = st.session_state.get('filtered_df', pd.DataFrame())
    st.subheader("Filtered DF")
    st.write(filtered_df.head())

    clustered_data = st.session_state.get('clustered_data', pd.DataFrame())
    st.subheader("Clustered Data")
    st.write(clustered_data.head())

    if 'embeddings' in st.session_state and st.session_state['embeddings'] is not None:
        st.subheader("Embeddings Information")
        st.write("Shape of Embeddings:", st.session_state['embeddings'].shape)
    else:
        st.write("No embeddings data available.")