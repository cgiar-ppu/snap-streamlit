# app.py

import streamlit as st
# Set page config first, before any other st commands
st.set_page_config(page_title="SNAP", layout="wide")

# Add warning filters
import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='.*torch.classes.*')

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
import torch

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

# Determine device - will use GPU if available, otherwise CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Log device being used - helpful for debugging
if device == 'cuda':
    st.sidebar.success(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    st.sidebar.info("Using CPU")

def init_nltk_resources():
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab')

init_nltk_resources()

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
    st.session_state['selected_additional_filters'] = {}

if dataset_option == 'PRMS 2022+2023 QAed':
    default_dataset_path = os.path.join(os.path.dirname(__file__), 'input', 'export_data_table_results_20240312_160222CET.xlsx')
    df = load_default_dataset(default_dataset_path)
    if df is not None:
        st.session_state['df'] = df.copy()
        st.session_state['using_default_dataset'] = True
        st.write("Using default dataset:")

        # Get all columns for filtering
        df_cols = df.columns.tolist()

        # Additional filter columns
        st.write("**Select Filters**")
        all_columns = df.columns.tolist()
        selected_additional_cols = st.multiselect("Select columns from your dataset to use as filters:", all_columns, default=st.session_state.get('additional_filters_selected', []))
        st.session_state['additional_filters_selected'] = selected_additional_cols

        # For each chosen filter column, show a multiselect of unique values
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
        text_columns_selected = st.multiselect(
            "Text Columns:",
            all_columns,
            default=['Title','Description'] if 'Title' in df.columns and 'Description' in df.columns else [],
            help="Choose columns containing text that you want to search through or analyze. Selected columns will be used for semantic search (finding similar content) and clustering (grouping similar documents). If multiple columns are selected, their text will be combined. It is necessary to select at least one column from the dataset as it will contain the text to be searched through."
        )
        st.session_state['text_columns'] = text_columns_selected

        # Apply filters to create filtered_df
        filtered_df = df.copy()

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

            st.write("**Additional Filters**")
            selected_additional_cols = st.multiselect("Select additional columns from your dataset to use as filters:", df_cols, default=[])
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
            # Add explanation about semantic search
            with st.expander("ℹ️ How Semantic Search Works"):
                st.markdown("""
                ### Understanding Semantic Search

                Unlike traditional keyword search that looks for exact matches, semantic search understands the meaning and context of your query. Here's how it works:

                1. **Query Processing**:
                   - Your search query is converted into a numerical representation (embedding) that captures its meaning
                   - Example: Searching for "Climate Smart Villages" will understand the concept, not just the words
                   - Related terms like "sustainable communities", "resilient farming", or "agricultural adaptation" might be found even if they don't contain the exact words

                2. **Similarity Matching**:
                   - Documents are ranked by how closely their meaning matches your query
                   - The similarity threshold controls how strict this matching is
                   - Higher threshold (e.g., 0.8) = more precise but fewer results
                   - Lower threshold (e.g., 0.3) = more results but might be less relevant

                3. **Advanced Features**:
                   - **Negative Keywords**: Use to explicitly exclude documents containing certain terms
                   - **Required Keywords**: Ensure specific terms appear in the results
                   - These work as traditional keyword filters after the semantic search

                ### Search Tips

                - **Phrase Queries**: Enter complete phrases for better context
                  - "Climate Smart Villages" (as one concept)
                  - Better than separate terms: "climate", "smart", "villages"

                - **Descriptive Queries**: Add context for better results
                  - Instead of: "water"
                  - Better: "water management in agriculture"

                - **Conceptual Queries**: Focus on concepts rather than specific terms
                  - Instead of: "increased yield"
                  - Better: "agricultural productivity improvements"

                ### Example Searches

                1. **Query**: "Climate Smart Villages"
                   - Will find: Documents about climate-resilient communities, adaptive farming practices, sustainable village development
                   - Even if they don't use these exact words

                2. **Query**: "Gender equality in agriculture"
                   - Will find: Women's empowerment in farming, female farmer initiatives, gender-inclusive rural development
                   - Related concepts are captured semantically

                3. **Query**: "Sustainable water management"
                   + Required keyword: "irrigation"
                   - Combines semantic understanding of water sustainability with specific irrigation focus
                """)
            
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
                    # Create two columns, form will be in the left column
                    left_col, right_col = st.columns(2)
                    with left_col:
                        with st.form("search_parameters"):
                            query = st.text_input("Enter your search query:")
                            include_keywords = st.text_input("Include only documents containing these words (comma-separated):")
                            similarity_threshold = st.slider("Similarity threshold", 0.0, 1.0, 0.35)
                            submitted = st.form_submit_button("Search")

                    if submitted:
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

                                # Create histogram of similarity scores
                                fig = px.histogram(
                                    x=similarities[0],
                                    nbins=30,
                                    labels={'x': 'Similarity Score', 'y': 'Number of Documents'},
                                    title='Distribution of Similarity Scores'
                                )
                                
                                # Add vertical line for threshold
                                fig.add_vline(
                                    x=similarity_threshold,
                                    line_dash="dash",
                                    line_color="red",
                                    annotation_text=f"Threshold: {similarity_threshold:.2f}",
                                    annotation_position="top"
                                )
                                
                                # Update layout with improved styling
                                fig.update_layout(
                                    title_x=0.5,
                                    showlegend=False,
                                    margin=dict(t=50, l=50, r=50, b=50),
                                    hoverlabel=dict(
                                        bgcolor="black",
                                        font_size=14,
                                        font_color="white"
                                    ),
                                    hovermode='x',
                                    xaxis=dict(
                                        showgrid=False,
                                        zeroline=False
                                    ),
                                    yaxis=dict(
                                        showgrid=False,
                                        zeroline=False
                                    )
                                )
                                
                                # Update bar style with borders
                                fig.update_traces(
                                    hovertemplate="Similarity Score: %{x:.3f}<br>Count: %{y}",
                                    marker_line_width=1,
                                    marker_line_color="rgb(150,150,150)",
                                    opacity=0.8
                                )
                                
                                # Display plot and explanation
                                st.write("### Similarity Score Distribution")
                                st.write("""
                                This histogram shows how many documents fall into each similarity score range:
                                - Documents to the right of the red line (threshold) will be included in results
                                - A good threshold balances precision (high similarity) with recall (enough results)
                                - Adjust the threshold to include more results (move left) or be more selective (move right)
                                """)
                                st.plotly_chart(fig)

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

                                    # Apply include keywords filter
                                    if include_keywords.strip():
                                        inc_words = [w.strip().lower() for w in include_keywords.split(',') if w.strip()]
                                        if inc_words:
                                            results = results[results.apply(lambda row: all(w in (' '.join(row.astype(str)).lower()) for w in inc_words), axis=1)]

                                    if results.empty:
                                        st.warning("No results found after applying keyword filters.")
                                        # Clear previous results if any
                                        if 'search_results' in st.session_state:
                                            del st.session_state['search_results']
                                        if 'search_results_processed_data' in st.session_state:
                                            del st.session_state['search_results_processed_data']
                                    else:
                                        # Store results in session state
                                        st.session_state['search_results'] = results.copy()
                                        # Generate processed_data for download
                                        output = io.BytesIO()
                                        writer = pd.ExcelWriter(output, engine='openpyxl')
                                        results.to_excel(writer, index=False)
                                        writer.close()
                                        processed_data = output.getvalue()
                                        st.session_state['search_results_processed_data'] = processed_data
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
with tab2:
    st.header("Clustering")
    if 'filtered_df' in st.session_state and st.session_state['filtered_df'] is not None:
        if not st.session_state['filtered_df'].empty:
            # Add explanation about clustering
            with st.expander("ℹ️ How Clustering Works"):
                st.markdown("""
                ### Understanding Document Clustering

                Clustering automatically groups similar documents together, helping you discover patterns and themes in your data. Here's how it works:

                1. **Cluster Formation**:
                   - Documents are grouped based on their semantic similarity
                   - Each cluster represents a distinct theme or topic
                   - Documents that are too different from others may remain unclustered (labeled as -1)
                   - The "Min Cluster Size" parameter controls how clusters are formed

                2. **Interpreting Results**:
                   - Each cluster is assigned a number (e.g., 0, 1, 2...)
                   - Cluster -1 contains "outlier" documents that didn't fit well in other clusters
                   - The size of each cluster indicates how common that theme is
                   - Keywords for each cluster show the main topics/concepts

                3. **Visualizations**:
                   - **Intertopic Distance Map**: Shows how clusters relate to each other
                     - Closer clusters are more semantically similar
                     - Size of circles indicates number of documents
                     - Hover to see top terms for each cluster
                   
                   - **Topic Document Visualization**: Shows individual documents
                     - Each point is a document
                     - Colors indicate cluster membership
                     - Distance between points shows similarity
                   
                   - **Topic Hierarchy**: Shows how topics are related
                     - Tree structure shows topic relationships
                     - Parent topics contain broader themes
                     - Child topics show more specific sub-themes

                ### How to Use Clusters

                1. **Exploration**:
                   - Use clusters to discover main themes in your data
                   - Look for unexpected groupings that might reveal insights
                   - Identify outliers that might need special attention

                2. **Analysis**:
                   - Compare cluster sizes to understand theme distribution
                   - Examine keywords to understand what defines each cluster
                   - Use hierarchy to see how themes are nested

                3. **Practical Applications**:
                   - Generate summaries for specific clusters
                   - Focus detailed analysis on clusters of interest
                   - Use clusters to organize and categorize documents
                   - Identify gaps or overlaps in your dataset

                ### Tips for Better Results

                - **Adjust Min Cluster Size**:
                  - Larger values (15-20): Fewer, broader clusters
                  - Smaller values (2-5): More specific, smaller clusters
                  - Balance between too many small clusters and too few large ones

                - **Choose Data Wisely**:
                  - Cluster full dataset for overall themes
                  - Cluster search results for focused analysis
                  - More documents generally give better clusters

                - **Interpret with Context**:
                  - Consider your domain knowledge
                  - Look for patterns across multiple visualizations
                  - Use cluster insights to guide further analysis
                """)
            
            col1, col2 = st.columns(2)
            with col1:
                with st.form("clustering_parameters"):
                    clustering_option = st.radio("Select data for clustering:", ('Full Dataset', 'Semantic Search Results'))
                    if 'min_cluster_size' not in st.session_state:
                        st.session_state.min_cluster_size = 5
                    min_cluster_size_val = st.slider(
                        "Min Cluster Size",
                        min_value=2,
                        max_value=50,
                        value=st.session_state.min_cluster_size,
                        help="Minimum size of each cluster in HDBSCAN; In other words, it's the minimum number of documents/texts that must be grouped together to form a valid cluster.\n\n- A larger value (e.g., 20) will result in fewer, larger clusters\n- A smaller value (e.g., 2-5) will allow for more clusters, including smaller ones\n- Documents that don't fit into any cluster meeting this minimum size requirement are labeled as noise (typically assigned to cluster -1)",
                        key="min_cluster_size"
                    )
                    submitted = st.form_submit_button("Run Clustering")

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
                            # Clean texts by removing stop words
                            stop_words = set(stopwords.words('english'))
                            texts_cleaned = []
                            for text in dfc['text'].tolist():
                                word_tokens = word_tokenize(text)
                                filtered_text = ' '.join([word for word in word_tokens if word.lower() not in stop_words])
                                texts_cleaned.append(filtered_text)

                            selected_indices = dfc.index
                            embeddings_clustering = embeddings[selected_indices]

                            if submitted:
                                with st.spinner("Performing clustering..."):
                                    sentence_model = get_embedding_model()
                                    # Note: HDBSCAN only runs on CPU, so we need to ensure embeddings are on CPU
                                    embeddings_for_clustering = embeddings_clustering.cpu().numpy() if torch.is_tensor(embeddings_clustering) else embeddings_clustering
                                    
                                    # Initialize HDBSCAN (CPU-only operation)
                                    hdbscan_model = HDBSCAN(min_cluster_size=min_cluster_size_val, 
                                                          metric='euclidean', 
                                                          cluster_selection_method='eom')
                                    
                                    topic_model = BERTopic(embedding_model=sentence_model, 
                                                         hdbscan_model=hdbscan_model)
                                    try:
                                        # Ensure embeddings are on CPU for clustering
                                        topics, _ = topic_model.fit_transform(texts_cleaned, embeddings=embeddings_for_clustering)
                                        dfc['Topic'] = topics
                                        st.session_state['topic_model'] = topic_model
                                        # Store the clustered data in session state
                                        st.session_state['clustered_data'] = dfc.copy()
                                        # Also update the filtered_df with topics
                                        if clustering_option == 'Full Dataset':
                                            st.session_state['filtered_df'].loc[dfc.index, 'Topic'] = dfc['Topic']

                                        # Add topic overview table first
                                        st.subheader("Topic Overview")
                                        
                                        # Get unique topics and their counts
                                        topic_counts = dfc['Topic'].value_counts().sort_index()
                                        
                                        # Create overview data
                                        overview_data = []
                                        for topic_id in topic_counts.index:
                                            # Get top words for the topic
                                            top_words = topic_model.get_topic(topic_id)
                                            # Format keywords and their weights
                                            if top_words:
                                                keywords = ", ".join([f"{word} ({weight:.3f})" for word, weight in top_words[:5]])
                                            else:
                                                keywords = "N/A"
                                            
                                            # Add to overview data
                                            overview_data.append({
                                                "Topic": int(topic_id),
                                                "Size": topic_counts[topic_id],
                                                "% of Total": f"{(topic_counts[topic_id] / len(dfc) * 100):.1f}%",
                                                "Top Keywords (with weights)": keywords
                                            })
                                        
                                        # Create and display overview DataFrame
                                        overview_df = pd.DataFrame(overview_data)
                                        st.dataframe(
                                            overview_df,
                                            column_config={
                                                "Topic": st.column_config.NumberColumn("Topic", help="Topic ID (-1 represents outliers)"),
                                                "Size": st.column_config.NumberColumn("Size", help="Number of documents in this topic"),
                                                "% of Total": st.column_config.TextColumn("% of Total", help="Percentage of total documents"),
                                                "Top Keywords (with weights)": st.column_config.TextColumn(
                                                    "Top Keywords (with weights)",
                                                    help="Top 5 keywords and their importance weights for this topic"
                                                )
                                            }
                                        )

                                        # Then show full clustering results
                                        st.subheader("Clustering Results")
                                        columns_to_display = [col for col in dfc.columns if col not in ['text']]
                                        st.write(dfc[columns_to_display])

                                        st.write("Visualizing Topics...")

                                        st.subheader("Intertopic Distance Map")
                                        fig1 = topic_model.visualize_topics()
                                        # Modify only the tooltip style for better readability
                                        fig1.update_traces(
                                            hoverlabel=dict(
                                                bgcolor='black',
                                                font_size=14,
                                                font_color='white'
                                            )
                                        )
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
                        # Ensure we're using the correct DataFrame with topics
                        if 'clustered_data' in st.session_state and not st.session_state['clustered_data'].empty:
                            df_summ = st.session_state['clustered_data'].copy()
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

                        left_col, right_col = st.columns(2)
                        with left_col:
                            # Initialize summary scope in session state if not present
                            if 'summary_scope' not in st.session_state:
                                st.session_state.summary_scope = "All clusters"
                            
                            # Place the radio button outside the form
                            st.session_state.summary_scope = st.radio(
                                "Generate summaries for:",
                                ["All clusters", "Specific clusters"]
                            )

                            with st.form("summarization_parameters"):
                                # Convert topics to integers for display
                                topic_options = [int(t) for t in cluster_df["Topic"].tolist()]
                                
                                # Show cluster selection based on summary scope
                                if st.session_state.summary_scope == "Specific clusters":
                                    selected_topics = st.multiselect("Select clusters to summarize", topic_options)
                                else:
                                    selected_topics = topic_options  # Use all topics

                                temperature = st.slider("Summarization Temperature", 0.0, 1.0, 0.7)
                                max_tokens = st.slider("Max Tokens for Summarization", 100, 3000, 1000)
                                submitted = st.form_submit_button("Generate Summaries")

                        if submitted:
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
                                    # Convert selected topics back to float for DataFrame filtering
                                    selected_topics_float = [float(t) for t in selected_topics]
                                    df_to_summarize = df_summ[df_summ['Topic'].isin(selected_topics_float)]
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