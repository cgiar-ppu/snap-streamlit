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

# Initialize session state for active tab
if 'active_tab' not in st.session_state:
    st.session_state.active_tab = "Semantic Search"

# Initialize session state variables for summarization
if 'enable_references' not in st.session_state:
    st.session_state.enable_references = True
if 'reference_id_column' not in st.session_state:
    st.session_state.reference_id_column = None
if 'has_url_column' not in st.session_state:
    st.session_state.has_url_column = True
if 'url_column' not in st.session_state:
    st.session_state.url_column = None
if 'summary_scope' not in st.session_state:
    st.session_state.summary_scope = "All clusters"

# Function to ensure all session state variables are initialized
def initialize_session_state():
    """Initialize all session state variables to prevent errors."""
    # Tab state
    if 'active_tab' not in st.session_state:
        st.session_state.active_tab = "Semantic Search"
    
    # Data state
    if 'df' not in st.session_state:
        st.session_state.df = pd.DataFrame()
    if 'filtered_df' not in st.session_state:
        st.session_state.filtered_df = pd.DataFrame()
    if 'using_default_dataset' not in st.session_state:
        st.session_state.using_default_dataset = False
    if 'text_columns' not in st.session_state:
        st.session_state.text_columns = []
    
    # Filter state
    if 'filter_state' not in st.session_state:
        st.session_state.filter_state = {'applied': False, 'filters': {}}
    if 'filter_values' not in st.session_state:
        st.session_state.filter_values = {}
    if 'additional_filters_selected' not in st.session_state:
        st.session_state.additional_filters_selected = []
    
    # Search state
    if 'search_results' not in st.session_state:
        st.session_state.search_results = pd.DataFrame()
    
    # Clustering state
    if 'min_cluster_size' not in st.session_state:
        st.session_state.min_cluster_size = 5
    if 'clustered_data' not in st.session_state:
        st.session_state.clustered_data = pd.DataFrame()
    if 'clustering_params' not in st.session_state:
        st.session_state.clustering_params = {}
    
    # Summarization state
    if 'enable_references' not in st.session_state:
        st.session_state.enable_references = True
    if 'reference_id_column' not in st.session_state:
        st.session_state.reference_id_column = None
    if 'has_url_column' not in st.session_state:
        st.session_state.has_url_column = True
    if 'url_column' not in st.session_state:
        st.session_state.url_column = None
    if 'summary_scope' not in st.session_state:
        st.session_state.summary_scope = "All clusters"
    if 'summary_results' not in st.session_state:
        st.session_state.summary_results = []
    if 'summary_params' not in st.session_state:
        st.session_state.summary_params = {}
    if 'high_level_summary' not in st.session_state:
        st.session_state.high_level_summary = ""

# Initialize all session state variables
initialize_session_state()

################################################################################
# NEW: Function to add references to summaries
################################################################################
def add_references_to_summary(summary, source_df, reference_column, url_column=None, llm=None):
    """
    Add references to a summary by identifying which parts of the summary come from which source documents.
    
    Args:
        summary (str): The summary text to enhance with references
        source_df (DataFrame): DataFrame containing the source documents
        reference_column (str): Column name to use for reference IDs
        url_column (str, optional): Column name containing URLs for hyperlinks
        llm (LLM, optional): Language model for source attribution
        
    Returns:
        str: Enhanced summary with references as HTML
    """
    if summary.strip() == "" or source_df.empty or reference_column not in source_df.columns:
        return summary
    
    # If no LLM is provided, we can't do source attribution
    if llm is None:
        return summary
    
    try:
        # Split the summary into sentences for processing
        sentences = re.split(r'(?<=[.!?])\s+', summary)
        
        # Prepare source texts with their reference IDs
        source_texts = []
        reference_ids = []
        urls = []
        
        for _, row in source_df.iterrows():
            if 'text' in row and pd.notna(row['text']) and reference_column in row and pd.notna(row[reference_column]):
                source_texts.append(str(row['text']))
                reference_ids.append(str(row[reference_column]))
                if url_column and url_column in row and pd.notna(row[url_column]):
                    urls.append(str(row[url_column]))
                else:
                    urls.append(None)
        
        # If we have no valid sources, return the original summary
        if not source_texts:
            return summary
        
        # Create a mapping between URLs and reference IDs if needed
        url_map = {}
        if url_column:
            for ref_id, url in zip(reference_ids, urls):
                if url:
                    url_map[ref_id] = url
        
        # Define the system prompt for source attribution
        system_prompt = """
        You are an expert at identifying the source of information. You will be given:
        1. A sentence from a summary
        2. A list of source texts with their IDs
        
        Your task is to identify which source text(s) the sentence most likely came from.
        Return ONLY the IDs of the source texts that contributed to the sentence, separated by commas.
        If you cannot confidently attribute the sentence to any source, return "unknown".
        """
        
        enhanced_sentences = []
        # Increase batch size to reduce API calls
        batch_size = 10
        
        # Limit source text length to avoid token limits
        max_source_text_length = 300  # characters per source
        
        # Process sentences in batches to reduce API calls
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i+batch_size]
            batch_results = []
            
            for sentence in batch:
                if sentence.strip():
                    # Create the prompt for this sentence with truncated source texts
                    source_text_snippets = [f"ID: {ref_id}, Text: {text[:max_source_text_length]}..." 
                                          for ref_id, text in zip(reference_ids, source_texts)]
                    
                    # Limit number of sources if there are too many
                    if len(source_text_snippets) > 10:
                        source_text_snippets = source_text_snippets[:10]
                        
                    user_prompt = f"""
                    Sentence: {sentence}
                    
                    Source texts:
                    {chr(10).join(source_text_snippets)}
                    
                    Which source ID(s) did this sentence most likely come from? Return only the ID(s) separated by commas, or "unknown".
                    """
                    
                    # Create the messages for the chat
                    system_message = SystemMessagePromptTemplate.from_template(system_prompt)
                    human_message = HumanMessagePromptTemplate.from_template("{user_prompt}")
                    chat_prompt = ChatPromptTemplate.from_messages([system_message, human_message])
                    
                    try:
                        chain = LLMChain(llm=llm, prompt=chat_prompt)
                        response = chain.run(user_prompt=user_prompt)
                        source_ids = response.strip()
                        
                        if source_ids.lower() == "unknown":
                            source_ids = ""
                        else:
                            # Extract just the IDs, removing extraneous chars
                            source_ids = re.sub(r'[^0-9,\s]', '', source_ids)
                            source_ids = re.sub(r'\s+', '', source_ids)
                        
                        batch_results.append((sentence, source_ids))
                    except Exception as e:
                        # If there's an error, just use the sentence without attribution
                        batch_results.append((sentence, ""))
                else:
                    batch_results.append((sentence, ""))
            
            # Turn each sentence into an enhanced sentence
            for sentence, source_ids in batch_results:
                if source_ids:
                    ids = [id.strip() for id in source_ids.split(',') if id.strip()]
                    ref_parts = []
                    for id_ in ids:
                        # If there's a URL for that reference ID, make it clickable
                        if id_ in url_map and url_map[id_]:
                            ref_parts.append(f'<a href="{url_map[id_]}" target="_blank">{id_}</a>')
                        else:
                            ref_parts.append(id_)
                    
                    ref_string = ", ".join(ref_parts)
                    enhanced_sentence = f"{sentence} [{ref_string}]"
                    enhanced_sentences.append(enhanced_sentence)
                else:
                    enhanced_sentences.append(sentence)
        
        enhanced_summary = " ".join(enhanced_sentences)
        return enhanced_summary
    except Exception as e:
        # If anything goes wrong, return the original summary
        return f"{summary} [Error adding references: {str(e)}]"

################################################################################
# End of new function
################################################################################

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
        # Add a progress bar
        progress_bar = st.progress(0)
        total_texts = len(texts)
        
        # Process in batches to show progress
        batch_size = 100
        all_embeddings = []
        
        for i in range(0, total_texts, batch_size):
            batch = texts[i:min(i+batch_size, total_texts)]
            batch_embeddings = model.encode(batch, show_progress_bar=False, device=device)
            all_embeddings.append(batch_embeddings)
            progress_bar.progress(min((i + batch_size) / total_texts, 1.0))
        
        # Combine all batches
        if len(all_embeddings) == 1:
            embeddings = all_embeddings[0]
        else:
            embeddings = np.vstack(all_embeddings)
        
        # Clear the progress bar
        progress_bar.empty()
        
    return embeddings

@st.cache_resource
def get_embedding_model():
    model = SentenceTransformer('all-MiniLM-L6-v2').to(device)
    return model

def load_or_compute_embeddings(df, using_default_dataset, uploaded_file_name=None, text_columns=None):
    if text_columns is None or len(text_columns) == 0:
        return None, None  # No columns chosen, cannot compute embeddings

    # Construct a path for storing embeddings
    embeddings_dir = os.path.dirname(__file__)
    cols_key = "_".join(sorted(text_columns))
    if using_default_dataset:
        embeddings_file = os.path.join(embeddings_dir, f'PRMS_2022_2023_QAed_{cols_key}.pkl')
    else:
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = os.path.splitext(uploaded_file_name)[0] if uploaded_file_name else "custom_dataset"
        embeddings_file = os.path.join(embeddings_dir, f"{base_name}_{cols_key}_{timestamp_str}.pkl")

    df_fill = df.fillna("")
    texts = df_fill[text_columns].astype(str).agg(' '.join, axis=1).tolist()

    # Check session_state cache first
    if ('embeddings' in st.session_state and 
        'last_text_columns' in st.session_state and 
        st.session_state['last_text_columns'] == text_columns and 
        len(st.session_state['embeddings']) == len(texts)):
        return st.session_state['embeddings'], st.session_state.get('embeddings_file', embeddings_file)

    # If embeddings file exists on disk, try to load and match size
    if os.path.exists(embeddings_file):
        try:
            with open(embeddings_file, 'rb') as f:
                embeddings = pickle.load(f)
            if len(embeddings) == len(texts):
                st.session_state['embeddings'] = embeddings
                st.session_state['embeddings_file'] = embeddings_file
                st.session_state['last_text_columns'] = text_columns
                return embeddings, embeddings_file
        except Exception as e:
            st.warning(f"Error loading embeddings: {str(e)}. Computing new embeddings.")

    # If we get here, we need to compute embeddings
    model = get_embedding_model()
    embeddings = generate_embeddings(texts, model)
    
    # Save embeddings to disk
    try:
        with open(embeddings_file, 'wb') as f:
            pickle.dump(embeddings, f)
    except Exception as e:
        st.warning(f"Error saving embeddings: {str(e)}")
    
    # Update session state
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
        st.subheader("Select Filters")
        
        if 'additional_filters_selected' not in st.session_state:
            st.session_state['additional_filters_selected'] = []
        if 'filter_values' not in st.session_state:
            st.session_state['filter_values'] = {}

        with st.form("filter_selection_form"):
            all_columns = df.columns.tolist()
            selected_additional_cols = st.multiselect(
                "Select columns from your dataset to use as filters:",
                all_columns,
                default=st.session_state['additional_filters_selected']
            )
            add_filters_submitted = st.form_submit_button("Add Additional Filters")
            
        if add_filters_submitted:
            if selected_additional_cols != st.session_state['additional_filters_selected']:
                st.session_state['additional_filters_selected'] = selected_additional_cols
                # Reset values for removed columns
                st.session_state['filter_values'] = {
                    k: v for k, v in st.session_state['filter_values'].items() 
                    if k in selected_additional_cols
                }

        # Show dynamic filters form if any selected columns
        if st.session_state['additional_filters_selected']:
            st.subheader("Apply Filters")
            with st.form("apply_filters_form"):
                for col_name in st.session_state['additional_filters_selected']:
                    unique_vals = sorted(df[col_name].dropna().unique().tolist())
                    selected_vals = st.multiselect(
                        f"Filter by {col_name}",
                        options=unique_vals,
                        default=st.session_state['filter_values'].get(col_name, [])
                    )
                    st.session_state['filter_values'][col_name] = selected_vals
                
                apply_filters_submitted = st.form_submit_button("Apply Filters to Dataset")

        # Text columns selection
        st.subheader("**Select Text Columns for Embedding**")
        text_columns_selected = st.multiselect(
            "Text Columns:",
            all_columns,
            default=['Title','Description'] if 'Title' in df.columns and 'Description' in df.columns else [],
            help="Choose columns containing text for semantic search and clustering. If multiple are selected, their text will be concatenated."
        )
        st.session_state['text_columns'] = text_columns_selected

        filtered_df = df.copy()

        if 'apply_filters_submitted' in locals() and apply_filters_submitted:
            for col_name in st.session_state['additional_filters_selected']:
                selected_vals = st.session_state['filter_values'].get(col_name, [])
                if selected_vals:
                    filtered_df = filtered_df[filtered_df[col_name].isin(selected_vals)]
            st.success("Filters applied successfully!")
            st.session_state['filtered_df'] = filtered_df.copy()
            st.session_state['filter_state'] = {
                'applied': True,
                'filters': st.session_state['filter_values'].copy()
            }
            # Reset any existing clustering results
            if 'clustered_data' in st.session_state:
                del st.session_state['clustered_data']
            if 'topic_model' in st.session_state:
                del st.session_state['topic_model']
            if 'current_clustering_data' in st.session_state:
                del st.session_state['current_clustering_data']
            if 'current_clustering_option' in st.session_state:
                del st.session_state['current_clustering_option']
            if 'hierarchy' in st.session_state:
                del st.session_state['hierarchy']

        elif 'filter_state' in st.session_state and st.session_state['filter_state']['applied']:
            for col_name, selected_vals in st.session_state['filter_state']['filters'].items():
                if selected_vals:
                    filtered_df = filtered_df[filtered_df[col_name].isin(selected_vals)]
            st.session_state['filtered_df'] = filtered_df.copy()

        if 'filtered_df' in st.session_state:
            st.write("Filtered Data Preview:")
            st.write(st.session_state['filtered_df'].head())
            st.write(f"Total number of results: {len(st.session_state['filtered_df'])}")

            output = io.BytesIO()
            writer = pd.ExcelWriter(output, engine='openpyxl')
            st.session_state['filtered_df'].to_excel(writer, index=False)
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
            df_cols = df.columns.tolist()

            st.subheader("**Select Text Columns for Embedding**")
            text_columns_selected = st.multiselect("Text Columns:", df_cols, default=df_cols[:1] if df_cols else [])
            st.session_state['text_columns'] = text_columns_selected

            st.write("**Additional Filters**")
            selected_additional_cols = st.multiselect("Select additional columns from your dataset to use as filters:", df_cols, default=[])
            st.session_state['additional_filters_selected'] = selected_additional_cols

            filtered_df = df.copy()
            for col_name in selected_additional_cols:
                if f'selected_filter_{col_name}' not in st.session_state:
                    st.session_state[f'selected_filter_{col_name}'] = []
                unique_vals = sorted(df[col_name].dropna().unique().tolist())
                selected_vals = st.multiselect(
                    f"Filter by {col_name}",
                    options=unique_vals,
                    default=st.session_state[f'selected_filter_{col_name}']
                )
                st.session_state[f'selected_filter_{col_name}'] = selected_vals
                if selected_vals:
                    filtered_df = filtered_df[filtered_df[col_name].isin(selected_vals)]

            st.session_state['filtered_df'] = filtered_df
            st.write("Filtered Data Preview:")
            st.write(filtered_df.head())
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

    # Show total count even outside the if-block
    if 'filtered_df' in st.session_state:
        st.write(f"Total number of results: {len(st.session_state['filtered_df'])}")

# Create tabs with callbacks
tab_names = ["Semantic Search", "Clustering", "Summarization", "Help", "Internal Validation"]
if 'active_tab' not in st.session_state:
    st.session_state.active_tab = tab_names[0]

# Function to set the active tab
def set_active_tab(tab_name):
    if tab_name in tab_names:
        st.session_state.active_tab = tab_name

# Create the tabs
tabs = st.tabs(tab_names)

# Map tabs to their indices
tab_indices = {name: i for i, name in enumerate(tab_names)}

# Help Tab
with tabs[3]:
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
       - In the "Clustering" tab, choose whether to cluster the full dataset, the filtered dataset, or the semantic search results.
       - Adjust `min_cluster_size` as desired.
       - Run clustering to get topics and visualizations.
    6. **Summarization**:
       - If you have clusters, you can select which clusters to summarize.
       - Choose temperature and max_tokens for the LLM.
       - Optionally enable references to original sources.
       - Generate summaries for the entire selection and/or per cluster.

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
with tabs[0]:
    st.header("Semantic Search")
    if 'filtered_df' in st.session_state and st.session_state['filtered_df'] is not None:
        if not st.session_state['filtered_df'].empty:
            with st.expander("ℹ️ How Semantic Search Works"):
                st.markdown("""
                ### Understanding Semantic Search

                Unlike traditional keyword search that looks for exact matches, semantic search understands the meaning and context of your query:
                - **Query Processing**: We embed your query into a high-dimensional vector capturing its meaning.
                - **Similarity Matching**: We compare your query vector to each document's vector (computed from the selected text columns). Documents with higher similarity are more relevant.
                - **Threshold**: Adjusting the threshold changes how strict the match must be.
                - **Include / Exclude Keywords**: You can enforce must-have or exclude terms after the semantic similarity step.
                """)

            df = st.session_state['df']
            filtered_df = st.session_state['filtered_df']

            text_columns = st.session_state.get('text_columns', [])
            if not text_columns:
                st.warning("No text columns selected. Please select at least one column for text embedding in the main view.")
            else:
                # Compute or load embeddings if needed
                if ('embeddings' not in st.session_state) or (st.session_state.get('last_text_columns') != text_columns):
                    embeddings, embeddings_file = load_or_compute_embeddings(df, st.session_state.get('using_default_dataset', False), st.session_state.get('uploaded_file_name', None), text_columns)
                else:
                    embeddings = st.session_state['embeddings']

                if embeddings is not None:
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
                                df_fill = filtered_df.fillna("")
                                search_texts = df_fill[text_columns].agg(' '.join, axis=1).tolist()
                                query_embedding = model.encode([query], device=device)
                                filtered_indices = filtered_df.index
                                filtered_embeddings = embeddings[filtered_indices]
                                similarities = cosine_similarity(query_embedding, filtered_embeddings)

                                # Plot histogram of similarity
                                fig = px.histogram(
                                    x=similarities[0],
                                    nbins=30,
                                    labels={'x': 'Similarity Score', 'y': 'Number of Documents'},
                                    title='Distribution of Similarity Scores'
                                )
                                fig.add_vline(
                                    x=similarity_threshold,
                                    line_dash="dash",
                                    line_color="red",
                                    annotation_text=f"Threshold: {similarity_threshold:.2f}",
                                    annotation_position="top"
                                )
                                fig.update_layout(
                                    title_x=0.5,
                                    showlegend=False,
                                    margin=dict(t=50, l=50, r=50, b=50),
                                    hoverlabel=dict(bgcolor="black", font_size=14, font_color="white"),
                                    hovermode='x',
                                    xaxis=dict(showgrid=False, zeroline=False),
                                    yaxis=dict(showgrid=False, zeroline=False)
                                )
                                fig.update_traces(
                                    hovertemplate="Similarity Score: %{x:.3f}<br>Count: %{y}",
                                    marker_line_width=1,
                                    marker_line_color="rgb(150,150,150)",
                                    opacity=0.8
                                )

                                st.write("### Similarity Score Distribution")
                                st.plotly_chart(fig)

                                above_threshold_indices = np.where(similarities[0] > similarity_threshold)[0]
                                if len(above_threshold_indices) == 0:
                                    st.warning("No results found above the similarity threshold.")
                                    if 'search_results' in st.session_state:
                                        del st.session_state['search_results']
                                    if 'search_results_processed_data' in st.session_state:
                                        del st.session_state['search_results_processed_data']
                                else:
                                    selected_indices = filtered_indices[above_threshold_indices]
                                    results = filtered_df.loc[selected_indices].copy()
                                    results['similarity_score'] = similarities[0][above_threshold_indices]
                                    results = results.sort_values(by='similarity_score', ascending=False)

                                    # Apply include keywords (post-search filter)
                                    if include_keywords.strip():
                                        inc_words = [w.strip().lower() for w in include_keywords.split(',') if w.strip()]
                                        if inc_words:
                                            results = results[
                                                results.apply(
                                                    lambda row: all(w in (' '.join(row.astype(str)).lower()) for w in inc_words),
                                                    axis=1
                                                )
                                            ]

                                    if results.empty:
                                        st.warning("No results found after applying keyword filters.")
                                        if 'search_results' in st.session_state:
                                            del st.session_state['search_results']
                                        if 'search_results_processed_data' in st.session_state:
                                            del st.session_state['search_results_processed_data']
                                    else:
                                        st.session_state['search_results'] = results.copy()
                                        output = io.BytesIO()
                                        writer = pd.ExcelWriter(output, engine='openpyxl')
                                        results.to_excel(writer, index=False)
                                        writer.close()
                                        processed_data = output.getvalue()
                                        st.session_state['search_results_processed_data'] = processed_data
                        else:
                            st.warning("Please enter a query to search.")

                    if 'search_results' in st.session_state and not st.session_state['search_results'].empty:
                        st.write("Search Results:")
                        results = st.session_state['search_results']
                        # Show 'similarity_score' last
                        columns_to_display = [c for c in results.columns if c != 'similarity_score'] + ['similarity_score']
                        st.write(results[columns_to_display])
                        st.write(f"Total number of results: {len(results)}")

                        processed_data = st.session_state.get('search_results_processed_data')
                        if processed_data:
                            st.download_button(
                                label="Download Full Results",
                                data=processed_data,
                                file_name='search_results.xlsx',
                                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                                key='download_search_results'
                            )
                    else:
                        st.info("No search results to display. Enter a query and click 'Search'.")

                else:
                    st.warning("No embeddings available because no text columns were chosen.")
        else:
            st.warning("The filtered dataset is empty. Please adjust your filters.")
    else:
        st.warning("Please select a dataset to proceed and select text columns.")

# Clustering Tab
with tabs[1]:
    st.header("Clustering")
    if 'filtered_df' in st.session_state and st.session_state['filtered_df'] is not None:
        if not st.session_state['filtered_df'].empty:
            with st.expander("ℹ️ How Clustering Works"):
                st.markdown("""
                Clustering automatically groups similar documents together based on their embeddings:
                - **HDBSCAN** finds clusters of documents with similar vector representations.
                - **Min Cluster Size** determines how many documents are needed to form a cluster.
                - Outliers that don't fit well into any cluster are labeled `-1`.
                """)

            col1, col2 = st.columns(2)
            with col1:
                with st.form("clustering_parameters"):
                    # Current version has three options
                    clustering_option = st.radio(
                        "Select data for clustering:",
                        ('Full Dataset', 'Filtered Dataset', 'Semantic Search Results')
                    )
                    if 'min_cluster_size' not in st.session_state:
                        st.session_state.min_cluster_size = 5
                    min_cluster_size_val = st.slider(
                        "Min Cluster Size",
                        min_value=2,
                        max_value=50,
                        value=st.session_state.min_cluster_size,
                        help="Minimum size of each cluster in HDBSCAN. Documents that can't form at least this many in a cluster become outliers (-1).",
                        key="min_cluster_size"
                    )
                    submitted = st.form_submit_button("Run Clustering")

            if clustering_option == 'Semantic Search Results':
                if st.session_state.get('search_results') is not None and not st.session_state['search_results'].empty:
                    df_to_cluster = st.session_state['search_results'].copy()
                else:
                    st.warning("No search results found. Please perform a semantic search first.")
                    df_to_cluster = None
            elif clustering_option == 'Filtered Dataset':
                if ('filtered_df' in st.session_state 
                        and not st.session_state['filtered_df'].empty 
                        and 'filter_state' in st.session_state 
                        and st.session_state['filter_state']['applied']):
                    df_to_cluster = st.session_state['filtered_df'].copy()
                else:
                    st.warning("No filtered dataset available or filters not applied yet.")
                    df_to_cluster = None
            else:  # Full Dataset
                df_to_cluster = st.session_state['df'].copy()

            if df_to_cluster is not None and not df_to_cluster.empty:
                st.session_state['current_clustering_option'] = clustering_option
                st.session_state['current_clustering_data'] = df_to_cluster.copy()
                text_columns = st.session_state.get('text_columns', [])
                if not text_columns:
                    st.warning("No text columns selected. Please select text columns to embed before clustering.")
                else:
                    # Check embeddings
                    if 'embeddings' not in st.session_state or st.session_state.get('last_text_columns') != text_columns:
                        df_full = st.session_state['df']
                        embeddings, embeddings_file = load_or_compute_embeddings(
                            df_full, 
                            st.session_state.get('using_default_dataset', False),
                            st.session_state.get('uploaded_file_name', None),
                            text_columns
                        )
                    else:
                        embeddings = st.session_state['embeddings']

                    if embeddings is not None:
                        dfc = df_to_cluster.copy()
                        dfc_fill = dfc.fillna("")
                        dfc['text'] = dfc_fill[text_columns].agg(' '.join, axis=1)

                        if len(dfc['text']) == 0:
                            st.warning("No text data available for clustering.")
                        else:
                            stop_words = set(stopwords.words('english'))
                            texts_cleaned = []
                            for text in dfc['text'].tolist():
                                word_tokens = word_tokenize(text)
                                filtered_text = ' '.join([w for w in word_tokens if w.lower() not in stop_words])
                                texts_cleaned.append(filtered_text)

                            selected_indices = dfc.index
                            embeddings_clustering = embeddings[selected_indices]

                            if submitted:
                                if df_to_cluster is not None and not df_to_cluster.empty:
                                    # Ensure all session state variables are initialized
                                    initialize_session_state()
                                    
                                    # Check if we already have clustering results with the same parameters
                                    reuse_existing = False
                                    if ('clustered_data' in st.session_state and 
                                        'clustering_params' in st.session_state and
                                        st.session_state['clustering_params'].get('min_cluster_size') == min_cluster_size_val and
                                        st.session_state['clustering_params'].get('data_hash') == hash(str(df_to_cluster.shape))):
                                        
                                        # We can reuse existing results
                                        reuse_existing = True
                                        clustered_df = st.session_state['clustered_data']
                                        st.success("Using cached clustering results.")
                                    
                                    if not reuse_existing:
                                        with st.spinner("Clustering documents..."):
                                            sentence_model = get_embedding_model()
                                            # HDBSCAN runs on CPU
                                            embeddings_for_clustering = (
                                                embeddings_clustering.cpu().numpy()
                                                if torch.is_tensor(embeddings_clustering)
                                                else embeddings_clustering
                                            )
                                            hdbscan_model = HDBSCAN(
                                                min_cluster_size=min_cluster_size_val, 
                                                metric='euclidean', 
                                                cluster_selection_method='eom',
                                                prediction_data=True
                                            )
                                            topic_model = BERTopic(
                                                embedding_model=sentence_model,
                                                hdbscan_model=hdbscan_model,
                                                calculate_probabilities=True,
                                                verbose=True
                                            )
                                            try:
                                                topics, probs = topic_model.fit_transform(texts_cleaned, embeddings=embeddings_for_clustering)
                                                dfc['Topic'] = topics
                                                st.session_state['topic_model'] = topic_model
                                                st.session_state['clustered_data'] = dfc.copy()
                                                
                                                # Store clustering parameters for caching
                                                st.session_state['clustering_params'] = {
                                                    'min_cluster_size': min_cluster_size_val,
                                                    'data_hash': hash(str(df_to_cluster.shape))
                                                }
                                                st.session_state['min_cluster_size_val'] = min_cluster_size_val
                                                
                                                # Set the active tab to stay on Clustering
                                                set_active_tab("Clustering")
                                                
                                                # Display clustering results
                                                clustered_df = dfc

                                                st.subheader("Topic Overview")
                                                cluster_info = []
                                                unique_topics = sorted(list(set(topics)))
                                                for t in unique_topics:
                                                    cluster_docs = dfc[dfc['Topic'] == t]
                                                    count = len(cluster_docs)
                                                    top_words = topic_model.get_topic(t)
                                                    if top_words:
                                                        top_keywords = ", ".join([w[0] for w in top_words[:5]])
                                                    else:
                                                        top_keywords = "N/A"
                                                    cluster_info.append((t, count, top_keywords))
                                                cluster_df = pd.DataFrame(cluster_info, columns=["Topic", "Count", "Top Keywords"])
                                                st.dataframe(cluster_df)

                                                st.subheader("Clustering Results")
                                                columns_to_display = [col for col in dfc.columns if col != 'text']
                                                st.write(dfc[columns_to_display])

                                                st.write("Visualizing Topics...")
                                                st.subheader("Intertopic Distance Map")
                                                fig1 = topic_model.visualize_topics()
                                                fig1.update_traces(
                                                    hoverlabel=dict(bgcolor='black', font_size=14, font_color='white')
                                                )
                                                st.plotly_chart(fig1)

                                                st.subheader("Topic Document Visualization")
                                                fig2 = topic_model.visualize_documents(texts_cleaned, embeddings=embeddings_clustering)
                                                st.plotly_chart(fig2)

                                                st.subheader("Topic Hierarchy Visualization")
                                                fig3 = topic_model.visualize_hierarchy()
                                                st.plotly_chart(fig3)

                                                st.write("Computing Hierarchical Topics...")
                                                hierarchy = topic_model.hierarchical_topics(texts_cleaned)
                                                st.session_state['hierarchy'] = hierarchy if hierarchy is not None else pd.DataFrame()

                                                st.subheader("Hierarchical Topic Treemap")
                                                hierarchy = st.session_state['hierarchy']
                                                if hierarchy is not None and not hierarchy.empty:
                                                    parent_dict = {row.Parent_Name: row for _, row in hierarchy.iterrows()}
                                                    root_row = hierarchy.iloc[hierarchy['Parent_ID'].argmax()]
                                                    root_name = root_row.Parent_Name
                                                    all_topics = root_row['Topics']
                                                    root_size = len(all_topics)

                                                    treemap_nodes = [{"names": "All Topics", "parents": "", "values": root_size}]

                                                    def build_nodes(name, parent_name):
                                                        if name in parent_dict:
                                                            row = parent_dict[name]
                                                            node_topics = row['Topics']
                                                            node_size = len(node_topics)
                                                            treemap_nodes.append({
                                                                "names": name,
                                                                "parents": parent_name,
                                                                "values": node_size
                                                            })
                                                            left_child = row['Child_Left_Name']
                                                            right_child = row['Child_Right_Name']
                                                            build_nodes(left_child, name)
                                                            build_nodes(right_child, name)
                                                        else:
                                                            treemap_nodes.append({
                                                                "names": name,
                                                                "parents": parent_name,
                                                                "values": 1
                                                            })

                                                    build_nodes(root_name, "All Topics")
                                                    treemap_df = pd.DataFrame(treemap_nodes)
                                                    fig_treemap = px.treemap(treemap_df, names='names', parents='parents', values='values')
                                                    fig_treemap.update_traces(root_color="lightgrey")
                                                    fig_treemap.update_layout(margin=dict(t=50, l=25, r=25, b=25))
                                                    st.plotly_chart(fig_treemap)
                                                else:
                                                    st.warning("No hierarchical topic information available for Treemap.")

                                            except Exception as e:
                                                st.error(f"An error occurred during clustering: {e}")
                                                st.session_state['clustering_error'] = str(e)
                    else:
                        st.warning("No embeddings available. Please select text columns and ensure embeddings are computed.")
        else:
            st.warning("No data available for clustering.")
    else:
        st.warning("Please select a dataset to proceed and select text columns.")

# Summarization Tab
with tabs[2]:
    st.header("Summarization")
    if 'filtered_df' in st.session_state and st.session_state['filtered_df'] is not None:
        if not st.session_state['filtered_df'].empty:
            # Determine final DataFrame to summarize
            if 'clustered_data' in st.session_state and not st.session_state['clustered_data'].empty:
                df_summ = st.session_state['clustered_data']
            else:
                df_summ = st.session_state['filtered_df']

            if df_summ is not None and not df_summ.empty:
                text_columns = st.session_state.get('text_columns', [])
                if not text_columns:
                    st.warning("No text columns selected. Please select text columns before summarization.")
                else:
                    # Only proceed if we have topics (because we do cluster-based summarization)
                    if 'Topic' in df_summ.columns and 'topic_model' in st.session_state:
                        topic_model = st.session_state['topic_model']
                        # Fill in the combined text
                        df_summ_fill = df_summ.fillna("")
                        df_summ['text'] = df_summ_fill[text_columns].agg(' '.join, axis=1)

                        topics = df_summ['Topic'].unique()
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
                            # We allow summarizing "All clusters" or "Specific clusters"
                            if 'summary_scope' not in st.session_state:
                                st.session_state.summary_scope = "All clusters"

                            st.session_state.summary_scope = st.radio(
                                "Generate summaries for:",
                                ["All clusters", "Specific clusters"]
                            )
                            
                            ############################################################################
                            # Enhanced Summary Options (references) - MOVED OUTSIDE THE FORM
                            ############################################################################
                            st.write("### Enhanced Summary Options")
                            # All columns except text / Topic / similarity_score are potential references
                            all_cols = df_summ.columns.tolist()
                            ignore_cols = ['text', 'Topic', 'similarity_score']
                            filtered_cols = [c for c in all_cols if c not in ignore_cols]

                            # Checkbox to enable references - default to True
                            st.session_state.enable_references = st.checkbox(
                                "Enable references in summaries",
                                value=True,
                                help="Include source document references in the summary"
                            )
                            
                            # Reference column selection
                            if filtered_cols:
                                ref_col_index = 0
                                if st.session_state.reference_id_column in filtered_cols:
                                    ref_col_index = filtered_cols.index(st.session_state.reference_id_column)
                                else:
                                    # Try to find a good default column (ID, Title, etc.)
                                    for i, col in enumerate(filtered_cols):
                                        if any(keyword in col.lower() for keyword in ['id', 'title', 'name']):
                                            ref_col_index = i
                                            break

                                reference_id_column = st.selectbox(
                                    "Select column to use for reference IDs:",
                                    filtered_cols,
                                    index=ref_col_index if ref_col_index < len(filtered_cols) else 0
                                )

                                st.session_state.reference_id_column = reference_id_column
                            else:
                                reference_id_column = None
                                st.warning("No suitable columns available to serve as reference IDs.")
                            
                            # URL column logic
                            has_url_column = st.checkbox(
                                "Add hyperlinks to references",
                                value=True
                            )
                            st.session_state.has_url_column = has_url_column

                            if has_url_column and filtered_cols:
                                # Filter likely URL columns
                                possible_url_cols = [c for c in filtered_cols if 'url' in c.lower() or 'link' in c.lower()]
                                if not possible_url_cols:
                                    possible_url_cols = filtered_cols  # fallback if no "url/link" columns
                                url_col_index = 0
                                if st.session_state.url_column in possible_url_cols:
                                    url_col_index = possible_url_cols.index(st.session_state.url_column)
                                
                                url_column = st.selectbox(
                                    "Select column containing URLs:",
                                    possible_url_cols,
                                    index=url_col_index if url_col_index < len(possible_url_cols) else 0
                                )
                                
                                st.session_state.url_column = url_column
                            else:
                                url_column = None
                                st.session_state.url_column = None
                            ############################################################################

                            with st.form("summarization_parameters"):
                                topic_options = [int(t) for t in cluster_df["Topic"].tolist()]
                                
                                if st.session_state.summary_scope == "Specific clusters":
                                    selected_topics = st.multiselect("Select clusters to summarize", topic_options)
                                else:
                                    selected_topics = topic_options

                                temperature = st.slider("Summarization Temperature", 0.0, 1.0, 0.7)
                                max_tokens = st.slider("Max Tokens for Summarization", 100, 3000, 1000)

                                submit_summary = st.form_submit_button("Generate Summaries")

                        if submit_summary:
                            # Set the active tab to stay on Summarization
                            set_active_tab("Summarization")
                            
                            # Ensure all session state variables are initialized
                            initialize_session_state()
                            
                            # Ensure all required session state variables are available
                            required_vars = ['enable_references', 'reference_id_column', 'has_url_column', 'url_column']
                            missing_vars = [var for var in required_vars if var not in st.session_state]
                            
                            if missing_vars:
                                for var in missing_vars:
                                    if var == 'enable_references':
                                        st.session_state.enable_references = True
                                    elif var == 'reference_id_column':
                                        st.session_state.reference_id_column = None
                                    elif var == 'has_url_column':
                                        st.session_state.has_url_column = True
                                    elif var == 'url_column':
                                        st.session_state.url_column = None
                                st.warning("Some session state variables were missing and have been initialized.")
                            
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
                                # Check if we can reuse existing summaries
                                reuse_summaries = False
                                if ('summary_results' in st.session_state and 
                                    'summary_params' in st.session_state and
                                    st.session_state['summary_params'].get('temperature') == temperature and
                                    st.session_state['summary_params'].get('max_tokens') == max_tokens and
                                    str(st.session_state['summary_params'].get('selected_topics')) == str(selected_topics)):
                                    
                                    # Ensure all required session state variables are available
                                    if 'enable_references' not in st.session_state:
                                        st.session_state.enable_references = True
                                    if 'reference_id_column' not in st.session_state:
                                        st.session_state.reference_id_column = None
                                    if 'has_url_column' not in st.session_state:
                                        st.session_state.has_url_column = True
                                    if 'url_column' not in st.session_state:
                                        st.session_state.url_column = None
                                    
                                    st.success("Using cached summary results.")
                                    high_level_summary = st.session_state.get('high_level_summary', '')
                                    summaries = st.session_state.get('summary_results', [])
                                    
                                    # Display the cached high-level summary
                                    st.write("### High-Level Summary:")
                                    st.write(high_level_summary)
                                    
                                    # Display the cached cluster-specific summaries
                                    if summaries:
                                        st.write("### Summaries per Selected Cluster")
                                        for summary_data in summaries:
                                            topic_val = summary_data['Topic']
                                            st.write(f"#### Cluster {topic_val}")
                                            
                                            # Check if there was an error
                                            if 'Error' in summary_data:
                                                st.warning(f"Warning: {summary_data.get('Error')}")
                                            
                                            # Display the summary
                                            if 'Enhanced_Summary' in summary_data:
                                                st.markdown(summary_data['Enhanced_Summary'], unsafe_allow_html=True)
                                            else:
                                                st.write(summary_data['Summary'])
                                        
                                        # Add download button for summaries
                                        summary_df = pd.DataFrame(summaries)
                                        # Remove HTML content for CSV download
                                        if 'Enhanced_Summary' in summary_df.columns:
                                            download_df = summary_df[['Topic', 'Summary']]
                                        else:
                                            download_df = summary_df
                                        
                                        # Create CSV download
                                        csv = download_df.to_csv(index=False)
                                        st.download_button(
                                            label="Download Summaries as CSV",
                                            data=csv,
                                            file_name="cluster_summaries.csv",
                                            mime="text/csv"
                                        )
                                        
                                        # Store the summary results for caching
                                        st.session_state['summary_results'] = summaries
                                        st.session_state['summary_params'] = {
                                            'temperature': temperature,
                                            'max_tokens': max_tokens,
                                            'selected_topics': selected_topics,
                                            'enable_references': st.session_state.enable_references,
                                            'reference_id_column': st.session_state.reference_id_column,
                                            'has_url_column': st.session_state.has_url_column,
                                            'url_column': st.session_state.url_column
                                        }
                                    
                                    # Skip the rest of the summarization process
                                    reuse_summaries = True
                                
                                if not reuse_summaries:
                                    llm = ChatOpenAI(api_key=openai_api_key, model_name='gpt-4o', temperature=temperature, max_tokens=max_tokens)

                                    if selected_topics:
                                        # Convert selected topics back to float for filtering
                                        selected_topics_float = [float(t) for t in selected_topics]
                                        df_to_summarize = df_summ[df_summ['Topic'].isin(selected_topics_float)]
                                    else:
                                        df_to_summarize = df_summ

                                    with st.spinner("Generating high-level summary..."):
                                        # Generate high-level summary
                                        docs_text = " ".join(df_to_summarize['text'].tolist())
                                        user_prompt = f"**Text to summarize**: {docs_text}"
                                        system_message = SystemMessagePromptTemplate.from_template(system_prompt)
                                        human_message = HumanMessagePromptTemplate.from_template("{user_prompt}")
                                        chat_prompt = ChatPromptTemplate.from_messages([system_message, human_message])

                                        chain = LLMChain(llm=llm, prompt=chat_prompt)
                                        response = chain.run(user_prompt=user_prompt)
                                        high_level_summary = response.strip()

                                        # Store the summary parameters for caching
                                        st.session_state['summary_params'] = {
                                            'temperature': temperature,
                                            'max_tokens': max_tokens,
                                            'selected_topics': selected_topics,
                                            'enable_references': st.session_state.enable_references,
                                            'reference_id_column': st.session_state.reference_id_column
                                        }
                                        st.session_state['high_level_summary'] = high_level_summary

                                    # Display high-level summary with references if enabled
                                    if st.session_state.enable_references and st.session_state.reference_id_column:
                                        with st.spinner("Adding references to high-level summary..."):
                                            enhanced_summary = add_references_to_summary(
                                                high_level_summary,
                                                df_to_summarize,
                                                st.session_state.reference_id_column,
                                                st.session_state.url_column if st.session_state.has_url_column else None,
                                                llm
                                            )
                                        st.write("### High-Level Summary (with references):")
                                        st.markdown(enhanced_summary, unsafe_allow_html=True)
                                        
                                        # Also provide a way to view the original summary
                                        with st.expander("View original summary (without references)"):
                                            st.write(high_level_summary)
                                    else:
                                        st.write("### High-Level Summary:")
                                        st.write(high_level_summary)

                                    # Summaries per cluster if multiple clusters
                                    if len(selected_topics) > 1:
                                        grouped_list = list(df_to_summarize.groupby('Topic'))
                                    else:
                                        grouped_list = list(df_to_summarize.groupby('Topic')) if selected_topics else []

                                    if grouped_list:
                                        st.write("### Summaries per Selected Cluster")
                                        progress_bar = st.progress(0)
                                        total_clusters = len(grouped_list)
                                        summaries = []
                                        
                                        # Ensure all required session state variables are available
                                        if 'enable_references' not in st.session_state:
                                            st.session_state.enable_references = True
                                        if 'reference_id_column' not in st.session_state:
                                            st.session_state.reference_id_column = None
                                        if 'has_url_column' not in st.session_state:
                                            st.session_state.has_url_column = True
                                        if 'url_column' not in st.session_state:
                                            st.session_state.url_column = None
                                                
                                        # Get the values to pass to the worker function
                                        enable_refs = st.session_state.enable_references
                                        ref_id_col = st.session_state.reference_id_column
                                        has_url_col = st.session_state.has_url_column
                                        url_col = st.session_state.url_column

                                        def generate_summary_per_topic(topic_group_tuple, enable_refs, ref_id_col, has_url_col, url_col):
                                            topic_val, group_df = topic_group_tuple
                                            try:
                                                # Ensure we have text data to summarize
                                                if 'text' not in group_df.columns or group_df['text'].empty:
                                                    return {
                                                        'Topic': topic_val,
                                                        'Summary': "No text data available for summarization.",
                                                        'Error': "Missing text data"
                                                    }
                                                
                                                # Limit text length to avoid token limits
                                                docs_text = " ".join(group_df['text'].astype(str).tolist())
                                                # Truncate if too long (rough estimate to avoid token limits)
                                                if len(docs_text) > 15000:
                                                    docs_text = docs_text[:15000] + "..."
                                                    
                                                user_prompt_local = f"**Text to summarize**: {docs_text}"
                                                system_message_local = SystemMessagePromptTemplate.from_template(system_prompt)
                                                human_message_local = HumanMessagePromptTemplate.from_template("{user_prompt}")
                                                chat_prompt_local = ChatPromptTemplate.from_messages([system_message_local, human_message_local])

                                                local_chain = LLMChain(llm=llm, prompt=chat_prompt_local)
                                                response_local = local_chain.run(user_prompt=user_prompt_local)
                                                summary_local = response_local.strip()

                                                # If references are enabled, create an enhanced summary
                                                if enable_refs and ref_id_col:
                                                    try:
                                                        summary_with_refs = add_references_to_summary(
                                                            summary_local,
                                                            group_df,
                                                            ref_id_col,
                                                            url_col if has_url_col else None,
                                                            llm
                                                        )
                                                        return {
                                                            'Topic': topic_val,
                                                            'Summary': summary_local,
                                                            'Enhanced_Summary': summary_with_refs
                                                        }
                                                    except Exception as e:
                                                        # If reference enhancement fails, return the basic summary
                                                        return {
                                                            'Topic': topic_val,
                                                            'Summary': summary_local,
                                                            'Error': f"Reference enhancement failed: {str(e)}"
                                                        }
                                                else:
                                                    return {
                                                        'Topic': topic_val,
                                                        'Summary': summary_local
                                                    }
                                            except Exception as e:
                                                # Handle any errors in the summarization process
                                                return {
                                                    'Topic': topic_val,
                                                    'Summary': f"Error generating summary: {str(e)}",
                                                    'Error': str(e)
                                                }

                                        with st.spinner("Summarizing each selected cluster..."):
                                            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                                                futures = {
                                                    executor.submit(generate_summary_per_topic, item, enable_refs, ref_id_col, has_url_col, url_col): item[0]
                                                    for item in grouped_list
                                                }
                                                for future in concurrent.futures.as_completed(futures):
                                                    result = future.result()
                                                    summaries.append(result)
                                                    progress_bar.progress((len(summaries)) / total_clusters)

                                        # Sort summaries by topic number
                                        summaries.sort(key=lambda x: x['Topic'])

                                        # Display summaries
                                        for summary_data in summaries:
                                            topic_val = summary_data['Topic']
                                            st.write(f"#### Cluster {topic_val}")
                                            
                                            # Check if there was an error
                                            if 'Error' in summary_data:
                                                st.warning(f"Warning: {summary_data.get('Error')}")
                                            
                                            # Display the summary
                                            if 'Enhanced_Summary' in summary_data:
                                                st.markdown(summary_data['Enhanced_Summary'], unsafe_allow_html=True)
                                            else:
                                                st.write(summary_data['Summary'])
                                        
                                        # Add download button for summaries
                                        if summaries:
                                            # Create a DataFrame for download
                                            summary_df = pd.DataFrame(summaries)
                                            # Remove HTML content for CSV download
                                            if 'Enhanced_Summary' in summary_df.columns:
                                                download_df = summary_df[['Topic', 'Summary']]
                                            else:
                                                download_df = summary_df
                                            
                                            # Create CSV download
                                            csv = download_df.to_csv(index=False)
                                            st.download_button(
                                                label="Download Summaries as CSV",
                                                data=csv,
                                                file_name="cluster_summaries.csv",
                                                mime="text/csv"
                                            )
                                            
                                            # Store the summary results for caching
                                            st.session_state['summary_results'] = summaries
                                            st.session_state['summary_params'] = {
                                                'temperature': temperature,
                                                'max_tokens': max_tokens,
                                                'selected_topics': selected_topics,
                                                'enable_references': st.session_state.enable_references,
                                                'reference_id_column': st.session_state.reference_id_column,
                                                'has_url_column': st.session_state.has_url_column,
                                                'url_column': st.session_state.url_column
                                            }
                    else:
                        st.warning("Please perform clustering first to generate topics.")
            else:
                st.warning("No data available for summarization.")
        else:
            st.warning("The filtered dataset is empty. Please adjust your filters.")
    else:
        st.warning("Please select a dataset and select text columns to proceed.")

# Internal Validation Tab
with tabs[4]:
    st.header("Internal Validation & Debugging")

    hierarchy = st.session_state.get('hierarchy', pd.DataFrame())
    st.subheader("Hierarchical Topics DataFrame")
    if not hierarchy.empty:
        st.dataframe(hierarchy)
    else:
        st.write("No hierarchical topics data available.")

    error_msg = st.session_state.get('clustering_error', None)
    if error_msg:
        st.subheader("Clustering Error Details")
        st.write(error_msg)

    st.subheader("Internal Variables")
    st.write("Min Cluster Size:", st.session_state.get('min_cluster_size_val', 'Not set'))
    st.write("Text Columns:", st.session_state.get('text_columns', 'Not set'))
    st.write("Using Default Dataset:", st.session_state.get('using_default_dataset', 'Not set'))
    st.write("Last Text Columns Used for Embeddings:", st.session_state.get('last_text_columns', 'Not set'))

    filtered_df_debug = st.session_state.get('filtered_df', pd.DataFrame())
    st.subheader("Filtered DF")
    st.write(filtered_df_debug.head())

    clustered_data_debug = st.session_state.get('clustered_data', pd.DataFrame())
    st.subheader("Clustered Data")
    st.write(clustered_data_debug.head())

    if 'embeddings' in st.session_state and st.session_state['embeddings'] is not None:
        st.subheader("Embeddings Information")
        st.write("Shape of Embeddings:", st.session_state['embeddings'].shape)
    else:
        st.write("No embeddings data available.")