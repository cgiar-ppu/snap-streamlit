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
from typing import List
import plotly.express as px
import torch

# For parallelism (if needed)
# from concurrent.futures import ThreadPoolExecutor

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

###############################################################################
# Helper: Attempt to get this file's directory or fallback to current working dir
###############################################################################
def get_base_dir():
    try:
        base_dir = os.path.dirname(__file__)
        if not base_dir:
            return os.getcwd()
        return base_dir
    except NameError:
        # In case __file__ is not defined (some environments)
        return os.getcwd()

BASE_DIR = get_base_dir()

###############################################################################
# NLTK Resource Initialization
###############################################################################
def init_nltk_resources():
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

init_nltk_resources()

###############################################################################
# Function: add_references_to_summary
###############################################################################
def add_references_to_summary(summary, source_df, reference_column, url_column=None, llm=None):
    """
    Add references to a summary by identifying which parts of the summary come 
    from which source documents. References will be appended as [ID], 
    optionally linked if a URL column is provided.

    Args:
        summary (str): The summary text to enhance with references.
        source_df (DataFrame): DataFrame containing the source documents.
        reference_column (str): Column name to use for reference IDs.
        url_column (str, optional): Column name containing URLs for hyperlinks.
        llm (LLM, optional): Language model for source attribution.
    Returns:
        str: Enhanced summary with references as HTML if possible.
    """
    if summary.strip() == "" or source_df.empty or reference_column not in source_df.columns:
        return summary
    
    # If no LLM is provided, we can't do source attribution
    if llm is None:
        return summary
    
    # Split the summary into sentences for processing
    sentences = re.split(r'(?<=[.!?])\s+', summary)

    # Prepare source texts with their reference IDs
    source_texts = []
    reference_ids = []
    urls = []
    for _, row in source_df.iterrows():
        if 'text' in row and pd.notna(row['text']) and pd.notna(row[reference_column]):
            source_texts.append(str(row['text']))
            reference_ids.append(str(row[reference_column]))
            if url_column and url_column in row and pd.notna(row[url_column]):
                urls.append(str(row[url_column]))
            else:
                urls.append(None)
    if not source_texts:
        return summary

    # Create a mapping between URLs and reference IDs
    url_map = {}
    for ref_id, u in zip(reference_ids, urls):
        if u:
            url_map[ref_id] = u

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
    batch_size = 3  # smaller batch to reduce token usage

    # Process sentences in small batches
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i+batch_size]
        batch_results = []

        for sentence in batch:
            sent_str = sentence.strip()
            if sent_str:
                # Create the prompt for this sentence
                user_prompt = f"""
                Sentence: {sent_str}

                Source texts:
                {'\n'.join([f"ID: {ref_id}, Text: {text[:500]}..." for ref_id, text in zip(reference_ids, source_texts)])}

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
                except Exception:
                    # If there's an error, just use the sentence without attribution
                    batch_results.append((sentence, ""))
            else:
                batch_results.append((sentence, ""))

        # Turn each sentence into an enhanced sentence
        for sentence, source_ids in batch_results:
            if source_ids:
                ids = [id_.strip() for id_ in source_ids.split(',') if id_.strip()]
                ref_parts = []
                for id_ in ids:
                    # If there's a URL for that reference ID, make it clickable
                    if id_ in url_map:
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


###############################################################################
# Device / GPU Info
###############################################################################
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    st.sidebar.success(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    st.sidebar.info("Using CPU")

###############################################################################
# Load or Compute Embeddings
###############################################################################
@st.cache_resource
def get_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2').to(device)

def generate_embeddings(texts, model):
    with st.spinner('Calculating embeddings...'):
        embeddings = model.encode(texts, show_progress_bar=True, device=device)
    return embeddings

@st.cache_data
def load_default_dataset(default_dataset_path):
    if os.path.exists(default_dataset_path):
        df_ = pd.read_excel(default_dataset_path)
        return df_
    else:
        st.error("Default dataset not found. Please ensure the file exists in the 'input' directory.")
        return None

@st.cache_data
def load_uploaded_dataset(uploaded_file):
    df_ = pd.read_excel(uploaded_file)
    return df_

def load_or_compute_embeddings(df, using_default_dataset, uploaded_file_name=None, text_columns=None):
    """
    Loads pre-computed embeddings from a pickle file if they match current data, 
    otherwise computes and caches them.
    """
    if not text_columns:
        return None, None

    base_name = "PRMS_2022_2023_2024_Batch1_QAed" if using_default_dataset else "custom_dataset"
    if uploaded_file_name:
        base_name = os.path.splitext(uploaded_file_name)[0]

    cols_key = "_".join(sorted(text_columns))
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    embeddings_dir = BASE_DIR
    if using_default_dataset:
        embeddings_file = os.path.join(embeddings_dir, f'{base_name}_{cols_key}.pkl')
    else:
        # For custom dataset, we still try to avoid regenerating each time
        embeddings_file = os.path.join(embeddings_dir, f"{base_name}_{cols_key}.pkl")

    df_fill = df.fillna("")
    texts = df_fill[text_columns].astype(str).agg(' '.join, axis=1).tolist()

    # If already in session_state with matching columns and length, reuse
    if ('embeddings' in st.session_state 
        and 'last_text_columns' in st.session_state 
        and st.session_state['last_text_columns'] == text_columns 
        and len(st.session_state['embeddings']) == len(texts)):
        return st.session_state['embeddings'], st.session_state.get('embeddings_file', None)

    # Try to load from disk
    if os.path.exists(embeddings_file):
        with open(embeddings_file, 'rb') as f:
            embeddings = pickle.load(f)
        if len(embeddings) == len(texts):
            st.write("Loaded pre-calculated embeddings.")
            st.session_state['embeddings'] = embeddings
            st.session_state['embeddings_file'] = embeddings_file
            st.session_state['last_text_columns'] = text_columns
            return embeddings, embeddings_file

    # Otherwise compute
    st.write("Generating embeddings...")
    model = get_embedding_model()
    embeddings = generate_embeddings(texts, model)
    with open(embeddings_file, 'wb') as f:
        pickle.dump(embeddings, f)

    st.session_state['embeddings'] = embeddings
    st.session_state['embeddings_file'] = embeddings_file
    st.session_state['last_text_columns'] = text_columns
    return embeddings, embeddings_file


###############################################################################
# Reset Filter Function
###############################################################################
def reset_filters():
    st.session_state['selected_additional_filters'] = {}


###############################################################################
# Sidebar: Dataset Selection
###############################################################################
st.sidebar.title("Data Selection")
dataset_option = st.sidebar.selectbox('Select Dataset', ('PRMS 2022+2023+2024 (Batch 1) QAed', 'Upload my dataset'))

if 'df' not in st.session_state:
    st.session_state['df'] = pd.DataFrame()
if 'filtered_df' not in st.session_state:
    st.session_state['filtered_df'] = pd.DataFrame()

if dataset_option == 'PRMS 2022+2023+2024 (Batch 1) QAed':
    default_dataset_path = os.path.join(BASE_DIR, 'input', 'export_data_table_results_20250303_042559CET.xlsx')
    df = load_default_dataset(default_dataset_path)
    if df is not None:
        st.session_state['df'] = df.copy()
        st.session_state['using_default_dataset'] = True
        st.write("Using default dataset:")

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
                # Reset removed columns
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
        default_text_cols = []
        if 'Title' in df.columns and 'Description' in df.columns:
            default_text_cols = ['Title', 'Description']

        text_columns_selected = st.multiselect(
            "Text Columns:",
            df_cols,
            default=default_text_cols,
            help="Choose columns containing text for semantic search and clustering. "
                 "If multiple are selected, their text will be concatenated."
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
            for k in ['clustered_data', 'topic_model', 'current_clustering_data',
                      'current_clustering_option', 'hierarchy']:
                if k in st.session_state:
                    del st.session_state[k]

        elif 'filter_state' in st.session_state and st.session_state['filter_state']['applied']:
            # Reapply stored filters
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
    # Upload custom dataset
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
            text_columns_selected = st.multiselect(
                "Text Columns:", 
                df_cols, 
                default=df_cols[:1] if df_cols else []
            )
            st.session_state['text_columns'] = text_columns_selected

            st.write("**Additional Filters**")
            selected_additional_cols = st.multiselect(
                "Select additional columns from your dataset to use as filters:", 
                df_cols, 
                default=[]
            )
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

    if 'filtered_df' in st.session_state:
        st.write(f"Total number of results: {len(st.session_state['filtered_df'])}")


###############################################################################
# Preserve active tab across reruns
###############################################################################
if 'active_tab_index' not in st.session_state:
    st.session_state.active_tab_index = 0

tabs_titles = ["Semantic Search", "Clustering", "Summarization", "Help", "Internal Validation"]
tabs = st.tabs(tabs_titles)
# We just create these references so we can navigate more easily
tab_semantic, tab_clustering, tab_summarization, tab_help, tab_internal = tabs

###############################################################################
# Tab: Help
###############################################################################
with tab_help:
    st.header("Help")
    st.markdown("""
    ### About SNAP

    SNAP allows you to explore, filter, search, cluster, and summarize textual datasets.

    **Workflow**:
    1. **Data Selection (Sidebar)**: Choose the default dataset or upload your own.
    2. **Filtering**: Set additional filters for your dataset.
    3. **Select Text Columns**: Which columns to embed.
    4. **Semantic Search** (Tab): Provide a query and threshold to find relevant documents.
    5. **Clustering** (Tab): Group documents into topics.
    6. **Summarization** (Tab): Summarize the clustered documents (with optional references).

    ### Troubleshooting
    - If you see no results, try lowering the similarity threshold or removing negative/required keywords.
    - Ensure you have at least one text column selected for embeddings.
    - Check "Internal Validation" for debugging details.
    """)

###############################################################################
# Tab: Semantic Search
###############################################################################
with tab_semantic:
    st.header("Semantic Search")
    if 'filtered_df' in st.session_state and not st.session_state['filtered_df'].empty:
        text_columns = st.session_state.get('text_columns', [])
        if not text_columns:
            st.warning("No text columns selected. Please select at least one column for text embedding.")
        else:
            df_full = st.session_state['df']
            # Load or compute embeddings if necessary
            embeddings, _ = load_or_compute_embeddings(
                df_full,
                st.session_state.get('using_default_dataset', False),
                st.session_state.get('uploaded_file_name'),
                text_columns
            )

            if embeddings is not None:
                with st.expander("ℹ️ How Semantic Search Works", expanded=False):
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

                with st.form("search_parameters"):
                    query = st.text_input("Enter your search query:")
                    include_keywords = st.text_input("Include only documents containing these words (comma-separated):")
                    similarity_threshold = st.slider("Similarity threshold", 0.0, 1.0, 0.35)
                    submitted = st.form_submit_button("Search")

                if submitted:
                    if query.strip():
                        with st.spinner("Performing Semantic Search..."):
                            model = get_embedding_model()
                            df_filtered = st.session_state['filtered_df'].fillna("")
                            search_texts = df_filtered[text_columns].agg(' '.join, axis=1).tolist()

                            # Filter the embeddings to the same subset
                            subset_indices = df_filtered.index
                            subset_embeddings = embeddings[subset_indices]

                            query_embedding = model.encode([query], device=device)
                            similarities = cosine_similarity(query_embedding, subset_embeddings)[0]

                            # Show distribution
                            fig = px.histogram(
                                x=similarities,
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
                            st.write("### Similarity Score Distribution")
                            st.plotly_chart(fig)

                            above_threshold_indices = np.where(similarities > similarity_threshold)[0]
                            if len(above_threshold_indices) == 0:
                                st.warning("No results found above the similarity threshold.")
                                if 'search_results' in st.session_state:
                                    del st.session_state['search_results']
                            else:
                                selected_indices = subset_indices[above_threshold_indices]
                                results = df_filtered.loc[selected_indices].copy()
                                results['similarity_score'] = similarities[above_threshold_indices]
                                results.sort_values(by='similarity_score', ascending=False, inplace=True)

                                # Include keyword filtering
                                if include_keywords.strip():
                                    inc_words = [w.strip().lower() for w in include_keywords.split(',') if w.strip()]
                                    if inc_words:
                                        results = results[
                                            results.apply(
                                                lambda row: all(
                                                    w in (' '.join(row.astype(str)).lower()) for w in inc_words
                                                ), 
                                                axis=1
                                            )
                                        ]
                                
                                if results.empty:
                                    st.warning("No results found after applying keyword filters.")
                                    if 'search_results' in st.session_state:
                                        del st.session_state['search_results']
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

                # Display search results if available
                if 'search_results' in st.session_state and not st.session_state['search_results'].empty:
                    st.write("## Search Results")
                    results = st.session_state['search_results']
                    cols_to_display = [c for c in results.columns if c != 'similarity_score'] + ['similarity_score']
                    st.write(results[cols_to_display])
                    st.write(f"Total number of results: {len(results)}")

                    if 'search_results_processed_data' in st.session_state:
                        st.download_button(
                            label="Download Full Results",
                            data=st.session_state['search_results_processed_data'],
                            file_name='search_results.xlsx',
                            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                            key='download_search_results'
                        )
                else:
                    st.info("No search results to display. Enter a query and click 'Search'.")
            else:
                st.warning("No embeddings available because no text columns were chosen.")
    else:
        st.warning("Filtered dataset is empty or not loaded. Please adjust your filters or upload data.")


###############################################################################
# Tab: Clustering
###############################################################################
with tab_clustering:
    st.header("Clustering")
    if 'filtered_df' in st.session_state and not st.session_state['filtered_df'].empty:
        # Add explanation about clustering
        with st.expander("ℹ️ How Clustering Works", expanded=False):
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

        df_to_cluster = None
        # UI to pick what data to cluster
        clustering_option = st.radio(
            "Select data for clustering:",
            ('Full Dataset', 'Filtered Dataset', 'Semantic Search Results')
        )

        # Decide which DataFrame is used
        if clustering_option == 'Semantic Search Results':
            if 'search_results' in st.session_state and not st.session_state['search_results'].empty:
                df_to_cluster = st.session_state['search_results'].copy()
            else:
                st.warning("No semantic search results found. Please run a search first.")
        elif clustering_option == 'Filtered Dataset':
            if 'filtered_df' in st.session_state and not st.session_state['filtered_df'].empty:
                df_to_cluster = st.session_state['filtered_df'].copy()
            else:
                st.warning("Filtered dataset is empty. Please check your filters.")
        else:
            if 'df' in st.session_state and not st.session_state['df'].empty:
                df_to_cluster = st.session_state['df'].copy()

        text_columns = st.session_state.get('text_columns', [])
        if not text_columns:
            st.warning("No text columns selected. Please select text columns to embed before clustering.")
        else:
            # Ensure embeddings are available
            df_full = st.session_state['df']
            embeddings, _ = load_or_compute_embeddings(
                df_full,
                st.session_state.get('using_default_dataset', False),
                st.session_state.get('uploaded_file_name'),
                text_columns
            )

            if df_to_cluster is not None and embeddings is not None and not df_to_cluster.empty:
                # Min cluster size
                if 'min_cluster_size' not in st.session_state:
                    st.session_state['min_cluster_size'] = 5

                min_cluster_size_val = st.slider(
                    "Min Cluster Size",
                    min_value=2,
                    max_value=50,
                    value=st.session_state['min_cluster_size'],
                    help="Minimum size of each cluster in HDBSCAN.",
                    key="min_cluster_size"
                )

                if st.button("Run Clustering"):
                    with st.spinner("Performing clustering..."):
                        dfc = df_to_cluster.copy().fillna("")
                        dfc['text'] = dfc[text_columns].astype(str).agg(' '.join, axis=1)

                        # Filter embeddings to those rows
                        selected_indices = dfc.index
                        embeddings_clustering = embeddings[selected_indices]

                        # Basic cleaning
                        stop_words = set(stopwords.words('english'))
                        texts_cleaned = []
                        for text in dfc['text'].tolist():
                            word_tokens = word_tokenize(text)
                            filtered_text = ' '.join([w for w in word_tokens if w.lower() not in stop_words])
                            texts_cleaned.append(filtered_text)

                        try:
                            # Build the HDBSCAN model
                            hdbscan_model = HDBSCAN(
                                min_cluster_size=min_cluster_size_val, 
                                metric='euclidean', 
                                cluster_selection_method='eom'
                            )
                            # Build the BERTopic model
                            topic_model = BERTopic(
                                embedding_model=get_embedding_model(),
                                hdbscan_model=hdbscan_model
                            )
                            # Convert embeddings to CPU numpy if needed
                            if torch.is_tensor(embeddings_clustering):
                                embeddings_for_clustering = embeddings_clustering.cpu().numpy()
                            else:
                                embeddings_for_clustering = embeddings_clustering

                            topics, _ = topic_model.fit_transform(
                                texts_cleaned, 
                                embeddings=embeddings_for_clustering
                            )
                            dfc['Topic'] = topics

                            st.session_state['topic_model'] = topic_model
                            st.session_state['clustered_data'] = dfc.copy()

                            st.subheader("Topic Overview")
                            unique_topics = sorted(list(set(topics)))
                            cluster_info = []
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
                            columns_to_display = [c for c in dfc.columns if c != 'text']
                            st.write(dfc[columns_to_display])

                            # Visualizations
                            st.write("### Intertopic Distance Map")
                            fig1 = topic_model.visualize_topics()
                            st.plotly_chart(fig1)

                            st.write("### Topic Document Visualization")
                            fig2 = topic_model.visualize_documents(texts_cleaned, embeddings=embeddings_for_clustering)
                            st.plotly_chart(fig2)

                            #st.write("Computing Hierarchical Topics...")
                            hierarchy = topic_model.hierarchical_topics(texts_cleaned)
                            st.session_state['hierarchy'] = hierarchy if hierarchy is not None else pd.DataFrame()

                            st.write("### Topic Hierarchy")
                            fig3 = topic_model.visualize_hierarchy()
                            st.plotly_chart(fig3)

                        except Exception as e:
                            st.error(f"An error occurred during clustering: {e}")
                            st.session_state['clustering_error'] = str(e)
            else:
                st.warning("No data available for clustering or embeddings not ready.")
    else:
        st.warning("Please select or upload a dataset and filter as needed.")


###############################################################################
# Tab: Summarization
###############################################################################
with tab_summarization:
    st.header("Summarization")
    # Add explanation about summarization
    with st.expander("ℹ️ How Summarization Works", expanded=False):
        st.markdown("""
        ### Understanding Document Summarization

        Summarization condenses multiple documents into concise, meaningful summaries while preserving key information. Here's how it works:

        1. **Summary Generation**:
           - Documents are processed using advanced language models
           - Key themes and important points are identified
           - Content is condensed while maintaining context
           - Both high-level and cluster-specific summaries are available

        2. **Reference System**:
           - Summaries can include references to source documents
           - References are shown as [ID] or as clickable links
           - Each statement can be traced back to its source
           - Helps maintain accountability and verification

        3. **Types of Summaries**:
           - **High-Level Summary**: Overview of all selected documents
             - Captures main themes across the entire selection
             - Ideal for quick understanding of large document sets
             - Shows relationships between different topics
           
           - **Cluster-Specific Summaries**: Focused on each cluster
             - More detailed for specific themes
             - Shows unique aspects of each cluster
             - Helps understand sub-topics in depth

        ### How to Use Summaries

        1. **Configuration**:
           - Choose between all clusters or specific ones
           - Set temperature for creativity vs. consistency
           - Adjust max tokens for summary length
           - Enable/disable reference system

        2. **Reference Options**:
           - Select column for reference IDs
           - Add hyperlinks to references
           - Choose URL column for clickable links
           - References help track information sources

        3. **Practical Applications**:
           - Quick overview of large datasets
           - Detailed analysis of specific themes
           - Evidence-based reporting with references
           - Compare different document groups

        ### Tips for Better Results

        - **Temperature Setting**:
          - Higher (0.7-1.0): More creative, varied summaries
          - Lower (0.1-0.3): More consistent, conservative summaries
          - Balance based on your needs for creativity vs. consistency

        - **Token Length**:
          - Longer limits: More detailed summaries
          - Shorter limits: More concise, focused summaries
          - Adjust based on document complexity

        - **Reference Usage**:
          - Enable references for traceability
          - Use hyperlinks for easy navigation
          - Choose meaningful reference columns
          - Helps validate summary accuracy

        ### Best Practices

        1. **For General Overview**:
           - Use high-level summary
           - Keep temperature moderate (0.5-0.7)
           - Enable references for verification
           - Focus on broader themes

        2. **For Detailed Analysis**:
           - Use cluster-specific summaries
           - Adjust temperature based on need
           - Include references with hyperlinks
           - Look for patterns within clusters

        3. **For Reporting**:
           - Combine both summary types
           - Use references extensively
           - Balance detail and brevity
           - Ensure source traceability
        """)

    df_summ = None
    # We'll try to summarize either the clustered data or just the filtered dataset
    if 'clustered_data' in st.session_state and not st.session_state['clustered_data'].empty:
        df_summ = st.session_state['clustered_data']
    elif 'filtered_df' in st.session_state and not st.session_state['filtered_df'].empty:
        df_summ = st.session_state['filtered_df']
    else:
        st.warning("No data available for summarization. Please cluster first or have some filtered data.")
    
    if df_summ is not None and not df_summ.empty:
        text_columns = st.session_state.get('text_columns', [])
        if not text_columns:
            st.warning("No text columns selected. Please select columns for text embedding first.")
        else:
            if 'Topic' not in df_summ.columns or 'topic_model' not in st.session_state:
                st.warning("No 'Topic' column found. Summaries per cluster are only available if you've run clustering.")
            else:
                topic_model = st.session_state['topic_model']
                df_summ['text'] = df_summ.fillna("").astype(str)[text_columns].agg(' '.join, axis=1)

                # List of topics
                topics = sorted(df_summ['Topic'].unique())
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
                st.write("Available Clusters:")
                st.dataframe(cluster_df)

                # Summarization settings
                st.subheader("Summarization Settings")
                # Summaries scope
                summary_scope = st.radio(
                    "Generate summaries for:",
                    ["All clusters", "Specific clusters"]
                )
                if summary_scope == "Specific clusters":
                    selected_topics = st.multiselect("Select clusters to summarize", topics)
                else:
                    selected_topics = topics

                # Add system prompt configuration
                default_system_prompt = """You are an expert summarizer skilled in creating concise and relevant summaries.
You will be given text and an objective context. Please produce a clear, cohesive,
and thematically relevant summary. 
Focus on key points, insights, or patterns that emerge from the text."""

                if 'system_prompt' not in st.session_state:
                    st.session_state['system_prompt'] = default_system_prompt

                with st.expander("🔧 Advanced Settings", expanded=False):
                    st.markdown("""
                    ### System Prompt Configuration
                    
                    The system prompt guides the AI in how to generate summaries. You can customize it to better suit your needs:
                    - Be specific about the style and focus you want
                    - Add domain-specific context if needed
                    - Include any special formatting requirements
                    """)
                    
                    system_prompt = st.text_area(
                        "Customize System Prompt",
                        value=st.session_state['system_prompt'],
                        height=150,
                        help="This prompt guides the AI in how to generate summaries. Edit it to customize the summary style and focus."
                    )
                    
                    if st.button("Reset to Default"):
                        system_prompt = default_system_prompt
                        st.session_state['system_prompt'] = default_system_prompt

                    st.markdown("### Generation Parameters")
                    temperature = st.slider(
                        "Temperature",
                        0.0, 1.0, 0.7,
                        help="Higher values (0.7-1.0) make summaries more creative but less predictable. Lower values (0.1-0.3) make them more focused and consistent."
                    )
                    max_tokens = st.slider(
                        "Max Tokens",
                        100, 3000, 1000,
                        help="Maximum length of generated summaries. Higher values allow for more detailed summaries but take longer to generate."
                    )

                st.session_state['system_prompt'] = system_prompt

                st.write("### Enhanced Summary References")
                st.write("Select columns for references (optional).")
                all_cols = [c for c in df_summ.columns if c not in ['text', 'Topic', 'similarity_score']]
                
                # By default, let's guess the first column as reference ID if available
                if 'reference_id_column' not in st.session_state:
                    st.session_state.reference_id_column = all_cols[0] if all_cols else None
                # If there's a column that looks like a URL, guess that
                url_guess = next((c for c in all_cols if 'url' in c.lower() or 'link' in c.lower()), None)
                if 'url_column' not in st.session_state:
                    st.session_state.url_column = url_guess

                enable_references = st.checkbox(
                    "Enable references in summaries", 
                    value=True,  # default to True as requested
                    help="Add source references to the final summary text."
                )
                reference_id_column = st.selectbox(
                    "Select column to use as reference ID:",
                    all_cols,
                    index=all_cols.index(st.session_state.reference_id_column) if st.session_state.reference_id_column in all_cols else 0
                )
                add_hyperlinks = st.checkbox(
                    "Add hyperlinks to references",
                    value=True,  # default to True
                    help="If the reference column has a matching URL, make it clickable."
                )
                url_column = None
                if add_hyperlinks:
                    url_column = st.selectbox(
                        "Select column containing URLs:",
                        all_cols,
                        index=all_cols.index(st.session_state.url_column) if (st.session_state.url_column in all_cols) else 0
                    )

                # Summarization button
                if st.button("Generate Summaries"):
                    openai_api_key = os.environ.get('OPENAI_API_KEY')
                    if not openai_api_key:
                        st.error("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
                    else:
                        llm = ChatOpenAI(
                            api_key=openai_api_key, 
                            model_name='gpt-4o',  # or 'gpt-4' if you have access
                            temperature=temperature, 
                            max_tokens=max_tokens
                        )

                        # Filter to selected topics
                        if selected_topics:
                            df_scope = df_summ[df_summ['Topic'].isin(selected_topics)]
                        else:
                            st.warning("No topics selected for summarization.")
                            df_scope = pd.DataFrame()

                        if df_scope.empty:
                            st.warning("No documents match the selected topics for summarization.")
                        else:
                            all_texts = df_scope['text'].tolist()
                            combined_text = " ".join(all_texts)
                            if not combined_text.strip():
                                st.warning("No text data available for summarization.")
                            else:
                                # Generate High-Level Summary
                                user_prompt = f"**Text to summarize**: {combined_text}"
                                system_message = SystemMessagePromptTemplate.from_template(st.session_state['system_prompt'])
                                human_message = HumanMessagePromptTemplate.from_template("{user_prompt}")
                                chat_prompt = ChatPromptTemplate.from_messages([system_message, human_message])

                                with st.spinner("Generating high-level summary..."):
                                    chain = LLMChain(llm=llm, prompt=chat_prompt)
                                    high_level_summary = chain.run(user_prompt=user_prompt).strip()

                                # For cluster-specific summaries, use the same customized prompt
                                local_system_message = SystemMessagePromptTemplate.from_template(st.session_state['system_prompt'])
                                local_human_message = HumanMessagePromptTemplate.from_template("{user_prompt}")
                                local_chat_prompt = ChatPromptTemplate.from_messages([local_system_message, local_human_message])

                                # Possibly add references to high-level summary
                                if enable_references and reference_id_column:
                                    with st.spinner("Adding references to high-level summary..."):
                                        enhanced_summary = add_references_to_summary(
                                            high_level_summary,
                                            df_scope,
                                            reference_id_column,
                                            url_column if add_hyperlinks else None,
                                            llm
                                        )
                                    st.write("### High-Level Summary (with references):")
                                    st.markdown(enhanced_summary, unsafe_allow_html=True)
                                    with st.expander("View original summary (without references)"):
                                        st.write(high_level_summary)
                                else:
                                    st.write("### High-Level Summary:")
                                    st.write(high_level_summary)

                                # Summaries per cluster
                                # Only if multiple clusters are selected
                                unique_selected_topics = df_scope['Topic'].unique()
                                if len(unique_selected_topics) > 1:
                                    st.write("### Summaries per Selected Cluster")
                                    summaries = []
                                    for topic_val in unique_selected_topics:
                                        cluster_df = df_scope[df_scope['Topic'] == topic_val]
                                        cluster_text = " ".join(cluster_df['text'].tolist())
                                        if not cluster_text.strip():
                                            continue
                                        user_prompt_local = f"**Text to summarize**: {cluster_text}"
                                        with st.spinner(f"Summarizing cluster {topic_val}..."):
                                            local_chain = LLMChain(llm=llm, prompt=local_chat_prompt)
                                            summary_local = local_chain.run(user_prompt=user_prompt_local).strip()

                                        if enable_references and reference_id_column:
                                            with st.spinner(f"Adding references to cluster {topic_val} summary..."):
                                                summary_with_refs = add_references_to_summary(
                                                    summary_local,
                                                    cluster_df,
                                                    reference_id_column,
                                                    url_column if add_hyperlinks else None,
                                                    llm
                                                )
                                            summaries.append({
                                                'Topic': topic_val,
                                                'Summary': summary_local,
                                                'Enhanced_Summary': summary_with_refs
                                            })
                                        else:
                                            summaries.append({
                                                'Topic': topic_val,
                                                'Summary': summary_local
                                            })

                                    if summaries:
                                        summary_df = pd.DataFrame(summaries)
                                        # Display
                                        if enable_references and 'Enhanced_Summary' in summary_df.columns:
                                            st.write("### Summaries per Cluster (with references):")
                                            for idx, row in summary_df.iterrows():
                                                st.write(f"**Topic {row['Topic']}**")
                                                st.markdown(row['Enhanced_Summary'], unsafe_allow_html=True)
                                                st.write("---")
                                            with st.expander("View original summaries in table format"):
                                                st.dataframe(summary_df[['Topic', 'Summary']])
                                        else:
                                            st.write("### Summaries per Cluster:")
                                            st.dataframe(summary_df)

                                        # Download
                                        if 'Enhanced_Summary' in summary_df.columns:
                                            dl_df = summary_df[['Topic', 'Summary']]
                                        else:
                                            dl_df = summary_df
                                        csv_bytes = dl_df.to_csv(index=False).encode('utf-8')
                                        b64 = base64.b64encode(csv_bytes).decode()
                                        href = f'<a href="data:file/csv;base64,{b64}" download="summaries.csv">Download Summaries CSV</a>'
                                        st.markdown(href, unsafe_allow_html=True)
    else:
        st.warning("No data available for summarization.")


###############################################################################
# Tab: Internal Validation
###############################################################################
with tab_internal:
    st.header("Internal Validation & Debugging")

    # Show hierarchy
    hierarchy = st.session_state.get('hierarchy', pd.DataFrame())
    st.subheader("Hierarchical Topics DataFrame")
    if not hierarchy.empty:
        st.dataframe(hierarchy)
    else:
        st.write("No hierarchical topics data available.")

    # Show any clustering error
    error_msg = st.session_state.get('clustering_error', None)
    if error_msg:
        st.subheader("Clustering Error Details")
        st.write(error_msg)

    # Session State Info
    st.subheader("Internal Variables")
    st.write("Min Cluster Size:", st.session_state.get('min_cluster_size', 'Not set'))
    st.write("Text Columns:", st.session_state.get('text_columns', 'Not set'))
    st.write("Using Default Dataset:", st.session_state.get('using_default_dataset', 'Not set'))
    st.write("Last Text Columns Used for Embeddings:", st.session_state.get('last_text_columns', 'Not set'))

    st.subheader("Filtered DF")
    st.write(st.session_state['filtered_df'].head() if not st.session_state['filtered_df'].empty else "Empty")

    st.subheader("Clustered Data")
    if 'clustered_data' in st.session_state:
        st.write(st.session_state['clustered_data'].head())
    else:
        st.write("No clustered data.")

    # Embeddings info
    if 'embeddings' in st.session_state and st.session_state['embeddings'] is not None:
        st.subheader("Embeddings Information")
        st.write("Shape of Embeddings:", st.session_state['embeddings'].shape)
    else:
        st.write("No embeddings data available.")