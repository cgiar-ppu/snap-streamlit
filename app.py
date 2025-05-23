# app.py

import streamlit as st

# Set page config first, before any other st commands
st.set_page_config(page_title="SNAP", layout="wide")

# Add warning filters
import warnings
# More specific warning filters for torch.classes
warnings.filterwarnings('ignore', message='.*torch.classes.*__path__._path.*')
warnings.filterwarnings('ignore', message='.*torch.classes.*registered via torch::class_.*')

import pandas as pd
import numpy as np
import os
import io
import time
from datetime import datetime
import base64
import re
import pickle
from typing import List, Dict, Any, Tuple
import plotly.express as px
import torch

# For parallelism
from concurrent.futures import ThreadPoolExecutor
from functools import partial

# Import necessary libraries for embeddings, clustering, and summarization
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from bertopic import BERTopic
from hdbscan import HDBSCAN
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# For summarization and chat
from langchain.chains import LLMChain
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from openai import OpenAI
from transformers import GPT2TokenizerFast

# Initialize OpenAI client and tokenizer
client = OpenAI()

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

# Function to get or create model directory
def get_model_dir():
    base_dir = get_base_dir()
    model_dir = os.path.join(base_dir, 'models')
    os.makedirs(model_dir, exist_ok=True)
    return model_dir

# Function to load tokenizer from local storage or download
def load_tokenizer():
    model_dir = get_model_dir()
    tokenizer_dir = os.path.join(model_dir, 'tokenizer')
    os.makedirs(tokenizer_dir, exist_ok=True)
    
    try:
        # Try to load from local directory first
        tokenizer = GPT2TokenizerFast.from_pretrained(tokenizer_dir)
        #st.success("Loaded tokenizer from local storage")
    except Exception as e:
        #st.warning("Downloading tokenizer (one-time operation)...")
        try:
            # Download and save to local directory
            tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")  # Use standard GPT2 tokenizer
            tokenizer.save_pretrained(tokenizer_dir)
            #st.success("Downloaded and saved tokenizer")
        except Exception as download_e:
            #st.error(f"Error downloading tokenizer: {str(download_e)}")
            raise
    
    return tokenizer

# Load tokenizer
try:
    tokenizer = load_tokenizer()
except Exception as e:
    #st.error("Failed to load tokenizer. Some functionality may be limited.")
    tokenizer = None

MAX_CONTEXT_WINDOW = 128000  # GPT-4o context window size

# Initialize chat history in session state if not exists
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

###############################################################################
# Helper: Get chat response from OpenAI
###############################################################################
def get_chat_response(messages):
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Error querying OpenAI: {e}")
        return None

###############################################################################
# Helper: Generate raw summary for a cluster (without references)
###############################################################################
def generate_raw_cluster_summary(
    topic_val: int,
    cluster_df: pd.DataFrame,
    llm: Any,
    chat_prompt: Any
) -> Dict[str, Any]:
    """Generate a summary for a single cluster without reference enhancement,
       automatically trimming text if it exceeds a safe token limit."""
    cluster_text = " ".join(cluster_df['text'].tolist())
    if not cluster_text.strip():
        return None

    # Define a safe limit (95% of max context window to leave room for prompts)
    safe_limit = int(MAX_CONTEXT_WINDOW * 0.95)
    
    # Encode the text into tokens
    encoded_text = tokenizer.encode(cluster_text, add_special_tokens=False)
    
    # If the text is too large, slice it
    if len(encoded_text) > safe_limit:
        #st.warning(f"Cluster {topic_val} text is too large ({len(encoded_text)} tokens). Trimming to {safe_limit} tokens.")
        encoded_text = encoded_text[:safe_limit]
        cluster_text = tokenizer.decode(encoded_text)
    
    user_prompt_local = f"**Text to summarize**: {cluster_text}"
    try:
        local_chain = LLMChain(llm=llm, prompt=chat_prompt)
        summary_local = local_chain.run(user_prompt=user_prompt_local).strip()
        return {'Topic': topic_val, 'Summary': summary_local}
    except Exception as e:
        st.error(f"Error generating summary for cluster {topic_val}: {str(e)}")
        return None

###############################################################################
# Helper: Enhance a summary with references
###############################################################################
def enhance_summary_with_references(
    summary_dict: Dict[str, Any],
    df_scope: pd.DataFrame,
    reference_id_column: str,
    url_column: str = None,
    llm: Any = None
) -> Dict[str, Any]:
    """Add references to a summary."""
    if not summary_dict or 'Summary' not in summary_dict:
        return summary_dict
    
    try:
        cluster_df = df_scope[df_scope['Topic'] == summary_dict['Topic']]
        enhanced = add_references_to_summary(
            summary_dict['Summary'],
            cluster_df,
            reference_id_column,
            url_column,
            llm
        )
        summary_dict['Enhanced_Summary'] = enhanced
        return summary_dict
    except Exception as e:
        st.error(f"Error enhancing summary for cluster {summary_dict.get('Topic')}: {str(e)}")
        return summary_dict

###############################################################################
# Helper: Process summaries in parallel
###############################################################################
def process_summaries_in_parallel(
    df_scope: pd.DataFrame,
    unique_selected_topics: List[int],
    llm: Any,
    chat_prompt: Any,
    enable_references: bool = False,
    reference_id_column: str = None,
    url_column: str = None,
    max_workers: int = 16
) -> List[Dict[str, Any]]:
    """Process multiple cluster summaries in parallel using ThreadPoolExecutor."""
    summaries = []
    total_topics = len(unique_selected_topics)
    
    # Create progress placeholders
    progress_text = st.empty()
    progress_bar = st.progress(0)
    
    try:
        # Phase 1: Generate raw summaries in parallel
        progress_text.text(f"Phase 1/3: Generating cluster summaries in parallel (0/{total_topics} completed)")
        completed_summaries = 0
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit summary generation tasks
            future_to_topic = {
                executor.submit(
                    generate_raw_cluster_summary,
                    topic_val,
                    df_scope[df_scope['Topic'] == topic_val],
                    llm,
                    chat_prompt
                ): topic_val
                for topic_val in unique_selected_topics
            }
            
            # Process completed summary tasks
            for future in future_to_topic:
                try:
                    result = future.result()
                    if result:
                        summaries.append(result)
                    completed_summaries += 1
                    # Update progress
                    progress = completed_summaries / total_topics
                    progress_bar.progress(progress)
                    progress_text.text(
                        f"Phase 1/3: Generating cluster summaries in parallel ({completed_summaries}/{total_topics} completed)"
                    )
                except Exception as e:
                    topic_val = future_to_topic[future]
                    st.error(f"Error in summary generation for cluster {topic_val}: {str(e)}")
                    completed_summaries += 1
                    continue
        
        # Phase 2: Enhance summaries with references in parallel (if enabled)
        if enable_references and reference_id_column and summaries:
            total_to_enhance = len(summaries)
            completed_enhancements = 0
            progress_text.text(f"Phase 2/3: Adding references to summaries (0/{total_to_enhance} completed)")
            progress_bar.progress(0)
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit reference enhancement tasks
                future_to_summary = {
                    executor.submit(
                        enhance_summary_with_references,
                        summary_dict,
                        df_scope,
                        reference_id_column,
                        url_column,
                        llm
                    ): summary_dict.get('Topic')
                    for summary_dict in summaries
                }
                
                # Process completed enhancement tasks
                enhanced_summaries = []
                for future in future_to_summary:
                    try:
                        result = future.result()
                        if result:
                            enhanced_summaries.append(result)
                        completed_enhancements += 1
                        # Update progress
                        progress = completed_enhancements / total_to_enhance
                        progress_bar.progress(progress)
                        progress_text.text(
                            f"Phase 2/3: Adding references to summaries ({completed_enhancements}/{total_to_enhance} completed)"
                        )
                    except Exception as e:
                        topic_val = future_to_summary[future]
                        st.error(f"Error in reference enhancement for cluster {topic_val}: {str(e)}")
                        completed_enhancements += 1
                        continue
                
                summaries = enhanced_summaries

        # Phase 3: Generate cluster names in parallel
        if summaries:
            total_to_name = len(summaries)
            completed_names = 0
            progress_text.text(f"Phase 3/3: Generating cluster names (0/{total_to_name} completed)")
            progress_bar.progress(0)

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit cluster naming tasks
                future_to_summary = {
                    executor.submit(
                        generate_cluster_name,
                        summary_dict.get('Enhanced_Summary', summary_dict['Summary']),
                        llm
                    ): summary_dict.get('Topic')
                    for summary_dict in summaries
                }

                # Process completed naming tasks
                named_summaries = []
                for future in future_to_summary:
                    try:
                        cluster_name = future.result()
                        topic_val = future_to_summary[future]
                        # Find the corresponding summary dict
                        summary_dict = next(s for s in summaries if s['Topic'] == topic_val)
                        summary_dict['Cluster_Name'] = cluster_name
                        named_summaries.append(summary_dict)
                        completed_names += 1
                        # Update progress
                        progress = completed_names / total_to_name
                        progress_bar.progress(progress)
                        progress_text.text(
                            f"Phase 3/3: Generating cluster names ({completed_names}/{total_to_name} completed)"
                        )
                    except Exception as e:
                        topic_val = future_to_summary[future]
                        st.error(f"Error in cluster naming for cluster {topic_val}: {str(e)}")
                        completed_names += 1
                        continue

                summaries = named_summaries
    finally:
        # Clean up progress indicators
        progress_text.empty()
        progress_bar.empty()
    
    return summaries

###############################################################################
# Helper: Generate cluster name
###############################################################################
def generate_cluster_name(summary_text: str, llm: Any) -> str:
    """Generate a concise, descriptive name for a cluster based on its summary."""
    system_prompt = """You are a cluster naming expert. Your task is to generate a very concise (3-6 words) but descriptive name for a cluster based on its summary. The name should capture the main theme or focus of the cluster.

Rules:
1. Keep it between 3-6 words
2. Be specific but concise
3. Capture the main theme/focus
4. Use title case
4. Do not include words like "Cluster", "Topic", or "Theme"
5. Focus on the content, not metadata

Example good names:
- Agricultural Water Management Innovation
- Gender Equality in Farming
- Climate-Smart Village Implementation
- Sustainable Livestock Practices"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Generate a concise cluster name based on this summary:\n\n{summary_text}"}
    ]

    try:
        response = get_chat_response(messages)
        # Clean up response (remove quotes, newlines, etc.)
        cluster_name = response.strip().strip('"').strip("'").strip()
        return cluster_name
    except Exception as e:
        st.error(f"Error generating cluster name: {str(e)}")
        return "Unnamed Cluster"

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
    """Initialize NLTK resources with better error handling and less verbose output"""
    nltk.data.path.append('/home/appuser/nltk_data')  # Ensure consistent data path
    
    resources = {
        'tokenizers/punkt': 'punkt_tab',  # Updated to use punkt_tab
        'corpora/stopwords': 'stopwords'
    }
    
    for resource_path, resource_name in resources.items():
        try:
            nltk.data.find(resource_path)
        except LookupError:
            try:
                nltk.download(resource_name, quiet=True)
            except Exception as e:
                st.warning(f"Error downloading NLTK resource {resource_name}: {e}")
    
    # Test tokenizer silently
    try:
        from nltk.tokenize import PunktSentenceTokenizer
        tokenizer = PunktSentenceTokenizer()
        tokenizer.tokenize("Test sentence.")
    except Exception as e:
        st.error(f"Error initializing NLTK tokenizer: {e}")
        try:
            nltk.download('punkt_tab', quiet=True)  # Updated to use punkt_tab
        except Exception as e:
            st.error(f"Failed to download punkt_tab tokenizer: {e}")

# Initialize NLTK resources
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

    # Split the summary into paragraphs first
    paragraphs = summary.split('\n\n')
    enhanced_paragraphs = []

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
    1. A sentence or bullet point from a summary
    2. A list of source texts with their IDs
    
    Your task is to identify which source text(s) the text most likely came from.
    Return ONLY the IDs of the source texts that contributed to the text, separated by commas.
    If you cannot confidently attribute the text to any source, return "unknown".
    """

    for paragraph in paragraphs:
        if not paragraph.strip():
            enhanced_paragraphs.append('')
            continue

        # Check if it's a bullet point list
        if any(line.strip().startswith('- ') or line.strip().startswith('* ') for line in paragraph.split('\n')):
            # Handle bullet points
            bullet_lines = paragraph.split('\n')
            enhanced_bullets = []
            for line in bullet_lines:
                if not line.strip():
                    enhanced_bullets.append(line)
                    continue
                
                if line.strip().startswith('- ') or line.strip().startswith('* '):
                    # Process each bullet point
                    user_prompt = f"""
                    Text: {line.strip()}

                    Source texts:
                    {'\n'.join([f"ID: {ref_id}, Text: {text[:500]}..." for ref_id, text in zip(reference_ids, source_texts)])}

                    Which source ID(s) did this text most likely come from? Return only the ID(s) separated by commas, or "unknown".
                    """

                    try:
                        system_message = SystemMessagePromptTemplate.from_template(system_prompt)
                        human_message = HumanMessagePromptTemplate.from_template("{user_prompt}")
                        chat_prompt = ChatPromptTemplate.from_messages([system_message, human_message])
                        chain = LLMChain(llm=llm, prompt=chat_prompt)
                        response = chain.run(user_prompt=user_prompt)
                        source_ids = response.strip()

                        if source_ids.lower() == "unknown":
                            enhanced_bullets.append(line)
                        else:
                            # Extract just the IDs
                            source_ids = re.sub(r'[^0-9,\s]', '', source_ids)
                            source_ids = re.sub(r'\s+', '', source_ids)
                            ids = [id_.strip() for id_ in source_ids.split(',') if id_.strip()]
                            
                            if ids:
                                ref_parts = []
                                for id_ in ids:
                                    if id_ in url_map:
                                        ref_parts.append(f'<a href="{url_map[id_]}" target="_blank">{id_}</a>')
                                    else:
                                        ref_parts.append(id_)
                                ref_string = ", ".join(ref_parts)
                                enhanced_bullets.append(f"{line} [{ref_string}]")
                            else:
                                enhanced_bullets.append(line)
                    except Exception:
                        enhanced_bullets.append(line)
                else:
                    enhanced_bullets.append(line)
            
            enhanced_paragraphs.append('\n'.join(enhanced_bullets))
        else:
            # Handle regular paragraphs
            sentences = re.split(r'(?<=[.!?])\s+', paragraph)
            enhanced_sentences = []

            for sentence in sentences:
                if not sentence.strip():
                    continue

                user_prompt = f"""
                Sentence: {sentence.strip()}

                Source texts:
                {'\n'.join([f"ID: {ref_id}, Text: {text[:500]}..." for ref_id, text in zip(reference_ids, source_texts)])}

                Which source ID(s) did this sentence most likely come from? Return only the ID(s) separated by commas, or "unknown".
                """

                try:
                    system_message = SystemMessagePromptTemplate.from_template(system_prompt)
                    human_message = HumanMessagePromptTemplate.from_template("{user_prompt}")
                    chat_prompt = ChatPromptTemplate.from_messages([system_message, human_message])
                    chain = LLMChain(llm=llm, prompt=chat_prompt)
                    response = chain.run(user_prompt=user_prompt)
                    source_ids = response.strip()

                    if source_ids.lower() == "unknown":
                        enhanced_sentences.append(sentence)
                    else:
                        # Extract just the IDs
                        source_ids = re.sub(r'[^0-9,\s]', '', source_ids)
                        source_ids = re.sub(r'\s+', '', source_ids)
                        ids = [id_.strip() for id_ in source_ids.split(',') if id_.strip()]
                        
                        if ids:
                            ref_parts = []
                            for id_ in ids:
                                if id_ in url_map:
                                    ref_parts.append(f'<a href="{url_map[id_]}" target="_blank">{id_}</a>')
                                else:
                                    ref_parts.append(id_)
                            ref_string = ", ".join(ref_parts)
                            enhanced_sentences.append(f"{sentence} [{ref_string}]")
                        else:
                            enhanced_sentences.append(sentence)
                except Exception:
                    enhanced_sentences.append(sentence)

            enhanced_paragraphs.append(' '.join(enhanced_sentences))

    # Join paragraphs back together with double newlines to preserve formatting
    return '\n\n'.join(enhanced_paragraphs)


st.sidebar.image("static/SNAP_logo.png", width=350)

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
    model_dir = get_model_dir()
    st_model_dir = os.path.join(model_dir, 'sentence_transformer')
    os.makedirs(st_model_dir, exist_ok=True)
    
    model_name = 'all-MiniLM-L6-v2'
    try:
        # Try to load from local directory first
        model = SentenceTransformer(st_model_dir)
        #st.success("Loaded sentence transformer from local storage")
    except Exception as e:
        #st.warning("Downloading sentence transformer model (one-time operation)...")
        try:
            # Download and save to local directory
            model = SentenceTransformer(model_name)
            model.save(st_model_dir)
            #st.success("Downloaded and saved sentence transformer model")
        except Exception as download_e:
            st.error(f"Error downloading sentence transformer model: {str(download_e)}")
            raise
    
    return model.to(device)

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

    base_name = "PRMS_2022_2023_2024_QAed" if using_default_dataset else "custom_dataset"
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

# Selector de vista
st.sidebar.radio("Select view", ["Automatic Mode", "Power User Mode"], key="view")

if st.session_state.view == "Power User Mode":      
    st.header("Power User Mode")     
    ###############################################################################
    # Sidebar: Dataset Selection
    ###############################################################################
    st.sidebar.title("Data Selection")
    dataset_option = st.sidebar.selectbox('Select Dataset', ('PRMS 2022+2023+2024 QAed', 'Upload my dataset'))

    if 'df' not in st.session_state:
        st.session_state['df'] = pd.DataFrame()
    if 'filtered_df' not in st.session_state:
        st.session_state['filtered_df'] = pd.DataFrame()

    if dataset_option == 'PRMS 2022+2023+2024 QAed':
        default_dataset_path = os.path.join(BASE_DIR, 'input', 'export_data_table_results_20251203_101413CET.xlsx')
        df = load_default_dataset(default_dataset_path)
        if df is not None:
            st.session_state['df'] = df.copy()
            st.session_state['using_default_dataset'] = True
            
            # Initialize filtered_df with full dataset by default
            if 'filtered_df' not in st.session_state or st.session_state['filtered_df'].empty:
                st.session_state['filtered_df'] = df.copy()
            
            # Initialize filter_state if not exists
            if 'filter_state' not in st.session_state:
                st.session_state['filter_state'] = {
                    'applied': False,
                    'filters': {}
                }
            
            # Set default text columns if not already set
            if 'text_columns' not in st.session_state or not st.session_state['text_columns']:
                default_text_cols = []
                if 'Title' in df.columns and 'Description' in df.columns:
                    default_text_cols = ['Title', 'Description']
                st.session_state['text_columns'] = default_text_cols
            
            #st.write("Using default dataset:")
            #st.write("Data Preview:")
            #st.dataframe(st.session_state['filtered_df'].head(), hide_index=True)
            #st.write(f"Total number of results: {len(st.session_state['filtered_df'])}")

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
                
                # Quick search section (outside form)
                for col_name in st.session_state['additional_filters_selected']:
                    unique_vals = sorted(df[col_name].dropna().unique().tolist())
                    
                    # Add a search box for quick selection
                    search_key = f"search_{col_name}"
                    if search_key not in st.session_state:
                        st.session_state[search_key] = ""
                        
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        search_term = st.text_input(
                            f"Search in {col_name}",
                            key=search_key,
                            help="Enter text to find and select all matching values"
                        )
                    with col2:
                        if st.button(f"Select Matching", key=f"select_{col_name}"):
                            # Handle comma-separated values
                            if search_term:
                                matching_vals = [
                                    val for val in unique_vals
                                    if any(search_term.lower() in str(part).lower() 
                                        for part in (val.split(',') if isinstance(val, str) else [val]))
                                ]
                                # Update the multiselect default value
                                current_selected = st.session_state['filter_values'].get(col_name, [])
                                st.session_state['filter_values'][col_name] = list(set(current_selected + matching_vals))
                                
                                # Show feedback about matches
                                if matching_vals:
                                    st.success(f"Found and selected {len(matching_vals)} matching values")
                                else:
                                    st.warning("No matching values found")

                # Filter application form
                with st.form("apply_filters_form"):
                    for col_name in st.session_state['additional_filters_selected']:
                        unique_vals = sorted(df[col_name].dropna().unique().tolist())
                        selected_vals = st.multiselect(
                            f"Filter by {col_name}",
                            options=unique_vals,
                            default=st.session_state['filter_values'].get(col_name, [])
                        )
                        st.session_state['filter_values'][col_name] = selected_vals

                    # Add clear filters button and apply filters button
                    col1, col2 = st.columns([1, 4])
                    with col1:
                        clear_filters = st.form_submit_button("Clear All")
                    with col2:
                        apply_filters_submitted = st.form_submit_button("Apply Filters to Dataset")

                    if clear_filters:
                        st.session_state['filter_values'] = {}
                        # Clear any existing summary data when filters are cleared
                        if 'summary_df' in st.session_state:
                            del st.session_state['summary_df']
                        if 'high_level_summary' in st.session_state:
                            del st.session_state['high_level_summary']
                        if 'enhanced_summary' in st.session_state:
                            del st.session_state['enhanced_summary']
                        st.rerun()

            # Text columns selection moved to Advanced Settings
            with st.expander("⚙️ Advanced Settings", expanded=False):
                st.subheader("**Select Text Columns for Embedding**")
                text_columns_selected = st.multiselect(
                    "Text Columns:",
                    df_cols,
                    default=st.session_state['text_columns'],
                    help="Choose columns containing text for semantic search and clustering. "
                        "If multiple are selected, their text will be concatenated."
                )
                st.session_state['text_columns'] = text_columns_selected

            # Apply filters to the dataset
            filtered_df = df.copy()
            if 'apply_filters_submitted' in locals() and apply_filters_submitted:
                # Clear any existing summary data when new filters are applied
                if 'summary_df' in st.session_state:
                    del st.session_state['summary_df']
                if 'high_level_summary' in st.session_state:
                    del st.session_state['high_level_summary']
                if 'enhanced_summary' in st.session_state:
                    del st.session_state['enhanced_summary']
                
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

            # Show current data preview and download button
            if st.session_state['filtered_df'] is not None:
                if st.session_state['filter_state']['applied']:
                    st.write("Filtered Data Preview:")
                else:
                    st.write("Current Data Preview:")
                st.dataframe(st.session_state['filtered_df'].head(), hide_index=True)
                st.write(f"Total number of results: {len(st.session_state['filtered_df'])}")

                output = io.BytesIO()
                writer = pd.ExcelWriter(output, engine='openpyxl')
                st.session_state['filtered_df'].to_excel(writer, index=False)
                writer.close()
                processed_data = output.getvalue()

                st.download_button(
                    label="Download Current Data",
                    data=processed_data,
                    file_name='data.xlsx',
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
                st.dataframe(filtered_df.head(), hide_index=True)
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

    tabs_titles = ["Semantic Search", "Clustering", "Summarization", "Chat", "Help"]
    tabs = st.tabs(tabs_titles)
    # We just create these references so we can navigate more easily
    tab_semantic, tab_clustering, tab_summarization, tab_chat, tab_help = tabs

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
                                # Clear any existing summary data when new search is run
                                if 'summary_df' in st.session_state:
                                    del st.session_state['summary_df']
                                if 'high_level_summary' in st.session_state:
                                    del st.session_state['high_level_summary']
                                if 'enhanced_summary' in st.session_state:
                                    del st.session_state['enhanced_summary']

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
                        st.dataframe(results[cols_to_display], hide_index=True)
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
            
            # Create a single form for clustering settings
            with st.form("clustering_form"):
                st.subheader("Clustering Settings")
                
                # Data source selection
                clustering_option = st.radio(
                    "Select data for clustering:",
                    ('Full Dataset', 'Filtered Dataset', 'Semantic Search Results')
                )
                
                # Clustering parameters
                min_cluster_size_val = st.slider(
                    "Min Cluster Size",
                    min_value=2,
                    max_value=50,
                    value=st.session_state.get('min_cluster_size', 5),
                    help="Minimum size of each cluster in HDBSCAN; In other words, it's the minimum number of documents/texts that must be grouped together to form a valid cluster.\n\n- A larger value (e.g., 20) will result in fewer, larger clusters\n- A smaller value (e.g., 2-5) will allow for more clusters, including smaller ones\n- Documents that don't fit into any cluster meeting this minimum size requirement are labeled as noise (typically assigned to cluster -1)"
                )
                
                run_clustering = st.form_submit_button("Run Clustering")
                
                if run_clustering:
                    st.session_state.active_tab_index = tabs_titles.index("Clustering")
                    st.session_state['min_cluster_size'] = min_cluster_size_val

            # Decide which DataFrame is used based on the selection
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
            else:  # Full Dataset
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

                if df_to_cluster is not None and embeddings is not None and not df_to_cluster.empty and run_clustering:
                    with st.spinner("Performing clustering..."):
                        # Clear any existing summary data when clustering is run
                        if 'summary_df' in st.session_state:
                            del st.session_state['summary_df']
                        if 'high_level_summary' in st.session_state:
                            del st.session_state['high_level_summary']
                        if 'enhanced_summary' in st.session_state:
                            del st.session_state['enhanced_summary']

                        dfc = df_to_cluster.copy().fillna("")
                        dfc['text'] = dfc[text_columns].astype(str).agg(' '.join, axis=1)

                        # Filter embeddings to those rows
                        selected_indices = dfc.index
                        embeddings_clustering = embeddings[selected_indices]

                        # Basic cleaning
                        stop_words = set(stopwords.words('english'))
                        texts_cleaned = []
                        for text in dfc['text'].tolist():
                            try:
                                # First try with word_tokenize
                                try:
                                    word_tokens = word_tokenize(text)
                                except LookupError:
                                    # If punkt is missing, try downloading it again
                                    nltk.download('punkt_tab', quiet=False)
                                    word_tokens = word_tokenize(text)
                                except Exception as e:
                                    # If word_tokenize fails, fall back to simple splitting
                                    st.warning(f"Using fallback tokenization due to error: {e}")
                                    word_tokens = text.split()
                                
                                filtered_text = ' '.join([w for w in word_tokens if w.lower() not in stop_words])
                                texts_cleaned.append(filtered_text)
                            except Exception as e:
                                st.error(f"Error processing text: {e}")
                                # Add the original text if processing fails
                                texts_cleaned.append(text)

                        try:
                            # Validation checks before clustering
                            if len(texts_cleaned) < min_cluster_size_val:
                                st.error(f"Not enough documents to form clusters. You have {len(texts_cleaned)} documents but minimum cluster size is set to {min_cluster_size_val}.")
                                st.session_state['clustering_error'] = "Insufficient documents for clustering"
                                st.session_state.active_tab_index = tabs_titles.index("Clustering")
                                st.stop()

                            # Convert embeddings to CPU numpy if needed
                            if torch.is_tensor(embeddings_clustering):
                                embeddings_for_clustering = embeddings_clustering.cpu().numpy()
                            else:
                                embeddings_for_clustering = embeddings_clustering

                            # Additional validation
                            if embeddings_for_clustering.shape[0] != len(texts_cleaned):
                                st.error("Mismatch between number of embeddings and texts.")
                                st.session_state['clustering_error'] = "Embedding and text count mismatch"
                                st.session_state.active_tab_index = tabs_titles.index("Clustering")
                                st.stop()

                            # Build the HDBSCAN model with error handling
                            try:
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

                                # Fit the model and get topics
                                topics, probs = topic_model.fit_transform(
                                    texts_cleaned, 
                                    embeddings=embeddings_for_clustering
                                )

                                # Validate clustering results
                                unique_topics = set(topics)
                                if len(unique_topics) < 2:
                                    st.warning("Clustering resulted in too few clusters. Retry or try reducing the minimum cluster size.")
                                    if -1 in unique_topics:
                                        non_noise_docs = sum(1 for t in topics if t != -1)
                                        st.info(f"Only {non_noise_docs} documents were assigned to clusters. The rest were marked as noise (-1).")
                                        if non_noise_docs < min_cluster_size_val:
                                            st.error("Not enough documents were successfully clustered. Try reducing the minimum cluster size.")
                                            st.session_state['clustering_error'] = "Insufficient clustered documents"
                                            st.session_state.active_tab_index = tabs_titles.index("Clustering")
                                            st.stop()

                                # Store results if validation passes
                                dfc['Topic'] = topics
                                st.session_state['topic_model'] = topic_model
                                st.session_state['clustered_data'] = dfc.copy()
                                st.session_state['clustering_texts_cleaned'] = texts_cleaned
                                st.session_state['clustering_embeddings'] = embeddings_for_clustering
                                st.session_state['clustering_completed'] = True

                                # Try to generate visualizations with error handling
                                try:
                                    st.session_state['intertopic_distance_fig'] = topic_model.visualize_topics()
                                except Exception as viz_error:
                                    st.warning("Could not generate topic visualization. This usually happens when there are too few total clusters. Try adjusting the minimum cluster size or adding more documents.")
                                    st.session_state['intertopic_distance_fig'] = None

                                try:
                                    st.session_state['topic_document_fig'] = topic_model.visualize_documents(
                                        texts_cleaned, 
                                        embeddings=embeddings_for_clustering
                                    )
                                except Exception as viz_error:
                                    st.warning("Could not generate document visualization. This might happen when the clustering results are not optimal. Try adjusting the clustering parameters.")
                                    st.session_state['topic_document_fig'] = None

                                try:
                                    hierarchy = topic_model.hierarchical_topics(texts_cleaned)
                                    st.session_state['hierarchy'] = hierarchy if hierarchy is not None else pd.DataFrame()
                                    st.session_state['hierarchy_fig'] = topic_model.visualize_hierarchy()
                                except Exception as viz_error:
                                    st.warning("Could not generate topic hierarchy visualization. This usually happens when there aren't enough distinct topics to form a hierarchy.")
                                    st.session_state['hierarchy'] = pd.DataFrame()
                                    st.session_state['hierarchy_fig'] = None

                            except ValueError as ve:
                                if "zero-size array to reduction operation maximum which has no identity" in str(ve):
                                    st.error("Clustering failed: No valid clusters could be formed. Try reducing the minimum cluster size.")
                                elif "Cannot use scipy.linalg.eigh for sparse A with k > N" in str(ve):
                                    st.error("Clustering failed: Too many components requested for the number of documents. Try with more documents or adjust clustering parameters.")
                                else:
                                    st.error(f"Clustering error: {str(ve)}")
                                st.session_state['clustering_error'] = str(ve)
                                st.session_state.active_tab_index = tabs_titles.index("Clustering")
                                st.stop()

                        except Exception as e:
                            st.error(f"An error occurred during clustering: {str(e)}")
                            st.session_state['clustering_error'] = str(e)
                            st.session_state['clustering_completed'] = False
                            st.session_state.active_tab_index = tabs_titles.index("Clustering")
                            st.stop()

                    # Display clustering results if they exist
                    if st.session_state.get('clustering_completed', False):
                        st.subheader("Topic Overview")
                        dfc = st.session_state['clustered_data']
                        topic_model = st.session_state['topic_model']
                        topics = dfc['Topic'].tolist()
                        
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

                        st.write("### Topic Overview")
                        st.dataframe(
                            cluster_df,
                            column_config={
                                "Topic": st.column_config.NumberColumn("Topic", help="Topic ID (-1 represents outliers)"),
                                "Count": st.column_config.NumberColumn("Count", help="Number of documents in this topic"),
                                "Top Keywords": st.column_config.TextColumn(
                                    "Top Keywords",
                                    help="Top 5 keywords that characterize this topic"
                                )
                            },
                            hide_index=True
                        )

                        st.subheader("Clustering Results")
                        columns_to_display = [c for c in dfc.columns if c != 'text']
                        st.dataframe(dfc[columns_to_display], hide_index=True)

                        # Display stored visualizations with error handling
                        st.write("### Intertopic Distance Map")
                        if st.session_state.get('intertopic_distance_fig') is not None:
                            try:
                                st.plotly_chart(st.session_state['intertopic_distance_fig'])
                            except Exception:
                                st.info("Topic visualization is not available for the current clustering results.")

                        st.write("### Topic Document Visualization")
                        if st.session_state.get('topic_document_fig') is not None:
                            try:
                                st.plotly_chart(st.session_state['topic_document_fig'])
                            except Exception:
                                st.info("Document visualization is not available for the current clustering results.")

                        st.write("### Topic Hierarchy")
                        if st.session_state.get('hierarchy_fig') is not None:
                            try:
                                st.plotly_chart(st.session_state['hierarchy_fig'])
                            except Exception:
                                st.info("Topic hierarchy visualization is not available for the current clustering results.")
                if not (df_to_cluster is not None and embeddings is not None and not df_to_cluster.empty and run_clustering):
                    pass
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

                    # If we have cluster names from previous summarization, add them
                    if 'summary_df' in st.session_state and 'Cluster_Name' in st.session_state['summary_df'].columns:
                        summary_df = st.session_state['summary_df']
                        # Create a mapping of topic to name for merging
                        topic_names = {t: name for t, name in zip(summary_df['Topic'], summary_df['Cluster_Name'])}
                        # Add cluster names to cluster_df
                        cluster_df['Cluster_Name'] = cluster_df['Topic'].map(lambda x: topic_names.get(x, 'Unnamed Cluster'))
                        # Reorder columns to show name after topic
                        cluster_df = cluster_df[['Topic', 'Cluster_Name', 'Count', 'Top Keywords']]
                        
                    st.write("### Available Clusters:")
                    st.dataframe(
                        cluster_df,
                        column_config={
                            "Topic": st.column_config.NumberColumn("Topic", help="Topic ID (-1 represents outliers)"),
                            "Cluster_Name": st.column_config.TextColumn("Cluster Name", help="AI-generated name describing the cluster theme"),
                            "Count": st.column_config.NumberColumn("Count", help="Number of documents in this topic"),
                            "Top Keywords": st.column_config.TextColumn(
                                "Top Keywords",
                                help="Top 5 keywords that characterize this topic"
                            )
                        },
                        hide_index=True
                    )

                    # Summarization settings
                    st.subheader("Summarization Settings")
                    # Summaries scope
                    summary_scope = st.radio(
                        "Generate summaries for:",
                        ["All clusters", "Specific clusters"]
                    )
                    if summary_scope == "Specific clusters":
                        # Format options to include cluster names if available
                        if 'Cluster_Name' in cluster_df.columns:
                            topic_options = [f"Cluster {t} - {name}" for t, name in zip(cluster_df['Topic'], cluster_df['Cluster_Name'])]
                            topic_to_id = {opt: t for opt, t in zip(topic_options, cluster_df['Topic'])}
                            selected_topic_options = st.multiselect("Select clusters to summarize", topic_options)
                            selected_topics = [topic_to_id[opt] for opt in selected_topic_options]
                        else:
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
                            # Set flag to indicate summarization button was clicked
                            st.session_state['_summarization_button_clicked'] = True
                            
                            llm = ChatOpenAI(
                                api_key=openai_api_key, 
                                model_name='gpt-4o-mini',  # or 'gpt-4o'
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
                                    # For cluster-specific summaries, use the customized prompt
                                    local_system_message = SystemMessagePromptTemplate.from_template(st.session_state['system_prompt'])
                                    local_human_message = HumanMessagePromptTemplate.from_template("{user_prompt}")
                                    local_chat_prompt = ChatPromptTemplate.from_messages([local_system_message, local_human_message])

                                    # Summaries per cluster
                                    # Only if multiple clusters are selected
                                    unique_selected_topics = df_scope['Topic'].unique()
                                    if len(unique_selected_topics) > 1:
                                        st.write("### Summaries per Selected Cluster")
                                        
                                        # Process summaries in parallel
                                        with st.spinner("Generating cluster summaries in parallel..."):
                                            summaries = process_summaries_in_parallel(
                                                df_scope=df_scope,
                                                unique_selected_topics=unique_selected_topics,
                                                llm=llm,
                                                chat_prompt=local_chat_prompt,
                                                enable_references=enable_references,
                                                reference_id_column=reference_id_column,
                                                url_column=url_column if add_hyperlinks else None,
                                                max_workers=min(16, len(unique_selected_topics))  # Limit workers based on clusters
                                            )

                                    if summaries:
                                        summary_df = pd.DataFrame(summaries)
                                        # Store the summaries DataFrame in session state
                                        st.session_state['summary_df'] = summary_df
                                        # Store additional summary info in session state
                                        st.session_state['has_references'] = enable_references
                                        st.session_state['reference_id_column'] = reference_id_column
                                        st.session_state['url_column'] = url_column if add_hyperlinks else None
                                        
                                        # Update cluster_df with new names
                                        if 'Cluster_Name' in summary_df.columns:
                                            topic_names = {t: name for t, name in zip(summary_df['Topic'], summary_df['Cluster_Name'])}
                                            cluster_df['Cluster_Name'] = cluster_df['Topic'].map(lambda x: topic_names.get(x, 'Unnamed Cluster'))
                                            cluster_df = cluster_df[['Topic', 'Cluster_Name', 'Count', 'Top Keywords']]
                                            
                                            # Immediately display updated cluster overview
                                            st.write("### Updated Topic Overview:")
                                            st.dataframe(
                                                cluster_df,
                                                column_config={
                                                    "Topic": st.column_config.NumberColumn("Topic", help="Topic ID (-1 represents outliers)"),
                                                    "Cluster_Name": st.column_config.TextColumn("Cluster Name", help="AI-generated name describing the cluster theme"),
                                                    "Count": st.column_config.NumberColumn("Count", help="Number of documents in this topic"),
                                                    "Top Keywords": st.column_config.TextColumn(
                                                        "Top Keywords",
                                                        help="Top 5 keywords that characterize this topic"
                                                    )
                                                },
                                                hide_index=True
                                            )
                                            
                                        # Now generate high-level summary from the cluster summaries
                                        with st.spinner("Generating high-level summary from cluster summaries..."):
                                            # Format cluster summaries with proper markdown and HTML
                                            formatted_summaries = []
                                            total_tokens = 0
                                            MAX_SAFE_TOKENS = int(MAX_CONTEXT_WINDOW * 0.75)  # Leave room for system prompt and completion
                                            summary_batches = []
                                            current_batch = []
                                            current_batch_tokens = 0

                                            for _, row in summary_df.iterrows():
                                                summary_text = row.get('Enhanced_Summary', row['Summary'])
                                                formatted_summary = f"### Cluster {row['Topic']} Summary:\n\n{summary_text}"
                                                summary_tokens = len(tokenizer(formatted_summary)["input_ids"])
                                                
                                                # If adding this summary would exceed the safe token limit, start a new batch
                                                if current_batch_tokens + summary_tokens > MAX_SAFE_TOKENS:
                                                    if current_batch:  # Only append if we have summaries in the current batch
                                                        summary_batches.append(current_batch)
                                                        current_batch = []
                                                        current_batch_tokens = 0
                                                
                                                current_batch.append(formatted_summary)
                                                current_batch_tokens += summary_tokens

                                            # Add the last batch if it has any summaries
                                            if current_batch:
                                                summary_batches.append(current_batch)

                                            # Generate overview for each batch
                                            batch_overviews = []
                                            with st.spinner("Generating batch summaries..."):
                                                for i, batch in enumerate(summary_batches, 1):
                                                    st.write(f"Processing batch {i} of {len(summary_batches)}...")
                                                    
                                                    batch_text = "\n\n".join(batch)
                                                    batch_prompt = f"""Below are summaries from a subset of clusters from results made using Transformers NLP on a set of results from the CGIAR reporting system. Each summary contains references to source documents in the form of hyperlinked IDs like [ID] or <a href="...">ID</a>.

Please create a comprehensive overview that synthesizes these clusters so that both the main themes and findings are covered in an organized way. IMPORTANT: 
1. Preserve all hyperlinked references exactly as they appear in the input summaries
2. Maintain the HTML anchor tags (<a href="...">) intact when using information from the summaries
3. Keep the markdown formatting for better readability
4. Note that this is part {i} of {len(summary_batches)} parts, so focus on the themes present in these specific clusters

Here are the cluster summaries to synthesize:

{batch_text}"""

                                                    # Generate overview for this batch
                                                    high_level_system_message = SystemMessagePromptTemplate.from_template(st.session_state['system_prompt'])
                                                    high_level_human_message = HumanMessagePromptTemplate.from_template("{user_prompt}")
                                                    high_level_chat_prompt = ChatPromptTemplate.from_messages([high_level_system_message, high_level_human_message])
                                                    high_level_chain = LLMChain(llm=llm, prompt=high_level_chat_prompt)
                                                    batch_overview = high_level_chain.run(user_prompt=batch_prompt).strip()
                                                    batch_overviews.append(batch_overview)

                                            # Now combine the batch overviews
                                            with st.spinner("Generating final combined summary..."):
                                                combined_overviews = "\n\n### Part ".join([f"{i+1}:\n\n{overview}" for i, overview in enumerate(batch_overviews)])
                                                final_prompt = f"""Below are {len(batch_overviews)} overview summaries, each covering different clusters of research results. Each part maintains its original references to source documents.

Please create a final comprehensive synthesis that:
1. Integrates the key themes and findings from all parts
2. Preserves all hyperlinked references exactly as they appear
3. Maintains the HTML anchor tags (<a href="...">) intact
4. Keeps the markdown formatting for better readability
5. Creates a coherent narrative across all parts
6. Highlights any themes that span multiple parts

Here are the overviews to synthesize:

### Part 1:

{combined_overviews}"""

                                                # Verify the final prompt's token count
                                                final_prompt_tokens = len(tokenizer(final_prompt)["input_ids"])
                                                if final_prompt_tokens > MAX_SAFE_TOKENS:
                                                    st.error(f"❌ Final synthesis prompt ({final_prompt_tokens:,} tokens) exceeds safe limit ({MAX_SAFE_TOKENS:,}). Using batch summaries separately.")
                                                    high_level_summary = "# Overall Summary\n\n" + "\n\n".join([f"## Batch {i+1}\n\n{overview}" for i, overview in enumerate(batch_overviews)])
                                                else:
                                                    # Generate final synthesis
                                                    high_level_chain = LLMChain(llm=llm, prompt=high_level_chat_prompt)
                                                    high_level_summary = high_level_chain.run(user_prompt=final_prompt).strip()

                                            # Store both versions of the summary
                                            st.session_state['high_level_summary'] = high_level_summary
                                            st.session_state['enhanced_summary'] = high_level_summary

                                            # Set flag to indicate summarization is complete
                                            st.session_state['summarization_completed'] = True

                                            # Update the display without rerunning
                                            st.write("### High-Level Summary:")
                                            st.markdown(high_level_summary, unsafe_allow_html=True)

                                        # Display cluster summaries
                                        st.write("### Cluster Summaries:")
                                        if enable_references and 'Enhanced_Summary' in summary_df.columns:
                                            for idx, row in summary_df.iterrows():
                                                cluster_name = row.get('Cluster_Name', 'Unnamed Cluster')
                                                st.write(f"**Topic {row['Topic']} - {cluster_name}**")
                                                st.markdown(row['Enhanced_Summary'], unsafe_allow_html=True)
                                                st.write("---")
                                            with st.expander("View original summaries in table format"):
                                                display_df = summary_df[['Topic', 'Cluster_Name', 'Summary']]
                                                display_df.columns = ['Topic', 'Cluster Name', 'Summary']
                                                st.dataframe(display_df, hide_index=True)
                                        else:
                                            st.write("### Summaries per Cluster:")
                                            if 'Cluster_Name' in summary_df.columns:
                                                display_df = summary_df[['Topic', 'Cluster_Name', 'Summary']]
                                                display_df.columns = ['Topic', 'Cluster Name', 'Summary']
                                                st.dataframe(display_df, hide_index=True)
                                            else:
                                                st.dataframe(summary_df, hide_index=True)

                                        # Download
                                        if 'Enhanced_Summary' in summary_df.columns:
                                            dl_df = summary_df[['Topic', 'Cluster_Name', 'Summary']]
                                            dl_df.columns = ['Topic', 'Cluster Name', 'Summary']
                                        else:
                                            dl_df = summary_df
                                        csv_bytes = dl_df.to_csv(index=False).encode('utf-8')
                                        b64 = base64.b64encode(csv_bytes).decode()
                                        href = f'<a href="data:file/csv;base64,{b64}" download="summaries.csv">Download Summaries CSV</a>'
                                        st.markdown(href, unsafe_allow_html=True)

            # Display existing summaries if available and summarization was completed
            if st.session_state.get('summarization_completed', False):
                if 'summary_df' in st.session_state and not st.session_state['summary_df'].empty:
                    if 'high_level_summary' in st.session_state:
                        st.write("### High-Level Summary:")
                        st.markdown(st.session_state['enhanced_summary'] if st.session_state.get('enhanced_summary') else st.session_state['high_level_summary'], unsafe_allow_html=True)

                    st.write("### Cluster Summaries:")
                    summary_df = st.session_state['summary_df']
                    if 'Enhanced_Summary' in summary_df.columns:
                        for idx, row in summary_df.iterrows():
                            cluster_name = row.get('Cluster_Name', 'Unnamed Cluster')
                            st.write(f"**Topic {row['Topic']} - {cluster_name}**")
                            st.markdown(row['Enhanced_Summary'], unsafe_allow_html=True)
                            st.write("---")
                        with st.expander("View original summaries in table format"):
                            display_df = summary_df[['Topic', 'Cluster_Name', 'Summary']]
                            display_df.columns = ['Topic', 'Cluster Name', 'Summary']
                            st.dataframe(display_df, hide_index=True)
                    else:
                        st.dataframe(summary_df, hide_index=True)

                    # Add download button for existing summaries
                    dl_df = summary_df[['Topic', 'Cluster_Name', 'Summary']] if 'Cluster_Name' in summary_df.columns else summary_df
                    if 'Cluster_Name' in dl_df.columns:
                        dl_df.columns = ['Topic', 'Cluster Name', 'Summary']
                    csv_bytes = dl_df.to_csv(index=False).encode('utf-8')
                    b64 = base64.b64encode(csv_bytes).decode()
                    href = f'<a href="data:file/csv;base64,{b64}" download="summaries.csv">Download Summaries CSV</a>'
                    st.markdown(href, unsafe_allow_html=True)
        else:
            st.warning("No data available for summarization.")

        # Display existing summaries if available (when returning to the tab)
        if not st.session_state.get('_summarization_button_clicked', False):  # Only show if not just generated
            if 'high_level_summary' in st.session_state:
                st.write("### Existing High-Level Summary:")
                if st.session_state.get('enhanced_summary'):
                    st.markdown(st.session_state['enhanced_summary'], unsafe_allow_html=True)
                    with st.expander("View original summary (without references)"):
                        st.write(st.session_state['high_level_summary'])
                else:
                    st.write(st.session_state['high_level_summary'])

            if 'summary_df' in st.session_state and not st.session_state['summary_df'].empty:
                st.write("### Existing Cluster Summaries:")
                summary_df = st.session_state['summary_df']
                if 'Enhanced_Summary' in summary_df.columns:
                    for idx, row in summary_df.iterrows():
                        cluster_name = row.get('Cluster_Name', 'Unnamed Cluster')
                        st.write(f"**Topic {row['Topic']} - {cluster_name}**")
                        st.markdown(row['Enhanced_Summary'], unsafe_allow_html=True)
                        st.write("---")
                    with st.expander("View original summaries in table format"):
                        display_df = summary_df[['Topic', 'Cluster_Name', 'Summary']]
                        display_df.columns = ['Topic', 'Cluster Name', 'Summary']
                        st.dataframe(display_df, hide_index=True)
                else:
                    st.dataframe(summary_df, hide_index=True)

                # Add download button for existing summaries
                dl_df = summary_df[['Topic', 'Cluster_Name', 'Summary']] if 'Cluster_Name' in summary_df.columns else summary_df
                if 'Cluster_Name' in dl_df.columns:
                    dl_df.columns = ['Topic', 'Cluster Name', 'Summary']
                csv_bytes = dl_df.to_csv(index=False).encode('utf-8')
                b64 = base64.b64encode(csv_bytes).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="summaries.csv">Download Summaries CSV</a>'
                st.markdown(href, unsafe_allow_html=True)


    ###############################################################################
    # Tab: Chat
    ###############################################################################
    with tab_chat:
        st.header("Chat with Your Data")
        
        # Add explanation about chat functionality
        with st.expander("ℹ️ How Chat Works", expanded=False):
            st.markdown("""
            ### Understanding Chat with Your Data

            The chat functionality allows you to have an interactive conversation about your data, whether it's filtered, clustered, or raw. Here's how it works:

            1. **Data Selection**:
            - Choose which dataset to chat about (filtered, clustered, or search results)
            - Optionally focus on specific clusters if clustering was performed
            - System automatically includes relevant context from your selection

            2. **Context Window**:
            - Shows how much of the GPT-4 context window is being used
            - Helps you understand if you need to filter data further
            - Displays token usage statistics

            3. **Chat Features**:
            - Ask questions about your data
            - Get insights and analysis
            - Reference specific documents or clusters
            - Download chat context for transparency

            ### Best Practices

            1. **Data Selection**:
            - Start with filtered or clustered data for more focused conversations
            - Select specific clusters if you want to dive deep into a topic
            - Consider the context window usage when selecting data

            2. **Asking Questions**:
            - Be specific in your questions
            - Ask about patterns, trends, or insights
            - Reference clusters or documents by their IDs
            - Build on previous questions for deeper analysis

            3. **Managing Context**:
            - Monitor the context window usage
            - Filter data further if context is too full
            - Download chat context for documentation
            - Clear chat history to start fresh

            ### Tips for Better Results

            - **Question Types**:
            - "What are the main themes in cluster 3?"
            - "Compare the findings between clusters 1 and 2"
            - "Summarize the methodology used across these documents"
            - "What are the common outcomes reported?"

            - **Follow-up Questions**:
            - Build on previous answers
            - Ask for clarification
            - Request specific examples
            - Explore relationships between findings
            """)

        # Function to check data source availability
        def get_available_data_sources():
            sources = []
            if 'filtered_df' in st.session_state and not st.session_state['filtered_df'].empty:
                sources.append("Filtered Dataset")
            if 'clustered_data' in st.session_state and not st.session_state['clustered_data'].empty:
                sources.append("Clustered Data")
            if 'search_results' in st.session_state and not st.session_state['search_results'].empty:
                sources.append("Search Results")
            if ('high_level_summary' in st.session_state or 
                ('summary_df' in st.session_state and not st.session_state['summary_df'].empty)):
                sources.append("Summarized Data")
            return sources

        # Get available data sources
        available_sources = get_available_data_sources()
        
        if not available_sources:
            st.warning("No data available for chat. Please filter, cluster, search, or summarize first.")
            st.stop()

        # Initialize or update data source in session state
        if 'chat_data_source' not in st.session_state:
            st.session_state.chat_data_source = available_sources[0]
        elif st.session_state.chat_data_source not in available_sources:
            st.session_state.chat_data_source = available_sources[0]

        # Data source selection with automatic fallback
        data_source = st.radio(
            "Select data to chat about:",
            available_sources,
            index=available_sources.index(st.session_state.chat_data_source),
            help="Choose which dataset you want to analyze in the chat."
        )

        # Update session state if data source changed
        if data_source != st.session_state.chat_data_source:
            st.session_state.chat_data_source = data_source
            # Clear any cluster-specific selections if switching data sources
            if 'chat_selected_cluster' in st.session_state:
                del st.session_state.chat_selected_cluster

        # Get the appropriate DataFrame based on selected source
        df_chat = None
        if data_source == "Filtered Dataset":
            df_chat = st.session_state['filtered_df']
        elif data_source == "Clustered Data":
            df_chat = st.session_state['clustered_data']
        elif data_source == "Search Results":
            df_chat = st.session_state['search_results']
        elif data_source == "Summarized Data":
            # Create DataFrame with selected summaries
            summary_rows = []
            
            # Add high-level summary if available
            if 'high_level_summary' in st.session_state:
                summary_rows.append({
                    'Summary_Type': 'High-Level Summary',
                    'Content': st.session_state.get('enhanced_summary', st.session_state['high_level_summary'])
                })
            
            # Add cluster summaries if available
            if 'summary_df' in st.session_state and not st.session_state['summary_df'].empty:
                summary_df = st.session_state['summary_df']
                for _, row in summary_df.iterrows():
                    summary_rows.append({
                        'Summary_Type': f"Cluster {row['Topic']} Summary",
                        'Content': row.get('Enhanced_Summary', row['Summary'])
                    })
            
            if summary_rows:
                df_chat = pd.DataFrame(summary_rows)

        if df_chat is not None and not df_chat.empty:
            # If we have clustered data, allow cluster selection
            selected_cluster = None
            if data_source != "Summarized Data" and 'Topic' in df_chat.columns:
                cluster_option = st.radio(
                    "Choose cluster scope:",
                    ["All Clusters", "Specific Cluster"]
                )
                if cluster_option == "Specific Cluster":
                    unique_topics = sorted(df_chat['Topic'].unique())
                    # Check if we have cluster names
                    if 'summary_df' in st.session_state and 'Cluster_Name' in st.session_state['summary_df'].columns:
                        summary_df = st.session_state['summary_df']
                        # Create a mapping of topic to name
                        topic_names = {t: name for t, name in zip(summary_df['Topic'], summary_df['Cluster_Name'])}
                        # Format the selectbox options
                        topic_options = [
                            (t, f"Cluster {t} - {topic_names.get(t, 'Unnamed Cluster')}")
                            for t in unique_topics
                        ]
                        selected_cluster = st.selectbox(
                            "Select cluster to focus on:",
                            [t[0] for t in topic_options],
                            format_func=lambda x: next(opt[1] for opt in topic_options if opt[0] == x)
                        )
                    else:
                        selected_cluster = st.selectbox(
                            "Select cluster to focus on:",
                            unique_topics,
                            format_func=lambda x: f"Cluster {x}"
                        )
                    if selected_cluster is not None:
                        df_chat = df_chat[df_chat['Topic'] == selected_cluster]
                        st.session_state.chat_selected_cluster = selected_cluster
                elif 'chat_selected_cluster' in st.session_state:
                    del st.session_state.chat_selected_cluster

            # Prepare the data for chat context
            text_columns = st.session_state.get('text_columns', [])
            if not text_columns and data_source != "Summarized Data":
                st.warning("No text columns selected. Please select text columns to enable chat functionality.")
                st.stop()

            # Instead of limiting to 210 documents, we'll limit by tokens
            MAX_ALLOWED_TOKENS = int(MAX_CONTEXT_WINDOW * 0.95)  # 95% of context window
            
            # Prepare system message first to account for its tokens
            system_msg = {
                "role": "system",
                "content": """You are a specialized assistant analyzing data from a research database. 
    Your role is to:
    1. Provide clear, concise answers based on the data provided
    2. Highlight relevant information from specific results when answering
    3. When referencing specific results, use their row index or ID if available
    4. Clearly state if information is not available in the results
    5. Maintain a professional and analytical tone
    6. Format your responses using Markdown:
    - Use **bold** for emphasis
    - Use bullet points and numbered lists for structured information
    - Create tables using Markdown syntax when presenting structured data
    - Use backticks for code or technical terms
    - Include hyperlinks when referencing external sources
    - Use headings (###) to organize long responses

    The data is provided in a structured format where:""" + ("""
    - Each result contains multiple fields
    - Text content is primarily in the following columns: """ + ", ".join(text_columns) + """
    - Additional metadata and fields are available for reference
    - If clusters are present, they are numbered (e.g., Cluster 0, Cluster 1, etc.)""" if data_source != "Summarized Data" else """
    - The data consists of AI-generated summaries of the documents
    - Each summary may contain references to source documents in markdown format
    - References are shown as [ID] or as clickable hyperlinks
    - Summaries may be high-level (covering all documents) or cluster-specific""") + """
    """
            }

            # Calculate system message tokens
            system_tokens = len(tokenizer(system_msg["content"])["input_ids"])
            remaining_tokens = MAX_ALLOWED_TOKENS - system_tokens

            # Prepare the data context with token limiting
            data_text = "Available Data:\n"
            included_rows = 0
            total_rows = len(df_chat)

            if data_source == "Summarized Data":
                # For summarized data, process row by row
                for idx, row in df_chat.iterrows():
                    row_text = f"\n{row['Summary_Type']}:\n{row['Content']}\n"
                    row_tokens = len(tokenizer(row_text)["input_ids"])
                    
                    if remaining_tokens - row_tokens > 0:
                        data_text += row_text
                        remaining_tokens -= row_tokens
                        included_rows += 1
                    else:
                        break
            else:
                # For regular data, process row by row
                for idx, row in df_chat.iterrows():
                    row_text = f"\nItem {idx}:\n"
                    for col in df_chat.columns:
                        if not pd.isna(row[col]) and str(row[col]).strip() and col != 'similarity_score':
                            row_text += f"{col}: {row[col]}\n"
                    
                    row_tokens = len(tokenizer(row_text)["input_ids"])
                    if remaining_tokens - row_tokens > 0:
                        data_text += row_text
                        remaining_tokens -= row_tokens
                        included_rows += 1
                    else:
                        break

            # Calculate token usage
            data_tokens = len(tokenizer(data_text)["input_ids"])
            total_tokens = system_tokens + data_tokens
            context_usage_percent = (total_tokens / MAX_CONTEXT_WINDOW) * 100

            # Display token usage and data coverage
            st.subheader("Context Window Usage")
            st.write(f"System Message: {system_tokens:,} tokens")
            st.write(f"Data Context: {data_tokens:,} tokens")
            st.write(f"Total: {total_tokens:,} tokens ({context_usage_percent:.1f}% of available context)")
            st.write(f"Documents included: {included_rows:,} out of {total_rows:,} ({(included_rows/total_rows*100):.1f}%)")
            
            if context_usage_percent > 90:
                st.warning("⚠️ High context usage! Consider reducing the number of results or filtering further.")
            elif context_usage_percent > 75:
                st.info("ℹ️ Moderate context usage. Still room for your question, but consider reducing results if asking a long question.")

            # Add download button for chat context
            chat_context = f"""System Message:
    {system_msg['content']}

    {data_text}"""
            st.download_button(
                label="📥 Download Chat Context",
                data=chat_context,
                file_name="chat_context.txt",
                mime="text/plain",
                help="Download the exact context that the chatbot receives"
            )

            # Chat interface
            col_chat1, col_chat2 = st.columns([3, 1])
            with col_chat1:
                user_input = st.text_area("Ask a question about your data:", key="chat_input")
            with col_chat2:
                if st.button("Clear Chat History"):
                    st.session_state.chat_history = []
                    st.rerun()

            # Store current tab index before processing
            current_tab = tabs_titles.index("Chat")
            
            if st.button("Send", key="send_button"):
                if user_input:
                    # Set the active tab index to stay on Chat
                    st.session_state.active_tab_index = current_tab
                    
                    with st.spinner("Processing your question..."):
                        # Add user's question to chat history
                        st.session_state.chat_history.append({"role": "user", "content": user_input})
                        
                        # Prepare messages for API call
                        messages = [system_msg]
                        messages.append({"role": "user", "content": f"Here is the data to reference:\n\n{data_text}\n\nUser question: {user_input}"})
                        
                        # Get response from OpenAI
                        response = get_chat_response(messages)
                        
                        if response:
                            st.session_state.chat_history.append({"role": "assistant", "content": response})

            # Display chat history
            st.subheader("Chat History")
            for message in st.session_state.chat_history:
                if message["role"] == "user":
                    st.write("**You:**", message["content"])
                else:
                    st.write("**Assistant:**")
                    st.markdown(message["content"], unsafe_allow_html=True)
                st.write("---")  # Add a separator between messages


    ###############################################################################
    # Tab: Internal Validation
    ###############################################################################

else:  # Simple view
    st.header("Automatic Mode")
    
    # Initialize session state for automatic view
    if 'df' not in st.session_state:
        default_dataset_path = os.path.join(BASE_DIR, 'input', 'export_data_table_results_20251203_101413CET.xlsx')
        df = load_default_dataset(default_dataset_path)
        if df is not None:
            st.session_state['df'] = df.copy()
            st.session_state['using_default_dataset'] = True
            st.session_state['filtered_df'] = df.copy()
            
            # Set default text columns if not already set
            if 'text_columns' not in st.session_state or not st.session_state['text_columns']:
                default_text_cols = []
                if 'Title' in df.columns and 'Description' in df.columns:
                    default_text_cols = ['Title', 'Description']
                st.session_state['text_columns'] = default_text_cols

    # Single search bar for automatic processing
    #st.write("Enter your query to automatically search, cluster, and summarize the results:")
    query = st.text_input("Write your query here:")

    

    
    if st.button("SNAP!"):
        if query.strip():
            # Step 1: Semantic Search
            st.write("### Step 1: Semantic Search")
            with st.spinner("Performing Semantic Search..."):
                text_columns = st.session_state.get('text_columns', [])
                if text_columns:
                    df_full = st.session_state['df']
                    embeddings, _ = load_or_compute_embeddings(
                        df_full,
                        st.session_state.get('using_default_dataset', False),
                        st.session_state.get('uploaded_file_name'),
                        text_columns
                    )
                    
                    if embeddings is not None:
                        model = get_embedding_model()
                        df_filtered = st.session_state['filtered_df'].fillna("")
                        search_texts = df_filtered[text_columns].agg(' '.join, axis=1).tolist()
                        
                        subset_indices = df_filtered.index
                        subset_embeddings = embeddings[subset_indices]
                        
                        query_embedding = model.encode([query], device=device)
                        similarities = cosine_similarity(query_embedding, subset_embeddings)[0]
                        
                        similarity_threshold = 0.35  # Default threshold
                        above_threshold_indices = np.where(similarities > similarity_threshold)[0]
                        
                        if len(above_threshold_indices) > 0:
                            selected_indices = subset_indices[above_threshold_indices]
                            results = df_filtered.loc[selected_indices].copy()
                            results['similarity_score'] = similarities[above_threshold_indices]
                            results.sort_values(by='similarity_score', ascending=False, inplace=True)
                            st.session_state['search_results'] = results.copy()
                            st.write(f"Found {len(results)} relevant documents")
                        else:
                            st.warning("No results found above the similarity threshold.")
                            st.stop()
                
            # Step 2: Clustering
            if 'search_results' in st.session_state and not st.session_state['search_results'].empty:
                st.write("### Step 2: Clustering")
                with st.spinner("Performing clustering..."):
                    df_to_cluster = st.session_state['search_results'].copy()
                    dfc = df_to_cluster.copy().fillna("")
                    dfc['text'] = dfc[text_columns].astype(str).agg(' '.join, axis=1)
                    
                    # Filter embeddings to those rows
                    selected_indices = dfc.index
                    embeddings_clustering = embeddings[selected_indices]
                    
                    # Basic cleaning
                    stop_words = set(stopwords.words('english'))
                    texts_cleaned = []
                    for text in dfc['text'].tolist():
                        try:
                            word_tokens = word_tokenize(text)
                            filtered_text = ' '.join([w for w in word_tokens if w.lower() not in stop_words])
                            texts_cleaned.append(filtered_text)
                        except Exception as e:
                            texts_cleaned.append(text)
                    
                    min_cluster_size = 5  # Default value
                    
                    try:
                        # Convert embeddings to CPU numpy if needed
                        if torch.is_tensor(embeddings_clustering):
                            embeddings_for_clustering = embeddings_clustering.cpu().numpy()
                        else:
                            embeddings_for_clustering = embeddings_clustering
                        
                        # Build the HDBSCAN model
                        hdbscan_model = HDBSCAN(
                            min_cluster_size=min_cluster_size, 
                            metric='euclidean', 
                            cluster_selection_method='eom'
                        )
                        
                        # Build the BERTopic model
                        topic_model = BERTopic(
                            embedding_model=get_embedding_model(),
                            hdbscan_model=hdbscan_model
                        )
                        
                        # Fit the model and get topics
                        topics, probs = topic_model.fit_transform(
                            texts_cleaned, 
                            embeddings=embeddings_for_clustering
                        )
                        
                        # Store results
                        dfc['Topic'] = topics
                        st.session_state['topic_model'] = topic_model
                        st.session_state['clustered_data'] = dfc.copy()
                        st.session_state['clustering_completed'] = True

                        # Display clustering results summary
                        unique_topics = sorted(list(set(topics)))
                        num_clusters = len([t for t in unique_topics if t != -1])  # Exclude noise cluster (-1)
                        noise_docs = len([t for t in topics if t == -1])
                        clustered_docs = len(topics) - noise_docs
                        
                        st.write(f"Found {num_clusters} distinct clusters")
                        #st.write(f"Documents successfully clustered: {clustered_docs}")
                        #if noise_docs > 0:
                        #    st.write(f"Documents not fitting in any cluster: {noise_docs}")
                        
                        # Show quick cluster overview
                        cluster_info = []
                        for t in unique_topics:
                            if t != -1:  # Skip noise cluster in the overview
                                cluster_docs = dfc[dfc['Topic'] == t]
                                count = len(cluster_docs)
                                top_words = topic_model.get_topic(t)
                                top_keywords = ", ".join([w[0] for w in top_words[:5]]) if top_words else "N/A"
                                cluster_info.append((t, count, top_keywords))
                        
                        if cluster_info:
                            #st.write("### Quick Cluster Overview:")
                            cluster_df = pd.DataFrame(cluster_info, columns=["Topic", "Count", "Top Keywords"])
                        #     st.dataframe(
                        #         cluster_df,
                        #         column_config={
                        #             "Topic": st.column_config.NumberColumn("Topic", help="Topic ID"),
                        #             "Count": st.column_config.NumberColumn("Count", help="Number of documents in this topic"),
                        #             "Top Keywords": st.column_config.TextColumn(
                        #                 "Top Keywords",
                        #                 help="Top 5 keywords that characterize this topic"
                        #             )
                        #         },
                        #         hide_index=True
                        #     )
                        
                        # Generate visualizations
                        try:
                            st.session_state['intertopic_distance_fig'] = topic_model.visualize_topics()
                        except Exception:
                            st.session_state['intertopic_distance_fig'] = None
                        
                        try:
                            st.session_state['topic_document_fig'] = topic_model.visualize_documents(
                                texts_cleaned, 
                                embeddings=embeddings_for_clustering
                            )
                        except Exception:
                            st.session_state['topic_document_fig'] = None
                        
                        try:
                            hierarchy = topic_model.hierarchical_topics(texts_cleaned)
                            st.session_state['hierarchy'] = hierarchy if hierarchy is not None else pd.DataFrame()
                            st.session_state['hierarchy_fig'] = topic_model.visualize_hierarchy()
                        except Exception:
                            st.session_state['hierarchy'] = pd.DataFrame()
                            st.session_state['hierarchy_fig'] = None
                            
                    except Exception as e:
                        st.error(f"An error occurred during clustering: {str(e)}")
                        st.stop()
                
                # Step 3: Summarization
                if st.session_state.get('clustering_completed', False):
                    st.write("### Step 3: Summarization")
                    
                    # Initialize OpenAI client
                    openai_api_key = os.environ.get('OPENAI_API_KEY')
                    if not openai_api_key:
                        st.error("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
                        st.stop()
                    
                    llm = ChatOpenAI(
                        api_key=openai_api_key, 
                        model_name='gpt-4o-mini',
                        temperature=0.7,
                        max_tokens=1000
                    )
                    
                    df_scope = st.session_state['clustered_data']
                    unique_selected_topics = df_scope['Topic'].unique()
                    
                    # Process summaries in parallel
                    with st.spinner("Generating summaries..."):
                        local_system_message = SystemMessagePromptTemplate.from_template("""You are an expert summarizer skilled in creating concise and relevant summaries.
You will be given text and an objective context. Please produce a clear, cohesive,
and thematically relevant summary. 
Focus on key points, insights, or patterns that emerge from the text.""")
                        local_human_message = HumanMessagePromptTemplate.from_template("{user_prompt}")
                        local_chat_prompt = ChatPromptTemplate.from_messages([local_system_message, local_human_message])
                        
                        # Find URL column if it exists
                        url_column = next((col for col in df_scope.columns if 'url' in col.lower() or 'link' in col.lower() or 'pdf' in col.lower()), None)
                        
                        summaries = process_summaries_in_parallel(
                            df_scope=df_scope,
                            unique_selected_topics=unique_selected_topics,
                            llm=llm,
                            chat_prompt=local_chat_prompt,
                            enable_references=True,
                            reference_id_column=df_scope.columns[0],
                            url_column=url_column,  # Add URL column for clickable links
                            max_workers=min(16, len(unique_selected_topics))
                        )
                        
                        if summaries:
                            summary_df = pd.DataFrame(summaries)
                            st.session_state['summary_df'] = summary_df
                            
                            # Display updated cluster overview
                            if 'Cluster_Name' in summary_df.columns:
                                st.write("### Updated Topic Overview:")
                                cluster_info = []
                                for t in unique_selected_topics:
                                    cluster_docs = df_scope[df_scope['Topic'] == t]
                                    count = len(cluster_docs)
                                    top_words = topic_model.get_topic(t)
                                    top_keywords = ", ".join([w[0] for w in top_words[:5]]) if top_words else "N/A"
                                    cluster_name = summary_df[summary_df['Topic'] == t]['Cluster_Name'].iloc[0]
                                    cluster_info.append((t, cluster_name, count, top_keywords))
                                
                                cluster_df = pd.DataFrame(cluster_info, columns=["Topic", "Cluster_Name", "Count", "Top Keywords"])
                                st.dataframe(
                                    cluster_df,
                                    column_config={
                                        "Topic": st.column_config.NumberColumn("Topic", help="Topic ID (-1 represents outliers)"),
                                        "Cluster_Name": st.column_config.TextColumn("Cluster Name", help="AI-generated name describing the cluster theme"),
                                        "Count": st.column_config.NumberColumn("Count", help="Number of documents in this topic"),
                                        "Top Keywords": st.column_config.TextColumn(
                                            "Top Keywords",
                                            help="Top 5 keywords that characterize this topic"
                                        )
                                    },
                                    hide_index=True
                                )
                            
                            # Generate and display high-level summary
                            with st.spinner("Generating high-level summary..."):
                                formatted_summaries = []
                                summary_batches = []
                                current_batch = []
                                current_batch_tokens = 0
                                MAX_SAFE_TOKENS = int(MAX_CONTEXT_WINDOW * 0.75)
                                
                                for _, row in summary_df.iterrows():
                                    summary_text = row.get('Enhanced_Summary', row['Summary'])
                                    formatted_summary = f"### Cluster {row['Topic']} Summary:\n\n{summary_text}"
                                    summary_tokens = len(tokenizer(formatted_summary)["input_ids"])
                                    
                                    if current_batch_tokens + summary_tokens > MAX_SAFE_TOKENS:
                                        if current_batch:
                                            summary_batches.append(current_batch)
                                            current_batch = []
                                            current_batch_tokens = 0
                                    
                                    current_batch.append(formatted_summary)
                                    current_batch_tokens += summary_tokens
                                
                                if current_batch:
                                    summary_batches.append(current_batch)
                                
                                # Process each batch separately first
                                batch_overviews = []
                                for i, batch in enumerate(summary_batches, 1):
                                    st.write(f"Processing summary batch {i} of {len(summary_batches)}...")
                                    batch_text = "\n\n".join(batch)
                                    batch_prompt = f"""Below are summaries from a subset of clusters from results made using Transformers NLP on a set of results from the CGIAR reporting system. Each summary contains references to source documents in the form of hyperlinked IDs like [ID] or <a href="...">ID</a>.

Please create a comprehensive overview that synthesizes these clusters so that both the main themes and findings are covered in an organized way. IMPORTANT: 
1. Preserve all hyperlinked references exactly as they appear in the input summaries
2. Maintain the HTML anchor tags (<a href="...">) intact when using information from the summaries
3. Keep the markdown formatting for better readability
4. Create clear sections with headings for different themes
5. Use bullet points or numbered lists where appropriate
6. Focus on synthesizing the main themes and findings

Here are the cluster summaries to synthesize:

{batch_text}"""
                                    
                                    high_level_chain = LLMChain(llm=llm, prompt=local_chat_prompt)
                                    batch_overview = high_level_chain.run(user_prompt=batch_prompt).strip()
                                    batch_overviews.append(batch_overview)
                                
                                # Now create the final synthesis
                                if len(batch_overviews) > 1:
                                    st.write("Generating final synthesis...")
                                    combined_overviews = "\n\n# Part ".join([f"{i+1}\n\n{overview}" for i, overview in enumerate(batch_overviews)])
                                    final_prompt = f"""Below are multiple overview summaries, each covering different aspects of CGIAR research results. Each part maintains its original references to source documents.

Please create a final comprehensive synthesis that:
1. Integrates the key themes and findings from all parts into a cohesive narrative
2. Preserves all hyperlinked references exactly as they appear
3. Maintains the HTML anchor tags (<a href="...">) intact
4. Uses clear section headings and structured formatting
5. Highlights cross-cutting themes and relationships between different aspects
6. Provides a clear introduction and conclusion

Here are the overviews to synthesize:

# Part 1

{combined_overviews}"""
                                    
                                    final_prompt_tokens = len(tokenizer(final_prompt)["input_ids"])
                                    if final_prompt_tokens > MAX_SAFE_TOKENS:
                                        # If too long, just combine with headers
                                        high_level_summary = "# Comprehensive Overview\n\n" + "\n\n# Part ".join([f"{i+1}\n\n{overview}" for i, overview in enumerate(batch_overviews)])
                                    else:
                                        high_level_chain = LLMChain(llm=llm, prompt=local_chat_prompt)
                                        high_level_summary = high_level_chain.run(user_prompt=final_prompt).strip()
                                else:
                                    # If only one batch, use its overview directly
                                    high_level_summary = batch_overviews[0]
                                
                                st.session_state['high_level_summary'] = high_level_summary
                                st.session_state['enhanced_summary'] = high_level_summary
                                
                                # Display summaries
                                st.write("### High-Level Summary:")
                                with st.expander("High-Level Summary", expanded=True):
                                    st.markdown(high_level_summary, unsafe_allow_html=True)

                                st.write("### Cluster Summaries:")
                                for idx, row in summary_df.iterrows():
                                    cluster_name = row.get('Cluster_Name', 'Unnamed Cluster')
                                    with st.expander(f"Topic {row['Topic']} - {cluster_name}", expanded=False):
                                        st.markdown(row.get('Enhanced_Summary', row['Summary']), unsafe_allow_html=True)
    st.markdown("##### About this tool")
    with st.expander("Click to expand/collapse", expanded=True):
        st.markdown("""
        This tool draws on CGIAR quality assured results data from 2022-2024 to provide verifiable responses to user questions around the themes and areas CGIAR has/is working on.
                    
        **Tips:**
        - **Craft a phrase** that describes your topic of interest (e.g., `"climate-smart agriculture"`, `"gender equality livestock"`).
        - Avoid writing full questions — **this is not a chatbot**.
        - Combine **related terms** for better results (e.g., `"irrigation water access smallholders"`).
        - Focus on **concepts or themes** — not single words like `"climate"` or `"yield"` alone.
        - Example good queries:
            - `"climate adaptation smallholder farming"`
            - `"digital agriculture innovations"`
            - `"nutrition-sensitive value chains"`

        **Example use case**:
        You're interested in CGIAR's contributions to **poverty reduction through improved maize varieties in Africa**.  
        A good search phrase would be:  
        👉 `"poverty reduction maize Africa"`  
        This will retrieve results related to improved crop varieties, livelihood outcomes, and region-specific interventions, even if the documents use different wording like *"enhanced maize genetics"*, *"smallholder income"*, or *"eastern Africa trials"*.
        """)