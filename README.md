# SNAP - Semantic Network Analysis Platform

SNAP is a Streamlit application that allows you to explore, filter, search, cluster, and summarize textual datasets using advanced NLP techniques.

## Setup

1. Clone this repository:
```bash
git clone <your-repo-url>
cd snap-streamlit
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up your OpenAI API key:
Create a `.env` file in the root directory and add:
```
OPENAI_API_KEY=your_api_key_here
```

## Models and Data

The application uses several machine learning models that are stored locally in the `models` directory:
- GPT2 Tokenizer (for text processing)
- Sentence Transformer (for semantic search and embeddings)

On first run, these models will be downloaded automatically and stored locally. Subsequent runs will use the cached models.

## Running the App

1. Activate your virtual environment if not already active:
```bash
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Run the Streamlit app:
```bash
streamlit run app.py
```

## Features

- **Semantic Search**: Find relevant documents using natural language queries
- **Document Clustering**: Group similar documents together to identify themes
- **Text Summarization**: Generate summaries of document clusters
- **Interactive Chat**: Ask questions about your data and get AI-powered responses

## Troubleshooting

If you encounter any issues:

1. Check that all dependencies are installed correctly:
```bash
pip install -r requirements.txt --upgrade
```

2. Ensure the models directory exists and has write permissions:
```bash
mkdir -p models/tokenizer models/sentence_transformer
```

3. Clear the models cache and let them redownload:
```bash
rm -rf models/*  # On Windows: rmdir /s /q models
```

4. Check your internet connection if models need to be downloaded

## License

[Your License Here] 