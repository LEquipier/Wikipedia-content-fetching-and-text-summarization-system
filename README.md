# Wikipedia Text Summarization System

This is a lightweight Wikipedia content fetching and text summarization system based on Streamlit and PyTorch. The system allows users to input Wikipedia entries, retrieve content, and generate summaries using both pre-trained and fine-tuned models, while providing detailed summary quality evaluation.

## Main Features

- Wikipedia content fetching and preprocessing
- Text summarization (supporting both pre-trained and fine-tuned models)
- Summary quality evaluation (ROUGE metrics, paragraph coverage, etc.)
- Model performance comparison
- User-friendly web interface

## System Architecture

1. **Data Collection Module** (`wiki_data_collector.py`)
   - Random Wikipedia article fetching
   - Data preprocessing and storage
   - Training data preparation

2. **Model Training Module** (`train_model.py`)
   - BART model fine-tuning
   - GPU training support
   - Training process monitoring and logging

3. **Text Summarization Module** (`text_summarizer.py`)
   - Pre-trained and fine-tuned model support
   - Summary length control
   - Summary quality evaluation

4. **Model Comparison Module** (`model_comparison.py`)
   - Pre-trained vs. fine-tuned model performance comparison
   - Multi-metric evaluation
   - Results visualization

5. **Web Application Module** (`light_app.py`)
   - User-friendly interface
   - Real-time summary generation
   - Summary quality evaluation display

## Code Details

### 1. Data Collection Module (`wiki_data_collector.py`)

```python
def get_random_wiki_articles(num_articles=10):
    """
    Fetch random Wikipedia articles
    - Set language to English
    - Display progress with progress bar
    - Ensure article length is sufficient (>1000 words)
    - Include error handling and retry mechanism
    """
    # Implementation code...

def save_articles_to_json(articles, filename="wiki_training_data.json"):
    """
    Save articles in JSON format
    - Create data directory
    - Save article content and summary
    - Output statistics
    """
    # Implementation code...
```

### 2. Model Training Module (`train_model.py`)

```python
def load_wiki_data(filepath="data/wiki_training_data.json"):
    """
    Load training data
    - Read JSON file
    - Prepare training samples
    - Convert to dataset format
    """
    # Implementation code...

def train_model(data, model_name="facebook/bart-large-cnn"):
    """
    Train summarization model
    - Load pre-trained model and tokenizer
    - Data preprocessing
    - Set training parameters
    - Execute training process
    - Save model
    """
    # Implementation code...
```

### 3. Text Summarization Module (`text_summarizer.py`)

```python
def initialize_summarizer():
    """
    Initialize summarizer
    - Load pre-trained model
    - Set device (CPU/GPU)
    - Configure model parameters
    """
    # Implementation code...

def summarize_text(text, max_length=200, min_length=100):
    """
    Generate text summary
    - Text preprocessing
    - Generate summary using model
    - Adjust summary length
    - Return processed summary
    """
    # Implementation code...

def evaluate_summary(reference, hypothesis):
    """
    Evaluate summary quality
    - Calculate ROUGE scores
    - Calculate information density
    - Calculate paragraph coverage
    - Generate comprehensive score
    """
    # Implementation code...
```

### 4. Model Comparison Module (`model_comparison.py`)

```python
def compare_models(num_articles=100):
    """
    Compare model performance
    - Fetch test articles
    - Generate summaries
    - Calculate evaluation metrics
    - Save comparison results
    """
    # Implementation code...
```

### 5. Web Application Module (`light_app.py`)

```python
# Page configuration
st.set_page_config(
    page_title="Lightweight Wikipedia Content Fetcher",
    page_icon="🤖",
    layout="wide"
)

# Main interface
def main():
    """
    Main application interface
    - Display title and description
    - Provide input fields
    - Display results
    - Handle user interaction
    """
    # Implementation code...
```

## Test Code

### 1. Data Collection Tests

```python
# Test fetching single article
def test_get_single_article():
    article = get_random_wiki_articles(1)
    assert len(article) == 1
    assert 'title' in article[0]
    assert 'content' in article[0]

# Test data saving
def test_save_articles():
    articles = [{"title": "Test", "content": "Test content"}]
    save_articles_to_json(articles, "test_data.json")
    assert os.path.exists("data/test_data.json")
```

### 2. Model Training Tests

```python
# Test data loading
def test_load_data():
    data = load_wiki_data()
    assert len(data) > 0
    assert 'text' in data.features
    assert 'summary' in data.features

# Test model training
def test_model_training():
    # Test training process with small dataset
    small_data = data.select(range(10))
    train_model(small_data)
    assert os.path.exists("models/wiki_summarizer")
```

### 3. Summary Generation Tests

```python
# Test summary generation
def test_summarize_text():
    text = "This is a test text for summarization."
    summary = summarize_text(text)
    assert len(summary) > 0
    assert len(summary) < len(text)

# Test summary evaluation
def test_evaluate_summary():
    reference = "This is a reference summary."
    hypothesis = "This is a generated summary."
    scores = evaluate_summary(reference, hypothesis)
    assert 'rouge-1' in scores
    assert 'comprehensive-score' in scores
```

### 4. Model Comparison Tests

```python
# Test model comparison
def test_model_comparison():
    results = compare_models(num_articles=5)
    assert 'pretrained' in results
    assert 'custom' in results
    assert len(results['pretrained']) > 0
```

## Installation Instructions

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-directory>
```

2. Install dependencies:
```bash
pip install -r requirement.txt
```

## Usage Process

1. Data collection (optional):
```bash
python wiki_data_collector.py
```

2. Model training (optional):
```bash
python train_model.py
```

3. Run the application:
```bash
streamlit run light_app.py
```

4. Access the application in your browser (default address: http://localhost:8501)

## Tech Stack

- [Streamlit](https://streamlit.io/) - Web application framework
- [PyTorch](https://pytorch.org/) - Deep learning framework
- [Transformers](https://huggingface.co/transformers/) - Pre-trained models
- [Wikipedia API](https://pypi.org/project/wikipedia/) - Wikipedia content fetching
- [ROUGE](https://pypi.org/project/rouge-score/) - Summary evaluation metrics

## Model Details

- **Pre-trained Model**: BART-large-cnn
- **Fine-tuned Model**: BART model fine-tuned on Wikipedia data
- **Evaluation Metrics**:
  - ROUGE-1, ROUGE-2, ROUGE-L
  - Paragraph coverage
  - Information density
  - Comprehensive score

## Notes

- Initial run will download pre-trained models, which may take some time
- Model training requires significant computational resources, GPU recommended
- Data collection process may take considerable time, recommended to do in batches
- Application supports custom summary length, adjustable as needed

## License

MIT 
