import streamlit as st
import wikipedia
import re
import time
import torch
from text_summarizer import evaluate_summary, initialize_summarizer, summarize_text
from rouge_score import rouge_scorer

# Set page configuration - Must be at the top
st.set_page_config(
    page_title="Lightweight Wikipedia Content Fetcher",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.write(f"Using device: {device}")
if device.type == 'cuda':
    st.write(f"GPU Model: {torch.cuda.get_device_name(0)}")
    st.write("GPU Memory Usage:")
    st.write(f"Allocated: {round(torch.cuda.memory_allocated(0)/1024**3,1)} GB")
    st.write(f"Reserved: {round(torch.cuda.memory_reserved(0)/1024**3,1)} GB")

# Initialize summarizer
initialize_summarizer()

# Add CSS styles
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
    }
    .success-box {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        color: #0c5460;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Show loading status
with st.spinner("Application is starting, please wait..."):
    st.write("Initializing application...")
    time.sleep(1)
    st.write("✅ Application initialization completed!")

st.title("🤖 Lightweight Wikipedia Content Fetcher")

# Add sidebar
with st.sidebar:
    st.header("About")
    st.markdown("""
    This is a lightweight Wikipedia content fetcher that doesn't use Hugging Face models.
    It provides basic Wikipedia content retrieval and text summarization functionality.
    """)
    
    st.header("Settings")
    
    # Add summary length control
    st.header("Summary Settings")
    summary_length = st.slider(
        "Summary Length",
        min_value=100,
        max_value=300,
        value=200,
        step=50,
        help="Control the length of generated summaries, higher values result in longer summaries"
    )

# Main interface
col1, col2 = st.columns(2)

with col1:
    st.subheader("🔍 Step 1: Enter Wikipedia Entry")
    title = st.text_input("Enter entry name", "Boston University")
    
    if st.button("Get Content"):
        with st.spinner("Fetching Wikipedia content..."):
            try:
                # Set language
                wikipedia.set_lang("en")
                
                # Get full page content
                page = wikipedia.page(title)
                full_content = page.content
                wiki_summary = page.summary
                
                # Save to session state
                st.session_state["full_content"] = full_content
                st.session_state["wiki_summary"] = wiki_summary
                st.session_state["title"] = title
                
            except wikipedia.exceptions.DisambiguationError as e:
                options = e.options[:5]
                st.error(f"This entry has multiple possible meanings: {', '.join(options)}")
                st.info("Please try using a more specific entry name.")
            except wikipedia.exceptions.PageError:
                st.error(f"Entry '{title}' not found, please try a different entry name")
                st.info("Please try using a different entry name or check your network connection.")
            except Exception as e:
                st.error(f"Error fetching Wikipedia content: {str(e)}")
                st.info("Please try using a different entry name or check your network connection.")
    
    # Display Wikipedia official summary (regardless of whether the summarize button is clicked)
    if "wiki_summary" in st.session_state:
        st.markdown(f"""
        <div class='info-box'>
            <strong>📘 Wikipedia Official Summary:</strong><br>
            <small>This is the official Wikipedia summary that will be used as the evaluation standard</small><br>
            {st.session_state["wiki_summary"]}
        </div>
        """, unsafe_allow_html=True)
        
        # Display content length info
        st.info(f"""
        📏 Content Statistics:
        - Wikipedia Full Text Length: {len(st.session_state["full_content"].split())} words
        - Wikipedia Summary Length: {len(st.session_state["wiki_summary"].split())} words
        """)

with col2:
    st.subheader("📝 Step 2: Text Summarization")
    if "full_content" in st.session_state:
        # Display current entry being processed
        st.info(f"Current Entry: {st.session_state['title']}")
        
        if st.button("Summarize Content"):
            with st.spinner("Summarizing content..."):
                try:
                    # Use full content to generate summary
                    text = st.session_state["full_content"]
                    wiki_summary = st.session_state["wiki_summary"]
                    
                    # Set summary length parameters to match Wikipedia summary length
                    wiki_length = len(wiki_summary.split())
                    target_length = wiki_length
                    max_length = int(target_length * 1.1)  # Allow 10% longer than Wikipedia summary
                    min_length = int(target_length * 0.9)  # Allow 10% shorter than Wikipedia summary
                    
                    # Generate summary using pretrained model
                    pretrained_summary = summarize_text(
                        text, 
                        max_length=max_length, 
                        min_length=min_length, 
                        use_pretrained=True
                    )
                    
                    # Generate summary using fine-tuned model
                    custom_summary = summarize_text(
                        text, 
                        max_length=max_length, 
                        min_length=min_length, 
                        use_pretrained=False
                    )
                    
                    # Save generated summaries to session_state
                    st.session_state["pretrained_summary"] = pretrained_summary
                    st.session_state["custom_summary"] = custom_summary
                    
                    # Evaluate both models
                    pretrained_scores = evaluate_summary(wiki_summary, pretrained_summary)
                    custom_scores = evaluate_summary(wiki_summary, custom_summary)
                    
                    # Save evaluation results to session_state
                    st.session_state["pretrained_scores"] = pretrained_scores
                    st.session_state["custom_scores"] = custom_scores
                    
                    # Display evaluation results
                    st.write("### Evaluation Results")
                    
                    # Display text length information
                    st.info(f"""
                    📏 Summary Length Comparison:
                    - Wikipedia Summary Length: {wiki_length} words
                    - Pretrained Model Summary Length: {len(pretrained_summary.split())} words
                    - Fine-tuned Model Summary Length: {len(custom_summary.split())} words
                    - Target Length Range: {min_length} - {max_length} words
                    """)
                    
                    # Display scores in a user-friendly way
                    col_metric, col_pretrained, col_custom = st.columns(3)
                    with col_metric:
                        st.write("Evaluation Metric")
                        st.write("ROUGE-1 (Word Overlap)")
                        st.write("ROUGE-2 (Bigram Overlap)")
                        st.write("ROUGE-L (Longest Sequence)")
                        st.write("Paragraph Coverage")
                        st.write("Comprehensive Score")
                    with col_pretrained:
                        st.write("Pretrained Model")
                        st.write(f"{pretrained_scores['rouge-1']:.3f}")
                        st.write(f"{pretrained_scores['rouge-2']:.3f}")
                        st.write(f"{pretrained_scores['rouge-l']:.3f}")
                        st.write(f"{pretrained_scores['paragraph-coverage']:.3f}")
                        st.write(f"{pretrained_scores['comprehensive-score']:.3f}")
                    with col_custom:
                        st.write("Fine-tuned Model")
                        st.write(f"{custom_scores['rouge-1']:.3f}")
                        st.write(f"{custom_scores['rouge-2']:.3f}")
                        st.write(f"{custom_scores['rouge-l']:.3f}")
                        st.write(f"{custom_scores['paragraph-coverage']:.3f}")
                        st.write(f"{custom_scores['comprehensive-score']:.3f}")
                    
                    # Display summaries from both models
                    st.write("### Summary Comparison")
                    col_pretrained, col_custom = st.columns(2)
                    with col_pretrained:
                        st.markdown(f"""
                        <div class='info-box'>
                            <strong>📋 Pretrained Model Summary:</strong><br>
                            {pretrained_summary}
                        </div>
                        """, unsafe_allow_html=True)
                    with col_custom:
                        st.markdown(f"""
                        <div class='info-box'>
                            <strong>📋 Fine-tuned Model Summary:</strong><br>
                            {custom_summary}
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Add evaluation suggestions
                    if custom_scores['comprehensive-score'] < pretrained_scores['comprehensive-score']:
                        st.warning("⚠️ Fine-tuned model performance needs improvement, suggestions:")
                        st.write("- Increase training data volume")
                        st.write("- Adjust model parameters")
                        st.write("- Optimize training strategy")
                    else:
                        st.success("✅ Fine-tuned model performance is better than pretrained model!")
                    
                except Exception as e:
                    st.error(f"Error summarizing content: {str(e)}")
                    st.info("Please try retrieving content again or use a different entry.")

# Add footer
st.markdown("---")
st.markdown("Powered by Streamlit | Lightweight Version") 

def evaluate_summary(reference, hypothesis):
    """
    Comprehensive evaluation method: combining multiple evaluation metrics
    """
    # 1. Calculate basic ROUGE scores
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = scorer.score(reference, hypothesis)
    
    # 2. Calculate information density
    ref_words = len(reference.split())
    hyp_words = len(hypothesis.split())
    density_score = hyp_words / ref_words if ref_words > 0 else 0
    
    # 3. Calculate paragraph coverage
    ref_paragraphs = [p.strip() for p in reference.split('\n\n') if p.strip()]
    covered_paragraphs = sum(1 for p in ref_paragraphs if any(w in hypothesis for w in p.split()[:5]))
    coverage_score = covered_paragraphs / len(ref_paragraphs) if ref_paragraphs else 0
    
    # 4. Comprehensive score
    return {
        'rouge-1': rouge_scores['rouge1'].fmeasure,
        'rouge-2': rouge_scores['rouge2'].fmeasure,
        'rouge-l': rouge_scores['rougeL'].fmeasure,
        'information-density': density_score,
        'paragraph-coverage': coverage_score,
        'comprehensive-score': (
            rouge_scores['rouge1'].fmeasure * 0.4 +
            rouge_scores['rouge2'].fmeasure * 0.3 +
            rouge_scores['rougeL'].fmeasure * 0.2 +
            coverage_score * 0.1
        )
    } 