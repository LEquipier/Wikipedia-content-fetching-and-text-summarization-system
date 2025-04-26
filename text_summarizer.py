# text_summarizer.py
import os
import streamlit as st
import time
import re
import logging
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from nltk.translate.meteor_score import meteor_score
from nltk.translate.bleu_score import sentence_bleu
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Initialize logging
logging.basicConfig(level=logging.ERROR, filename='text_summarizer_errors.log', filemode='a', format='%(asctime)s - %(levelname)s - %(message)s')

# Global variable to store the model
model = None
tokenizer = None

def initialize_summarizer():
    """Initialize the summarizer"""
    global model, tokenizer
    if model is None or tokenizer is None:
        # Load pretrained model
        pretrained_model_name = "facebook/bart-large-cnn"
        print("Loading pretrained model...")
        pretrained_tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
        pretrained_model = AutoModelForSeq2SeqLM.from_pretrained(pretrained_model_name)
        print("Pretrained model loaded successfully")
        
        # Load custom model
        try:
            # Use the same model path as in training code
            custom_model_path = "./models/wiki_summarizer"  # Path consistent with training code
            print(f"Attempting to load custom model, path: {custom_model_path}")
            
            if os.path.exists(custom_model_path):
                print("Found custom model directory, starting to load...")
                # Check if model files exist
                required_files = ['config.json', 'pytorch_model.bin', 'tokenizer.json']
                missing_files = [f for f in required_files if not os.path.exists(os.path.join(custom_model_path, f))]
                
                if missing_files:
                    print(f"Warning: Missing required model files: {missing_files}")
                    raise FileNotFoundError(f"Missing model files: {missing_files}")
                
                custom_tokenizer = AutoTokenizer.from_pretrained(custom_model_path, local_files_only=True)
                custom_model = AutoModelForSeq2SeqLM.from_pretrained(custom_model_path, local_files_only=True)
                print("Custom model loaded successfully")
            else:
                print(f"Warning: Custom model directory not found: {custom_model_path}")
                raise FileNotFoundError(f"Model directory not found: {custom_model_path}")
                
        except Exception as e:
            print(f"Error loading custom model: {str(e)}")
            print("Will use pretrained model as fallback")
            custom_tokenizer = pretrained_tokenizer
            custom_model = pretrained_model
        
        # Move models to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        pretrained_model = pretrained_model.to(device)
        custom_model = custom_model.to(device)
        
        # Save models and tokenizers
        model = {
            "pretrained": pretrained_model,
            "custom": custom_model
        }
        tokenizer = {
            "pretrained": pretrained_tokenizer,
            "custom": custom_tokenizer
        }
        
        # Print model information
        print("\nModel Information:")
        print(f"Pretrained Model: {pretrained_model_name}")
        print(f"Custom Model Path: {custom_model_path}")
        print(f"Pretrained Model Parameters: {sum(p.numel() for p in pretrained_model.parameters())}")
        print(f"Custom Model Parameters: {sum(p.numel() for p in custom_model.parameters())}")

def summarize_text(text, max_length=200, min_length=100, use_pretrained=False):
    """Generate text summary"""
    if model is None or tokenizer is None:
        initialize_summarizer()
    
    # Select model and tokenizer
    model_type = "pretrained" if use_pretrained else "custom"
    current_model = model[model_type]
    current_tokenizer = tokenizer[model_type]
    
    print(f"\nUsing model type: {model_type}")
    print(f"Input text length: {len(text)}")
    print(f"Target length range: {min_length} - {max_length} words")
    
    # Encode input text
    inputs = current_tokenizer(text, max_length=1024, return_tensors="pt", truncation=True)
    inputs = {k: v.to(current_model.device) for k, v in inputs.items()}
    
    # Generate summary
    summary_ids = current_model.generate(
        inputs["input_ids"],
        max_length=max_length,
        min_length=min_length,
        num_beams=4,
        length_penalty=1.0,
        no_repeat_ngram_size=3,
        do_sample=False,
        temperature=1.0,
        top_k=50,
        top_p=0.95,
        early_stopping=True
    )
    
    # Decode summary
    summary = current_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    summary_length = len(summary.split())
    
    # If summary length differs significantly from target length, try adjusting parameters
    target_length = (max_length + min_length) // 2
    if abs(summary_length - target_length) > target_length * 0.15:  # If difference exceeds 15%
        print(f"Summary length ({summary_length}) differs significantly from target length ({target_length}), adjusting parameters")
        
        # Adjust parameters based on current length
        if summary_length > target_length:
            # If too long, increase length penalty
            length_penalty = 0.7
            max_length = int(max_length * 0.9)
        else:
            # If too short, decrease length penalty
            length_penalty = 1.3
            max_length = int(max_length * 1.1)
            
        print(f"Adjusted parameters: length_penalty={length_penalty}, max_length={max_length}")
        
        summary_ids = current_model.generate(
            inputs["input_ids"],
            max_length=max_length,
            min_length=min_length,
            num_beams=4,
            length_penalty=length_penalty,
            no_repeat_ngram_size=3,
            do_sample=False,
            temperature=1.0,
            top_k=50,
            top_p=0.95,
            early_stopping=True
        )
        summary = current_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        summary_length = len(summary.split())
    
    print(f"Final summary length: {summary_length} words")
    return summary

def setup_ui():
    """Setup the Streamlit UI for text summarization"""
    st.title("Text Summarization and Evaluation")
    
    # Step 1: Input text
    st.header("Step 1: Input Text")
    input_text = st.text_area("Enter text to summarize:")
    
    # Step 2: Summarize content
    st.header("Step 2: Summarize Content")
    if st.button("Summarize Content"):
        if input_text:
            # Generate summary
            summary = summarize_text(input_text)
            st.write("### Summary")
            st.write(summary)
            
            # Step 3: Evaluate summary
            st.header("Step 3: Evaluate Summary")
            reference_summary = st.text_area("Enter reference summary for evaluation:")
            if reference_summary:
                evaluation_scores = evaluate_summary(reference_summary, summary)
                st.write("### Evaluation Scores")
                st.json(evaluation_scores)
        else:
            st.error("Please enter text to summarize.")

def preprocess_text(text):
    """
    Preprocess text by removing Markdown, footnotes, and extra lines.
    
    Parameters:
        text (str): Original text
    
    Returns:
        str: Cleaned text
    """
    # Remove Markdown links and footnotes
    text = re.sub(r'\[.*?\]\(.*?\)', '', text)  # Remove Markdown links
    text = re.sub(r'\[\d+\]', '', text)  # Remove footnotes like [1]
    
    # Remove extra newlines
    text = re.sub(r'\n+', '\n', text)
    
    return text

def summarize_long_text(text, max_length=200, min_length=100):
    """
    Process summarization of long text
    
    Parameters:
        text (str): Long text to summarize
        max_length (int): Maximum length of the summary
        min_length (int): Minimum length of the summary
    
    Returns:
        str: Summarized text
    """
    # Split text into paragraphs
    paragraphs = text.split('\n\n')
    
    # If too many paragraphs, take more paragraphs
    if len(paragraphs) > 8:
        paragraphs = paragraphs[:8]
    
    # Summarize each paragraph
    summaries = []
    for para in paragraphs:
        if len(para.split()) > 50:  # Only summarize longer paragraphs
            try:
                with st.spinner("Summarizing paragraph..."):
                    para = preprocess_text(para)
                    result = summarize_text(para, max_length=max_length//2, min_length=min_length//2)
                    summaries.append(result)
            except:
                summaries.append(para)
        else:
            summaries.append(para)
    
    # Combine summaries
    return " ".join(summaries)

def simple_summarize(text, num_sentences=5):
    """
    Simple text summarization method (alternative)
    
    Parameters:
        text (str): Text to summarize
        num_sentences (int): Number of sentences to include in summary
    
    Returns:
        str: Summarized text
    """
    # Simple summarization method: take the first few sentences
    sentences = text.split('.')
    if len(sentences) > num_sentences:
        return '. '.join(sentences[:num_sentences]) + '.'
    return text 

def evaluate_summary(reference, hypothesis):
    """
    综合评估方法：结合多种评估指标
    
    Parameters:
        reference (str): The reference summary
        hypothesis (str): The generated summary
    
    Returns:
        dict: A dictionary with evaluation scores
    """
    # 1. 计算基础ROUGE分数
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = scorer.score(reference, hypothesis)
    
    # 2. 计算信息密度
    ref_words = len(reference.split())
    hyp_words = len(hypothesis.split())
    density_score = hyp_words / ref_words if ref_words > 0 else 0
    
    # 3. 计算段落覆盖率
    ref_paragraphs = [p.strip() for p in reference.split('\n\n') if p.strip()]
    covered_paragraphs = sum(1 for p in ref_paragraphs if any(w in hypothesis for w in p.split()[:5]))
    coverage_score = covered_paragraphs / len(ref_paragraphs) if ref_paragraphs else 0
    
    # 4. 综合评分
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