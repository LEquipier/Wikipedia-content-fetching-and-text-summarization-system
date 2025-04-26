# qa_engine.py
import os
import streamlit as st
import time
import requests
import logging

# Initialize logging
logging.basicConfig(level=logging.ERROR, filename='qa_engine_errors.log', filemode='a', format='%(asctime)s - %(levelname)s - %(message)s')

# Add debug information
st.write("Preparing Q&A functionality...")

# Global variable to store the model
qa_pipeline = None

# Use cache decorator to avoid reloading the model
@st.cache_resource
def load_qa_model():
    """Load the Q&A model"""
    try:
        from transformers import pipeline
        
        # Use a more lightweight model
        model_name = "deepset/roberta-base-squad2"
        
        # Show loading progress
        progress_bar = st.progress(0)
        for i in range(100):
            # Update progress bar
            time.sleep(0.01)
            progress_bar.progress(i + 1)
        
        # Load model using pipeline
        qa_pipeline = pipeline(
            "question-answering", 
            model=model_name,
            device=-1  # Use CPU
        )
        
        st.success("Q&A model loaded successfully!")
        return qa_pipeline
    except Exception as e:
        logging.error(f"Error loading Q&A model: {str(e)}")
        st.error(f"Error loading Q&A model: {str(e)}")
        st.info("Will use alternative Q&A method.")
        return None

# Try to load the model
with st.spinner("Loading Q&A model, this may take some time..."):
    qa_pipeline = load_qa_model()

def answer_question(question, context):
    """Answer the question"""
    if qa_pipeline is None:
        return "Q&A model not loaded correctly, please check error messages."
    
    try:
        with st.spinner("Generating answer..."):
            result = qa_pipeline(question=question, context=context)
            return result["answer"]
    except Exception as e:
        logging.error(f"Cannot answer this question: {str(e)}")
        st.error(f"Cannot answer this question: {str(e)}")
        return "Cannot answer this question, please try using a simpler question or retrieve content again."