# question_gen.py
import os
import streamlit as st
import time
import logging

# Initialize logging
logging.basicConfig(level=logging.ERROR, filename='question_gen_errors.log', filemode='a', format='%(asctime)s - %(levelname)s - %(message)s')

# Add debug information
st.write("Preparing question generation functionality...")

# Global variable to store the model
qg_pipeline = None

# Use cache decorator to avoid reloading the model
@st.cache_resource
def load_qg_model():
    """Load the question generation model"""
    try:
        from transformers import pipeline
        
        # Use a more lightweight model
        model_name = "google/flan-t5-small"
        
        # Show loading progress
        progress_bar = st.progress(0)
        for i in range(100):
            # Update progress bar
            time.sleep(0.01)
            progress_bar.progress(i + 1)
        
        # Load model using pipeline
        qg_pipeline = pipeline(
            "text2text-generation", 
            model=model_name,
            device=-1  # Use CPU
        )
        
        st.success("Question generation model loaded successfully!")
        return qg_pipeline
    except Exception as e:
        logging.error(f"Error loading question generation model: {str(e)}")
        st.error(f"Error loading question generation model: {str(e)}")
        st.info("Will use alternative question generation method.")
        return None

# Try to load the model
with st.spinner("Loading question generation model, this may take some time..."):
    qg_pipeline = load_qg_model()

def generate_questions(summary):
    """Generate questions using the summary as a prompt"""
    if qg_pipeline is None:
        return ["What is the main content of this topic?", "What is the historical background of this topic?", "What are the application areas of this topic?"]
    
    try:
        # Use summary as prompt
        prompt = f"Generate questions about: {summary}"
        
        # Generate questions
        with st.spinner("Generating questions..."):
            result = qg_pipeline(prompt, max_length=100, num_return_sequences=3)
            
            # Extract questions
            questions = [item["generated_text"] for item in result]
            
            # If no questions generated, return default questions
            if not questions:
                return ["What is the main content of this topic?", "What is the historical background of this topic?", "What are the application areas of this topic?"]
            
            return questions
    except Exception as e:
        logging.error(f"Error generating questions: {str(e)}")
        st.error(f"Error generating questions: {str(e)}")
        # Return default questions on error
        return ["What is the main content of this topic?", "What is the historical background of this topic?", "What are the application areas of this topic?"]