# wikipedia_utils.py
import wikipedia
import re
import streamlit as st

def get_wikipedia_content(title="Python (programming language)", sentences=5, language="en"):
    """
    Get Wikipedia content
    
    Parameters:
        title (str): Wikipedia entry title
        sentences (int): Number of sentences to retrieve
        language (str): Language code, default is English
    
    Returns:
        str: Wikipedia content summary
    """
    try:
        # Set language
        st.write(f"Setting language to: {language}")
        wikipedia.set_lang(language)
        
        # Try to get content
        st.write(f"Retrieving content for entry '{title}'...")
        content = wikipedia.summary(title, sentences=sentences)
        
        # Clean content (remove extra whitespace and special characters)
        content = re.sub(r'\s+', ' ', content).strip()
        
        st.write("Successfully retrieved Wikipedia content!")
        return content
    except wikipedia.exceptions.DisambiguationError as e:
        # Handle disambiguation pages
        options = e.options[:5]  # Only take the first 5 options
        error_msg = f"This entry has multiple possible meanings: {', '.join(options)}"
        st.error(error_msg)
        raise Exception(error_msg)
    except wikipedia.exceptions.PageError:
        # Handle page not found
        error_msg = f"Entry '{title}' not found, please try a different entry name"
        st.error(error_msg)
        raise Exception(error_msg)
    except Exception as e:
        # Handle other errors
        error_msg = f"Error fetching Wikipedia content: {str(e)}"
        st.error(error_msg)
        raise Exception(error_msg)