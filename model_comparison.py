import json
import torch
from text_summarizer import summarize_text, evaluate_summary
from wiki_data_collector import get_random_wiki_articles
import numpy as np
from tqdm import tqdm
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def compare_models(num_articles=100):
    """
    Compare performance between pretrained and fine-tuned models on random Wikipedia articles
    """
    # Get random articles
    logger.info(f"Fetching {num_articles} random Wikipedia articles...")
    articles = get_random_wiki_articles(num_articles)
    
    # Initialize results storage
    pretrained_scores = {
        'rouge-1': [],
        'rouge-2': [],
        'rouge-l': [],
        'paragraph-coverage': [],
        'comprehensive-score': []
    }
    
    custom_scores = {
        'rouge-1': [],
        'rouge-2': [],
        'rouge-l': [],
        'paragraph-coverage': [],
        'comprehensive-score': []
    }
    
    # Process each article
    logger.info("Starting model comparison...")
    for i, article in enumerate(tqdm(articles)):
        try:
            # Get Wikipedia summary length
            wiki_length = len(article['summary'].split())
            target_length = wiki_length
            max_length = int(target_length * 1.1)  # Allow 10% longer
            min_length = int(target_length * 0.9)  # Allow 10% shorter
            
            # Generate summaries
            pretrained_summary = summarize_text(
                article['full_content'],
                max_length=max_length,
                min_length=min_length,
                use_pretrained=True
            )
            
            custom_summary = summarize_text(
                article['full_content'],
                max_length=max_length,
                min_length=min_length,
                use_pretrained=False
            )
            
            # Evaluate summaries
            pretrained_eval = evaluate_summary(article['summary'], pretrained_summary)
            custom_eval = evaluate_summary(article['summary'], custom_summary)
            
            # Store scores
            for metric in pretrained_scores.keys():
                pretrained_scores[metric].append(pretrained_eval[metric])
                custom_scores[metric].append(custom_eval[metric])
            
            # Log progress every 10 articles
            if (i + 1) % 10 == 0:
                logger.info(f"Processed {i + 1} articles")
                
        except Exception as e:
            logger.error(f"Error processing article {i + 1}: {str(e)}")
            continue
    
    # Calculate average scores
    logger.info("\nCalculating average scores...")
    results = {
        'pretrained': {metric: np.mean(scores) for metric, scores in pretrained_scores.items()},
        'custom': {metric: np.mean(scores) for metric, scores in custom_scores.items()}
    }
    
    # Print results
    logger.info("\n=== Model Comparison Results ===")
    logger.info(f"Total articles processed: {len(articles)}")
    logger.info("\nAverage Scores:")
    
    metrics = ['rouge-1', 'rouge-2', 'rouge-l', 'paragraph-coverage', 'comprehensive-score']
    for metric in metrics:
        pretrained_avg = results['pretrained'][metric]
        custom_avg = results['custom'][metric]
        improvement = ((custom_avg - pretrained_avg) / pretrained_avg) * 100
        
        logger.info(f"\n{metric.upper()}:")
        logger.info(f"Pretrained Model: {pretrained_avg:.4f}")
        logger.info(f"Custom Model: {custom_avg:.4f}")
        logger.info(f"Improvement: {improvement:+.2f}%")
    
    # Save results to file
    with open('model_comparison_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    logger.info("\nResults saved to 'model_comparison_results.json'")

if __name__ == "__main__":
    compare_models() 