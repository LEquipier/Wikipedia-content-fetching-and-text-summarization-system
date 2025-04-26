import wikipedia
import json
import time
import random
from tqdm import tqdm
import os
import warnings
from bs4 import BeautifulSoup

# Ignore BeautifulSoup warnings
warnings.filterwarnings("ignore", category=UserWarning, module='bs4')

def get_random_wiki_articles(num_articles=10):
    """
    Get random Wikipedia articles
    """
    # Set language
    wikipedia.set_lang("en")
    
    # Get random article list
    articles = []
    print(f"Starting to fetch {num_articles} Wikipedia articles...")
    
    # Show progress with progress bar
    with tqdm(total=num_articles) as pbar:
        while len(articles) < num_articles:
            try:
                # Get random article
                random_article = wikipedia.random(1)
                page = wikipedia.page(random_article)
                
                # Ensure article content is long enough (at least 1000 words)
                if len(page.content.split()) > 1000:
                    article_data = {
                        "title": page.title,
                        "url": page.url,
                        "full_content": page.content,
                        "summary": page.summary
                    }
                    articles.append(article_data)
                    pbar.update(1)
                    
                    # Add random delay to avoid too frequent requests
                    time.sleep(random.uniform(1, 3))
                    
            except wikipedia.exceptions.DisambiguationError:
                print(f"Skipping ambiguous page: {random_article}")
                continue
            except wikipedia.exceptions.PageError:
                print(f"Page not found: {random_article}")
                continue
            except Exception as e:
                print(f"Error fetching article: {str(e)}")
                time.sleep(5)  # Wait longer when error occurs
                continue
    
    return articles

def save_articles_to_json(articles, filename="wiki_training_data.json"):
    """
    Save articles to JSON file
    """
    # Ensure data directory exists
    if not os.path.exists("data"):
        os.makedirs("data")
    
    filepath = os.path.join("data", filename)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(articles, f, ensure_ascii=False, indent=2)
    
    print(f"\nData saved to: {filepath}")
    print(f"Total articles collected: {len(articles)}")
    print(f"Average article length: {sum(len(article['full_content'].split()) for article in articles) / len(articles):.0f} words")
    print(f"Average summary length: {sum(len(article['summary'].split()) for article in articles) / len(articles):.0f} words")

if __name__ == "__main__":
    try:
        # Get 100 random articles
        articles = get_random_wiki_articles(100)
        
        # Save to JSON file
        save_articles_to_json(articles)
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
        if articles:
            print(f"Saving collected {len(articles)} articles...")
            save_articles_to_json(articles)
    except Exception as e:
        print(f"Program error: {str(e)}")
        if articles:
            print(f"Saving collected {len(articles)} articles...")
            save_articles_to_json(articles) 