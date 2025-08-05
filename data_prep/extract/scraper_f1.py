import requests
from bs4 import BeautifulSoup
from datetime import datetime
import json
from urllib.parse import urlparse, urljoin
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RAW_DATA_DIR = Path("../../data/raw")
MAIN_DATA_FILE = "f1_news.json"
ADDITIONAL_SOURCES = [
    "https://www.formula1.com/en/latest",
    "https://en.wikipedia.org/wiki/Formula_One",
    "https://en.wikipedia.org/wiki/2024_Formula_One_World_Championship",
    "https://en.wikipedia.org/wiki/2025_Formula_One_World_Championship",
    "https://en.wikipedia.org/wiki/List_of_Formula_One_World_Drivers%27_Champions",
    "https://www.formula1.com/en/results/2025/races",
    "https://www.formula1.com/en/racing/2025"
]

class Scraper:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'Mozilla/5.0'})
        self.existing_urls = self._load_existing_urls()
        RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)  # Create directory if it doesn't exist

    def _load_existing_urls(self):
        file_path = RAW_DATA_DIR / MAIN_DATA_FILE
        if file_path.exists():
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return {article['metadata']['url'] for article in data}
        return set()

    def _scrape_url(self, url):
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Simple title extraction
            title = "No Title"
            if soup.find('h1'):
                title = soup.find('h1').get_text(strip=True)
            elif soup.find('title'):
                title = soup.find('title').get_text(strip=True)
            
            # Simple content extraction
            content_parts = []
            paragraphs = soup.find_all('p')
            for p in paragraphs:
                text = p.get_text(strip=True)
                if text and len(text) > 50:
                    content_parts.append(text)
                if len(content_parts) >= 5:  # Limit to first 5 good paragraphs
                    break
            
            content = '\n\n'.join(content_parts)
            domain = urlparse(url).netloc
            
            if not content:
                logger.warning(f"No content extracted from {url}")
                return None

            logger.info(f"Scraped {domain}: {title[:50]}...")
            
            article = {
                'id': f"{domain}_{hash(url) % 1000000}",
                'content': f"Title: {title}\nSource: {domain}\n\n{content}\n\nRead more: {url}",
                'metadata': {
                    'title': title,
                    'url': url,
                    'source': domain,
                    'scraped_at': datetime.now().isoformat()
                }
            }
            return article
        except Exception as e:
            logger.error(f"Error scraping {url}: {e}")
            return None

    def scrape_all(self, max_articles=50):
        articles = []
        for url in ADDITIONAL_SOURCES:
            if len(articles) >= max_articles:
                break
            
            article = self._scrape_url(url)
            if article and article['metadata']['url'] not in self.existing_urls:
                articles.append(article)
                logger.info(f"Added article: {article['metadata']['title'][:50]}...")
            
            time.sleep(2)  # Respectful delay between requests
        
        return articles

    def save_articles(self, articles):
        file_path = RAW_DATA_DIR / MAIN_DATA_FILE
        if not articles:
            logger.info("No new articles to save")
            return
            
        # Load existing articles
        existing_data = []
        if file_path.exists():
            with open(file_path, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
        
        # Add new articles
        all_articles = existing_data + articles
        
        # Save all articles
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(all_articles, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(articles)} new articles to {file_path}")
        logger.info(f"Total articles in file: {len(all_articles)}")

def main():
    scraper = Scraper()
    logger.info(f"Scraping {len(ADDITIONAL_SOURCES)} sources...")
    articles = scraper.scrape_all()
    scraper.save_articles(articles)
    logger.info(f"Completed scraping. Total articles: {len(articles)}")

if __name__ == "__main__":
    main()