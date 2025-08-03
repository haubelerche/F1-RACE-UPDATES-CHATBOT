
import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any
from loguru import logger
import json
import time
from urllib.parse import urljoin, urlparse
import re
import schedule
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.config import F1_OFFICIAL_URL, F1_PAGINATED_URL, RAW_DATA_DIR, MAX_ARTICLES_PER_SCRAPE

# Constants for continuous data collection
MAIN_DATA_FILE = "f1_news.json"

class F1NewsScraper:
    """
    Scraper for Formula1.com news and articles with pagination support
    """
    
    def __init__(self):
        self.base_url = "https://www.formula1.com"
        self.news_url = F1_OFFICIAL_URL
        self.paginated_url = F1_PAGINATED_URL
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        self.scraped_articles = set()  # Track scraped articles to avoid duplicates
        self.main_data_file = RAW_DATA_DIR / MAIN_DATA_FILE
        self.existing_articles = self._load_existing_articles()
    
    def _load_existing_articles(self) -> Dict[str, Dict]:
        """Load existing articles from the main data file"""
        if not self.main_data_file.exists():
            return {}
        
        try:
            with open(self.main_data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Convert list to dict with URL as key for faster lookup
                return {article.get('metadata', {}).get('url', ''): article for article in data}
        except Exception as e:
            logger.error(f"Error loading existing articles: {e}")
            return {}
    
    def _is_duplicate_article(self, article_url: str) -> bool:
        """Check if article already exists"""
        return article_url in self.existing_articles
    
    def scrape_latest_news(self, max_articles: int = MAX_ARTICLES_PER_SCRAPE, max_pages: int = 5) -> List[Dict[str, Any]]:
        """
        Scrape latest F1 news articles from multiple pages
        
        Args:
            max_articles: Maximum number of articles to scrape
            max_pages: Maximum number of pages to scrape
            
        Returns:
            List of article data
        """
        all_articles = []
        
        try:
            logger.info(f"Scraping F1 news from paginated URL: {self.paginated_url}")
            
            # Scrape multiple pages
            for page_num in range(1, max_pages + 1):
                logger.info(f"Scraping page {page_num}")
                
                page_articles = self._scrape_page(page_num)
                all_articles.extend(page_articles)
                
                # Stop if we have enough articles
                if len(all_articles) >= max_articles:
                    all_articles = all_articles[:max_articles]
                    break
                
                # Add delay between pages
                time.sleep(2)
            
            # Remove duplicates based on URL
            unique_articles = self._deduplicate_articles(all_articles)
            
            logger.info(f"Successfully scraped {len(unique_articles)} unique articles from {page_num} pages")
            
        except Exception as e:
            logger.error(f"Error scraping news: {e}")
        
        return unique_articles
    
    def _scrape_page(self, page_number: int) -> List[Dict[str, Any]]:
        """Scrape a specific page of F1 news"""
        articles = []
        
        try:
            url = f"{self.paginated_url}{page_number}"
            logger.debug(f"Scraping URL: {url}")
            
            response = self.session.get(url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract article links from this page
            article_links = self._extract_article_links(soup)
            
            logger.info(f"Found {len(article_links)} article links on page {page_number}")
            
            # Scrape each article with threading for better performance
            with ThreadPoolExecutor(max_workers=3) as executor:
                future_to_url = {executor.submit(self._scrape_article, link): link for link in article_links}
                
                for future in as_completed(future_to_url):
                    try:
                        article_data = future.result()
                        if article_data:
                            articles.append(article_data)
                    except Exception as e:
                        url = future_to_url[future]
                        logger.error(f"Error scraping article {url}: {e}")
            
        except Exception as e:
            logger.error(f"Error scraping page {page_number}: {e}")
        
        return articles
    
    def _deduplicate_articles(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate articles based on URL"""
        seen_urls = set()
        unique_articles = []
        
        for article in articles:
            url = article.get('metadata', {}).get('url', '')
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique_articles.append(article)
        
        return unique_articles
    
    def _extract_article_links(self, soup: BeautifulSoup) -> List[str]:
        """Extract article links from the news page"""
        links = []
        
        # Common selectors for F1 website (may need adjustment)
        selectors = [
            'a[href*="/en/latest/article/"]',
            'a[href*="/en/latest/news/"]',
            '.listing-item a',
            '.teaser a',
            '.card a'
        ]
        
        for selector in selectors:
            elements = soup.select(selector)
            for element in elements:
                href = element.get('href')
                if href:
                    # Convert relative URLs to absolute
                    full_url = urljoin(self.base_url, href)
                    if self._is_valid_article_url(full_url):
                        links.append(full_url)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_links = []
        for link in links:
            if link not in seen:
                seen.add(link)
                unique_links.append(link)
        
        return unique_links
    
    def _is_valid_article_url(self, url: str) -> bool:
        """Check if URL is a valid F1 article"""
        parsed = urlparse(url)
        path = parsed.path.lower()
        
        # Check if it's an F1 article URL
        return (
            'formula1.com' in parsed.netloc and
            ('/latest/' in path or '/news/' in path) and
            len(path.split('/')) > 3  # Ensure it's not just the main page
        )
    
    def _scrape_article(self, url: str) -> Dict[str, Any]:
        """Scrape individual article"""
        try:
            # Check if article already exists
            if self._is_duplicate_article(url):
                logger.debug(f"Skipping duplicate article: {url}")
                return None
            
            response = self.session.get(url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract article data
            title = self._extract_title(soup)
            content = self._extract_content(soup)
            date = self._extract_date(soup)
            category = self._extract_category(soup)
            
            if not title or not content:
                return None
            
            article_data = {
                'id': self._generate_article_id(url),
                'content': self._format_article_content(title, content, date, category, url),
                'metadata': {
                    'type': 'news_article',
                    'title': title,
                    'url': url,
                    'date': date,
                    'category': category,
                    'source': 'Formula1.com',
                    'scraped_at': datetime.now().isoformat()
                }
            }
            
            return article_data
            
        except Exception as e:
            logger.error(f"Error scraping article {url}: {e}")
            return None
    
    def _extract_title(self, soup: BeautifulSoup) -> str:
        """Extract article title"""
        selectors = [
            'h1.headline',
            'h1.title',
            'h1',
            '.article-title',
            '.headline'
        ]
        
        for selector in selectors:
            element = soup.select_one(selector)
            if element:
                return element.get_text().strip()
        
        # Fallback to page title
        title_tag = soup.find('title')
        if title_tag:
            return title_tag.get_text().strip()
        
        return ""
    
    def _extract_content(self, soup: BeautifulSoup) -> str:
        """Extract article content"""
        content_parts = []
        
        # Common content selectors
        selectors = [
            '.article-body',
            '.content-body',
            '.article-content',
            '.post-content',
            'article .content'
        ]
        
        content_container = None
        for selector in selectors:
            element = soup.select_one(selector)
            if element:
                content_container = element
                break
        
        if not content_container:
            # Fallback to finding paragraphs
            content_container = soup
        
        # Extract paragraphs
        paragraphs = content_container.find_all('p')
        for p in paragraphs:
            text = p.get_text().strip()
            if text and len(text) > 20:  # Filter out very short paragraphs
                content_parts.append(text)
        
        return '\n\n'.join(content_parts)
    
    def _extract_date(self, soup: BeautifulSoup) -> str:
        """Extract article date"""
        # Look for date in various formats
        selectors = [
            'time[datetime]',
            '.date',
            '.published',
            '.article-date',
            '[data-date]'
        ]
        
        for selector in selectors:
            element = soup.select_one(selector)
            if element:
                # Try datetime attribute first
                date_str = element.get('datetime')
                if not date_str:
                    date_str = element.get('data-date')
                if not date_str:
                    date_str = element.get_text().strip()
                
                if date_str:
                    # Try to parse and format date
                    try:
                        # Handle various date formats
                        for fmt in ['%Y-%m-%d', '%Y-%m-%dT%H:%M:%S', '%d %B %Y', '%B %d, %Y']:
                            try:
                                parsed_date = datetime.strptime(date_str[:19], fmt)
                                return parsed_date.isoformat()
                            except ValueError:
                                continue
                        
                        # If parsing fails, return as-is
                        return date_str
                    except:
                        return date_str
        
        return ""
    
    def _extract_category(self, soup: BeautifulSoup) -> str:
        """Extract article category"""
        selectors = [
            '.category',
            '.tag',
            '.article-category',
            '.breadcrumb a:last-child'
        ]
        
        for selector in selectors:
            element = soup.select_one(selector)
            if element:
                return element.get_text().strip()
        
        return "General"
    
    def _generate_article_id(self, url: str) -> str:
        """Generate unique ID for article"""
        # Extract meaningful part from URL
        parsed = urlparse(url)
        path_parts = [p for p in parsed.path.split('/') if p]
        
        if path_parts:
            # Use last part of path as ID
            article_id = path_parts[-1]
            # Clean up ID
            article_id = re.sub(r'[^\w\-]', '', article_id)
            return f"f1_article_{article_id}"
        
        # Fallback to hash
        return f"f1_article_{hash(url) % 1000000}"
    
    def _format_article_content(self, title: str, content: str, date: str, category: str, url: str) -> str:
        """Format article as readable text for knowledge base"""
        formatted = f"""Title: {title}
Date: {date}
Category: {category}
Source: Formula1.com

{content}

Read more: {url}"""
        
        return formatted


class F1DataCollector:
    """
    Collect and process F1 data from web scraping with scheduling support
    """
    
    def __init__(self):
        self.scraper = F1NewsScraper()
        self.is_running = False
        self.scheduler_thread = None
        self.main_data_file = RAW_DATA_DIR / MAIN_DATA_FILE
    
    def collect_news_data(self, max_articles: int = MAX_ARTICLES_PER_SCRAPE, max_pages: int = 5) -> List[Dict[str, Any]]:
        """Collect news data formatted for knowledge base"""
        try:
            # Scrape articles
            articles = self.scraper.scrape_latest_news(max_articles, max_pages)
            
            # Filter and process articles
            processed_articles = []
            for article in articles:
                if self._is_valid_article(article):
                    processed_articles.append(article)
            
            logger.info(f"Processed {len(processed_articles)} valid articles")
            return processed_articles
            
        except Exception as e:
            logger.error(f"Error collecting news data: {e}")
            return []
    
    def start_scheduled_scraping(self, interval_minutes: int = 30):
        """Start scheduled scraping every specified minutes"""
        if self.is_running:
            logger.warning("Scheduler is already running")
            return
        
        logger.info(f"Starting scheduled scraping every {interval_minutes} minutes")
        
        # Schedule the job
        schedule.every(interval_minutes).minutes.do(self._scheduled_scrape_job)
        
        # Start scheduler in a separate thread
        self.is_running = True
        self.scheduler_thread = threading.Thread(target=self._run_scheduler, daemon=True)
        self.scheduler_thread.start()
        
        # Run initial scrape
        self._scheduled_scrape_job()
    
    def stop_scheduled_scraping(self):
        """Stop scheduled scraping"""
        logger.info("Stopping scheduled scraping")
        self.is_running = False
        schedule.clear()
        
        if self.scheduler_thread and self.scheduler_thread.is_alive():
            self.scheduler_thread.join(timeout=5)
    
    def _run_scheduler(self):
        """Run the scheduler in a loop"""
        while self.is_running:
            try:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
            except Exception as e:
                logger.error(f"Scheduler error: {e}")
                time.sleep(60)
    
    def _scheduled_scrape_job(self):
        """Job function for scheduled scraping"""
        try:
            logger.info("Starting scheduled scrape job")
            
            # Collect articles
            articles = self.collect_news_data(max_articles=30, max_pages=3)
            
            if articles:
                # Save to main data file (append mode)
                self.save_data(articles, append_mode=True)
                logger.info(f"Scheduled scrape completed: {len(articles)} new articles processed")
            else:
                logger.info("No new articles found in scheduled scrape")
                
        except Exception as e:
            logger.error(f"Error in scheduled scrape job: {e}")
    
    def _is_valid_article(self, article: Dict) -> bool:
        """Check if article is valid for knowledge base"""
        content = article.get('content', '')
        title = article.get('metadata', {}).get('title', '')
        
        # Basic validation
        return (
            len(content) > 100 and  # Minimum content length
            len(title) > 5 and      # Has meaningful title
            'f1' in content.lower() or 'formula' in content.lower()  # F1 related
        )
    
    def save_data(self, new_articles: List[Dict], append_mode: bool = True):
        """Save collected articles to the main data file"""
        try:
            if not new_articles:
                logger.info("No new articles to save")
                return
            
            if append_mode and self.main_data_file.exists():
                # Load existing data
                with open(self.main_data_file, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                
                # Get existing URLs for duplicate check
                existing_urls = {article.get('metadata', {}).get('url', '') for article in existing_data}
                
                # Filter out duplicates from new articles
                truly_new_articles = []
                for article in new_articles:
                    url = article.get('metadata', {}).get('url', '')
                    if url not in existing_urls:
                        truly_new_articles.append(article)
                
                if truly_new_articles:
                    # Combine existing and new articles
                    all_articles = existing_data + truly_new_articles
                    logger.info(f"Adding {len(truly_new_articles)} new articles to existing {len(existing_data)} articles")
                else:
                    logger.info("No new unique articles found")
                    return
            else:
                all_articles = new_articles
                logger.info(f"Creating new data file with {len(new_articles)} articles")
            
            # Save combined data
            with open(self.main_data_file, 'w', encoding='utf-8') as f:
                json.dump(all_articles, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Successfully saved to {self.main_data_file} (Total: {len(all_articles)} articles)")
            
            # Update scraper's existing articles cache
            self.scraper.existing_articles = {
                article.get('metadata', {}).get('url', ''): article for article in all_articles
            }
            
        except Exception as e:
            logger.error(f"Error saving articles: {e}")


def main():
    """Main function for continuous web scraping every 30 minutes"""
    import argparse
    
    parser = argparse.ArgumentParser(description='F1 News Scraper - Continuous Mode')
    parser.add_argument('--interval', type=int, default=30, help='Scraping interval in minutes (default: 30)')
    parser.add_argument('--pages', type=int, default=5, help='Maximum pages to scrape (default: 5)')
    parser.add_argument('--articles', type=int, default=50, help='Maximum articles to scrape (default: 50)')
    parser.add_argument('--oneshot', action='store_true', help='Run once instead of continuous mode')
    
    args = parser.parse_args()
    
    collector = F1DataCollector()
    
    if args.oneshot:
        # One-time scraping
        logger.info("Running one-time scraping")
        
        # Collect news data
        articles = collector.collect_news_data(max_articles=args.articles, max_pages=args.pages)
        
        # Save data
        collector.save_data(articles, append_mode=True)
        
        # Print summary
        print(f"✅ Processed {len(articles)} articles and saved to {MAIN_DATA_FILE}")
        
        # Show sample
        if articles:
            print("\n📰 Sample article:")
            print(f"ID: {articles[0]['id']}")
            print(f"Title: {articles[0]['metadata']['title']}")
            print(f"URL: {articles[0]['metadata']['url']}")
            print(f"Content preview: {articles[0]['content'][:200]}...")
    else:
        # Continuous scraping mode (default)
        logger.info(f"Starting continuous scraping every {args.interval} minutes")
        print(f"🕒 F1 News Scraper running every {args.interval} minutes")
        print(f"📁 All data will be saved to: {MAIN_DATA_FILE}")
        print("🔄 Starting first scrape...")
        print("Press Ctrl+C to stop...")
        
        try:
            collector.start_scheduled_scraping(interval_minutes=args.interval)
            
            # Keep the main thread alive
            while True:
                time.sleep(10)
                
        except KeyboardInterrupt:
            logger.info("Received shutdown signal")
            collector.stop_scheduled_scraping()
            print("\n✅ Scraper stopped gracefully")


def run_scheduled_scraper(interval_minutes: int = 30):
    """Convenience function to start scheduled scraping"""
    collector = F1DataCollector()
    
    logger.info(f"Starting F1 News Scraper with {interval_minutes}-minute intervals")
    
    try:
        collector.start_scheduled_scraping(interval_minutes=interval_minutes)
        
        print(f"F1 News Scraper started!")
        print(f"Scraping every {interval_minutes} minutes")
        print(f"Data will be saved to: {MAIN_DATA_FILE}")
        print("First scrape starting now...")
        print("Press Ctrl+C to stop")
        
        # Keep running
        while True:
            time.sleep(10)
            
    except KeyboardInterrupt:
        logger.info("Shutting down scheduler...")
        collector.stop_scheduled_scraping()
        print("\nF1 News Scraper stopped")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        collector.stop_scheduled_scraping()
        raise


if __name__ == "__main__":
    main()
