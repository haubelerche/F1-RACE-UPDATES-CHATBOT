"""
F1 News Scraper Scheduler
Runs the F1 news scraper on a schedule
"""
import sys
import time
from pathlib import Path
from loguru import logger

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data_prep.scraper_f1 import run_scheduled_scraper


def main():
    """Main scheduler entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='F1 News Scraper Scheduler')
    parser.add_argument(
        '--interval', 
        type=int, 
        default=30, 
        help='Scraping interval in minutes (default: 30)'
    )
    parser.add_argument(
        '--daemon', 
        action='store_true', 
        help='Run as daemon process'
    )
    
    args = parser.parse_args()
    
    logger.info(f"Starting F1 News Scraper Scheduler")
    logger.info(f"Interval: {args.interval} minutes")
    
    if args.daemon:
        logger.info("Running in daemon mode")
    
    try:
        run_scheduled_scraper(interval_minutes=args.interval)
    except KeyboardInterrupt:
        logger.info("Scheduler stopped by user")
    except Exception as e:
        logger.error(f"Scheduler error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
