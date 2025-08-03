
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"

# Create directories if they don't exist
for dir_path in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, EMBEDDINGS_DIR]:
    dir_path.mkdir(exist_ok=True)

# API URLs
OPENF1_BASE_URL = "https://api.openf1.org/v1"
F1_OFFICIAL_URL = "https://www.formula1.com/en/latest"
F1_PAGINATED_URL = "https://www.formula1.com/en/latest?page="

# Model settings
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ChromaDB settings
CHROMA_PERSIST_DIR = str(EMBEDDINGS_DIR)
COLLECTION_NAME = "f1_knowledge_base"

# Generation settings
MAX_CONTEXT_LENGTH = 4000
MAX_RESPONSE_LENGTH = 500
TEMPERATURE = 0.7

# Data update settings
UPDATE_INTERVAL_HOURS = 6
MAX_ARTICLES_PER_SCRAPE = 50

# Streamlit settings
APP_TITLE = "F1 Race Updates Chatbot"

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = PROJECT_ROOT / "logs" / "f1_chatbot.log"

# Create logs directory
LOG_FILE.parent.mkdir(exist_ok=True)