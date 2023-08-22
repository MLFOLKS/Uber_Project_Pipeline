import logging
import os
import sys
from datetime import datetime

logging_format = "[%(asctime)s: %(lineno)d : %(name)s : %(levelname)s : %(module)s : %(message)s]"
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
log_dir = os.path.join(os.getcwd(), 'logs', LOG_FILE)  # folder name
log_file_path = os.path.join(log_dir, LOG_FILE)
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format=logging_format,

    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler(sys.stdout)
    ]
)