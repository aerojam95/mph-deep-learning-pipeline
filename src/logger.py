#=============================================================================
# Module: logging object for mph-deep-learning-pipeline
#=============================================================================

#=============================================================================
# Module imports
#=============================================================================

# Standard modules
import logging
import os

# Custom modules

#=============================================================================
# Variables
#=============================================================================

# logging variables
logging_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
log_file_path = "../outputs/logs/pipeline.log"
logger_name = "mph_pipeline_logger"

# Check log file exists
if not os.path.exists(log_file_path):
     raise FileNotFoundError(f"The log file {log_file_path} does not exist")

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format=f"{logging_format}",
                    handlers=[
                        logging.FileHandler(f"{log_file_path}"),
                        logging.StreamHandler()
                    ])

# Create logger
logger = logging.getLogger(f"{logger_name}")