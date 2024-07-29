#=============================================================================
# Module: logging object for mph-deep-learning-pipeline
#=============================================================================

#=============================================================================
# Module imports
#=============================================================================

# Standard modules
import logging

# Custom modules

#=============================================================================
# Variables
#=============================================================================

# logging variables
logging_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
log_file_path = "../outputs/logs/pipeline.log"
logger_name = "mph_pipeline_logger"

# Configure logging
logging.basicConfig(level=logging.DEBUG,
                    format=f"{logging_format}",
                    handlers=[
                        logging.FileHandler(f"{log_file_path}"),
                        logging.StreamHandler()
                    ])

# Create logger
logger = logging.getLogger(f"{logger_name}")