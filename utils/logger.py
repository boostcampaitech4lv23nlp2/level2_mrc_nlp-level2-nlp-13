import logging
import os


def get_logger(file_path: str):
    if not os.path.exists(file_path):
        if not os.path.exists("./logs"):
            os.makedirs("./logs")
        with open(file_path, "w+") as f:
            f.write("***** Logging *****\n")

    logger = logging.getLogger(__name__)

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(file_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
