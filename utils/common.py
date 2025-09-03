def get_position_as_tuple(start_pos):
    return (int(start_pos[0]), int(start_pos[1]))

import logging
import os

def setup_logging(log_dir: str, log_filename: str = "evaluation_log.txt"):
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, log_filename)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_file, mode='w'), logging.StreamHandler()]
    )
