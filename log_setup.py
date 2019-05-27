import datetime
import logging
import os

# logging setup:
def init_logging(max_log_files=10, logging_level="DEBUG"):
    def log_switch(logging_level):
        switcher = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL,
        }
        return switcher.get(logging_level, "Invalid Logging Level")

    LOG_DIR = "./logs/"
    # get current datetime as string for log filename:
    log_datetime_str = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    # setup logging to file in ./logs/<date-time>.log:
    logging.basicConfig(
        format="%(levelname)s:%(message)s",
        filename=LOG_DIR + log_datetime_str + ".log",
        level=log_switch(logging_level),
    )
    log_list = os.listdir(LOG_DIR)
    log_list_paths = [os.path.abspath(LOG_DIR) + "/{0}".format(x) for x in log_list]
    num_log_files = len(log_list)
    # prune oldest files if number of log files is >max_log_files
    while num_log_files > max_log_files:
        oldest_log_file = min(log_list_paths, key=os.path.getctime)
        os.remove(os.path.abspath(oldest_log_file))

        log_list = os.listdir(LOG_DIR)
        log_list_paths = [os.path.abspath(LOG_DIR) + "/{0}".format(x) for x in log_list]
        num_log_files = len(log_list)
    return