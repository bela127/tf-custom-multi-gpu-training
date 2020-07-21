import datetime

def auto_name_log_folder(log_folder, query_for_name = True):
    if not log_folder:
        current_time = datetime.datetime.now()
        date_time = current_time.strftime("%d_%m_%Y-%H_%M_%S")
        if query_for_name:
            log_folder = input(f"input a log folder name or nothing to use {date_time}:")           
        if not log_folder:
            log_folder = date_time
    return log_folder