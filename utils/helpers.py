import os

def get_project_root():
    """ Lấy đường dẫn thư mục gốc """
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
