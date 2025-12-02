# import os
# import logging
# from PyPDF2 import PdfReader

# logger = logging.getLogger(__name__)


# def extract_text_from_folder(folder_path: str):
#     extracted_data = {}
    
#     try:
#         if not os.path.exists(folder_path):
#             logger.error(f"Folder not found: {folder_path}")
#             return {}

#         logger.info(f"Scanning folder for PDF files: {folder_path}")

#         for file in os.listdir(folder_path):
#             if file.lower().endswith(".pdf"):
#                 file_path = os.path.join(folder_path, file)
#                 logger.info(f"Processing PDF: {file_path}")

#                 try:
#                     reader = PdfReader(file_path)
#                     text_content = []
                    
#                     for i, page in enumerate(reader.pages):
#                         try:
#                             text = page.extract_text() or ""
#                             text_content.append(text)
#                         except Exception as e:
#                             logger.warning(f"Failed to extract text from page {i} in {file}: {e}")

#                     full_text = "\n".join(text_content)
#                     extracted_data[file] = full_text
#                     logger.info(f"Successfully extracted text from: {file}")

#                 except Exception as e:
#                     logger.error(f"Error reading PDF '{file}': {e}")

#         return extracted_data

#     except Exception as e:
#         logger.error(f"Unhandled error during PDF extraction from folder {folder_path}: {e}")
#         raise


# def get_all_paper_folders(base_directory: str = "papers"):
#     try:
#         if not os.path.exists(base_directory):
#             logger.error(f"Base directory not found: {base_directory}")
#             return []

#         folders = []
#         for item in os.listdir(base_directory):
#             item_path = os.path.join(base_directory, item)
#             if os.path.isdir(item_path) and item.isdigit():
#                 folders.append(item)
        
#         # Sort folders numerically
#         folders.sort(key=lambda x: int(x))
#         logger.info(f"Found {len(folders)} paper folders: {folders}")
#         return folders

#     except Exception as e:
#         logger.error(f"Error scanning paper folders: {e}")
#         raise

import os
import logging
from PyPDF2 import PdfReader

logger = logging.getLogger(__name__)

def extract_text_from_folder(folder_path: str):
    extracted_data = {}

    if not os.path.exists(folder_path):
        logger.error(f"Folder not found: {folder_path}")
        return {}

    logger.info(f"Scanning folder for PDF files: {folder_path}")

    for file in os.listdir(folder_path):
        if file.lower().endswith(".pdf"):
            file_path = os.path.join(folder_path, file)
            logger.info(f"Processing PDF: {file_path}")
            try:
                reader = PdfReader(file_path)
                text_content = []

                for page in reader.pages:
                    try:
                        text = page.extract_text() or ""
                        # Remove extra whitespace and normalize
                        clean_text = " ".join(text.split())
                        text_content.append(clean_text)
                    except Exception as e:
                        logger.warning(f"Failed to extract text from page in {file}: {e}")

                full_text = "\n".join(text_content)
                extracted_data[file] = full_text
                logger.info(f"Extracted and cleaned text from: {file}")

            except Exception as e:
                logger.error(f"Error reading PDF '{file}': {e}")

    return extracted_data

def get_all_paper_folders(base_directory: str = "papers"):
    if not os.path.exists(base_directory):
        logger.error(f"Base directory not found: {base_directory}")
        return []

    folders = [item for item in os.listdir(base_directory)
               if os.path.isdir(os.path.join(base_directory, item)) and item.isdigit()]
    folders.sort(key=lambda x: int(x))
    logger.info(f"Found {len(folders)} paper folders: {folders}")
    return folders
