from langchain_community.document_loaders import PyPDFLoader,TextLoader,CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.excel import UnstructuredExcelLoader
import hashlib
import os 
from typing import List, Dict, Any
from pathlib import Path
def all_pdf(pdf_path:str)-> list[Any]:
    docs = []
    data_path = Path(pdf_path)
    pdf_files = data_path.glob("/**/*.pdf")
    txt_files = data_path.glob("/**/*.txt")
    csv_files = data_path.glob("/**/*.csv")
    for pdf in pdf_files:
        print(f"Loading {pdf}")
        try:
            loader = PyPDFLoader(str(pdf))
            loaded = loader.load()
            print(f"[Debug] Loaded {len(loaded)} pages from {pdf}.")
            docs.extend(loaded)
        except Exception as e:
            print(f"[Error] Failed to load {pdf}: {e}")
    for txt in txt_files:
        print(f"Loading {txt}")
        try:
            loader = TextLoader(str(txt))
            loaded = loader.load()
            print(f"[Debug] Loaded {len(loaded)} pages from {txt}.")
            docs.extend(loaded)
        except Exception as e:
            print(f"[Error] Failed to load {txt}: {e}")
    for csv in csv_files:
        print(f"Loading {csv}")
        try:
            loader = CSVLoader(str(csv))
            loaded = loader.load()
            print(f"[Debug] Loaded {len(loaded)} pages from {csv}.")
            docs.extend(loaded)
        except Exception as e:
            print(f"[Error] Failed to load {csv}: {e}")
    excel_files = list(data_path.glob("**/*.xlsx"))
    print(f"[Debug] Found {len(excel_files)} Excel files.")
    for excel_file in excel_files:
        print(f"[Debug] Loading Excel file: {excel_file}")
        try:
            loader = UnstructuredExcelLoader(str(excel_file))
            loaded = loader.load()
            print(f"[Debug] Loaded {len(loaded)} pages from {excel_file}.")
            docs.extend(loaded)
        except Exception as e:
            print(f"[Error] Failed to load {excel_file}: {e}")
    return docs


