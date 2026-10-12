import os
import requests

BASE_URL = "http://localhost:8000/api"

def test_health():
    print("--- Testing Documents Endpoint ---")
    res = requests.get(f"{BASE_URL}/documents")
    print("Status Code:", res.status_code)
    print("Response:", res.json())
    print()

def test_upload(pdf_path):
    print("--- Testing PDF Upload ---")
    filename = os.path.basename(pdf_path)
    with open(pdf_path, "rb") as f:
        files = {"file": (filename, f, "application/pdf")}
        res = requests.post(f"{BASE_URL}/upload", files=files)
    print("Status Code:", res.status_code)
    print("Response:", res.json())
    print()
    return filename

def test_chat(query, document_name=None):
    print("--- Testing Chat Endpoint ---")
    payload = {
        "message": query,
        "history": [],
        "documents": [document_name] if document_name else []
    }
    res = requests.post(f"{BASE_URL}/chat", json=payload)
    print("Status Code:", res.status_code)
    print("Response:", res.json())

if __name__ == "__main__":
    test_health()

    pdf_path = "C:/Users/Debajyoti/Downloads/SANKETNonCoal_17042025.pdf"
    uploaded_doc_name = test_upload(pdf_path)

    query = (
        "Extract the non-coal mine accident statistics for Rajasthan from Statement 7.1 "
        "and display them in a table with columns: Mineral, District, Fatal Accidents, "
        "Serious Accidents, Persons Killed, and Persons Injured."
    )
    test_chat(query, document_name=uploaded_doc_name)