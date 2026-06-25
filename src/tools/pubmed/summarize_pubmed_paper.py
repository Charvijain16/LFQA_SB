import json
import os
import re
import time
from bs4 import BeautifulSoup
from langchain_openai import ChatOpenAI
import pandas as pd
import requests
from langchain.tools import StructuredTool

from src.models import initialize_models
from src.tools.pydantic_inputs import SummPubmedPydanticInput


def fetch_pmcid_xmls(pmcid, max_retry=5, timeout=10):
    retry = 0
    while retry < max_retry:
        try:
            url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pmc&id={pmcid}"
            response = requests.get(url, timeout=timeout)
            if response.status_code == 200:
                xml_content = response.content
                raw_text = extract_text_from_xml(xml_content)
                raw_text = strip_references(raw_text)
                if not raw_text.strip():
                    return None, f"No text extracted for PMCID {pmcid}"
                return raw_text  # ✅ Exit on success
            else:
                print(f"Failed to fetch PMCID {pmcid}: HTTP {response.status_code}")
                retry += 1
                time.sleep(2)
        except requests.exceptions.RequestException as e:
            print(f"Error fetching PMCID {pmcid}: {e}")
            retry += 1
            time.sleep(2)
    
    # If all retries fail
    print(f"Failed to fetch PMCID {pmcid} after {max_retry} retries.")
    return None, f"The file for PMCID {pmcid} couldn't be downloaded from the API"


# def extract_text_from_pdf(pdf_path):
#     try:
#         from pdf2image.exceptions import PDFInfoNotInstalledError

#         # Use your exact Poppler path
#         poppler_path = "/opt/homebrew/opt/poppler/bin"

#         # Convert PDF to images
#         pages = convert_from_path(
#             pdf_path,
#             dpi=300,
#             poppler_path=poppler_path
#         )
        
#         # OCR each page
#         text = ""
#         for page in pages:
#             text += pytesseract.image_to_string(page) + "\n"
#         return text.strip()

#     except PDFInfoNotInstalledError:
#         print("ERROR: Poppler not found or 'pdfinfo.exe' missing.")
#         return ""
#     except Exception as e:
#         print(f"ERROR reading {pdf_path}: {e}")
#         return ""

def extract_text_from_xml(xml_content):
    """Extract text from XML file (PMC format)"""
    try:
        # Parse with BeautifulSoup for better handling of malformed XML
        soup = BeautifulSoup(xml_content, 'xml')
        
        # Extract text from main content sections
        text_sections = []
        
        # Try to find abstract
        abstract = soup.find('abstract')
        if abstract:
            text_sections.append(abstract.get_text(strip=True))
        
        # Try to find body text
        body = soup.find('body')
        if body:
            text_sections.append(body.get_text(strip=True))
        
        # If no body, try to find article content
        if not body:
            article = soup.find('article')
            if article:
                text_sections.append(article.get_text(strip=True))
        
        # If still no content, extract all text
        if not text_sections:
            text_sections.append(soup.get_text(strip=True))
        
        return '\n\n'.join(text_sections)
    
    except Exception as e:
        print(f"ERROR reading XML: {e}")
        return ""


def strip_references(text):
    match = re.split(r'\bReferences\b|\bREFERENCES\b|\bBibliography\b', text, maxsplit=1)
    return match[0].strip() if match else text.strip()


def chat_with_llama(full_text, question):
    user_prompt = {
        "role": "user",
        "content": (
            "Analyze the following full text of a systems biology/computational biology research paper and answer the question strictly based on the text content.\n\n"
            f"{full_text}\n\n"
            f"Question: {question}\n\n"
            "Important Instructions:\n"
            "- Use only the information explicitly present in the full text.\n"
            "- Do not rely on external knowledge or make any assumptions.\n"
            "- If it doesnt provide any relevant information, briefly in one line say how the paper deflects from the original question."
            "- Your response should be in plain text (not JSON), concise and should summarize all relevant parts from the full text that directly answer the question."
        )
    }
    system_prompt = {
        "role": "system",
        "content": "You are an AI assistant specializing in systems biology and computational biology research paper analysis. Extract only the required information."
    }
    content = ""
    for attempt in range(3):
        try:
            llm = initialize_models.LLM
            messages = [
                system_prompt,
                user_prompt,
            ]

            response=llm.invoke(messages)
            content = response.content
            if content:
                break
        except Exception as e:
            print(f"Model's attempt {attempt+1} failed: {e}")
            time.sleep(1)
    return content



def summarize_pubmed_papers(search_string, question):
    search_string = search_string.strip()
    list_of_ids = search_string.split(",")
    
    df = pd.DataFrame()
    
    pmc_ids = [str(x.strip()) for x in list_of_ids]
    output = []
    for item in pmc_ids:
        raw_text = fetch_pmcid_xmls(item)
        llm_data = chat_with_llama(raw_text, question)
        output.append(
                {"id": item, "summary": llm_data, "references": ""}
            )
    
    return json.dumps(df.to_dict(orient="records")), json.dumps(output)


summarize_pubmed_papers_tool = StructuredTool.from_function(
    func=summarize_pubmed_papers,
    name="Summarize_PubMedCentral_Page",
    description="use this tool, to summarize information on the PubMedCentral paper with respect to the question",
    args_schema=SummPubmedPydanticInput,
)

summarize_pubmed_papers_tool.description = """
    Use this tool, to summarize information on the PubMedCentral paper with respect to the question.
    The tool takes the following arguments:
    - search_string: The search string is an identifier or couples of identifier separated by comma for PubMedCentral Page(s) beginning with 'PMC'.
    - question: The original question to find the answer for.
    Example input:
    {{"search_string": "PMC6815265", "question": 'What is the function of the PDZ domain of the DLG1 protein in humans?'}}
"""

# if __name__ == "__main__":
#     # _, output = summarize_pubmed_papers("PMC10844633", "What is the function of eosinophil peroxidase (EPX) in regulating osteoclast activity in mice?")
#     _, output = summarize_pubmed_papers("PMC10627182", "What is the role of Myosin-X in spindle morphogenesis in the mouse oocyte?")
#     print(output)