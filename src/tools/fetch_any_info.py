import json
import pandas as pd
import http.client
import logging
import re
import time
from urllib.request import HTTPError
import requests
import xmltodict
from Bio import Entrez
from bs4 import BeautifulSoup
from http.client import http
from Bio import UniProt
from Bio.KEGG import REST
from langchain.tools import StructuredTool

from src.models import initialize_models
from src.tools.pydantic_inputs import FetchInfoAnySourcePydanticInput
from src.utils import get_rel_info_by_llm, xml_to_json
from src.tools.pubmed.summarize_pubmed_paper import fetch_pmcid_xmls


def remove_xref(text):
    # Define the regular expression pattern to match xref expressions
    xref_pattern = r"<xref\s.*?>.*?</xref>"

    # Use re.sub() to remove all matches of xref expressions from the text
    clean_text = re.sub(xref_pattern, "", text)

    soup = BeautifulSoup(clean_text, "html.parser")

    # Remove all HTML tags
    clean_text = soup.get_text(separator=" ")

    return clean_text


def get_text_from_paragraphs(paragraphs, text):
    if paragraphs is None:
        text += ""
    else:
        for i, element in enumerate(paragraphs):
            if element is None:
                continue
            if type(element) == str:
                text += element
                continue
            italic = element.get("italic")
            if italic:
                if isinstance(italic, list):
                    text += italic[0] if (italic[0]) == str else ""
                elif isinstance(italic, str):
                    text += italic
                else:
                    text += ""
            else:
                text += ""
            text += element["#text"] if element.get("#text") else ""
    return text


def get_text_from_sections(sections, text):
    if sections is None:
        text = ""
    else:
        for i, item in enumerate(sections):
            if item is None or type(item) == str:
                continue
            # title = item.get('title')
            paragraph = item.get("p")
            section = item.get("sec")
            # text += title + "\n"
            if paragraph is not None:
                text = get_text_from_paragraphs(paragraph, text)
            if section is not None:
                text = get_text_from_sections(section, text)
            else:
                continue
    return text


def get_full_text_from_pmc(include_reviews: bool, pmc_ids):
    df = pd.DataFrame()

    pmc_ids = [str(x.strip()) for x in pmc_ids]

    def fetch_papers(pmc_ids, retries=5, path=None):
        Entrez.email = "charvi.jain@tu-dresden.de"
        Entrez.api_key = "ef00f25c2ba58d2c308f3f17c7b9157efe08"
        Entrez.sleep_between_tries = 10
        Entrez.max_tries = 3
        try:
            handle = Entrez.efetch(db="pmc", retmode="xml", rettype="full", id=pmc_ids)
            xml_data = handle.read().decode("utf-8")
            # bug fix: remove journal-title tags from the xml data
            modified_data_string = re.sub(
                r"<journal-title>.*?</journal-title>\n?", "", xml_data
            )
            # saving here as it can become too large to read in memory
            # with open(path, 'w') as file:
            #     file.write(modified_data_string)
            xml_dict = xmltodict.parse(modified_data_string)
            return xml_dict
        except http.client.IncompleteRead as e:
            if retries > 0:
                logging.info("Retrying to fetch the data")
                fetch_papers(pmc_ids, retries - 1, path=path)
            else:
                logging.error("Failed to fetch the data")
                raise e
        except HTTPError as e:
            if retries > 0:
                logging.info("Retrying to fetch the data")
                fetch_papers(pmc_ids, retries - 1, path=path)
            else:
                logging.error("Failed to fetch the data")
                raise e

    data = []
    chunk_size = 10
    for chunk in range(0, len(pmc_ids), chunk_size):
        # path = os.path.join(os.path.dirname(logging.getLogger().handlers[0].baseFilename), f"fetcheddata_pmc_{chunk}.xml")
        temp = fetch_papers(pmc_ids[chunk : chunk + chunk_size])
        if temp is None:
            time.sleep(5)
            temp = fetch_papers(pmc_ids[chunk : chunk + chunk_size])
        if len(temp.get("pmc-articleset", {}).get("article", [])) > 1:
            data.extend(temp["pmc-articleset"]["article"])

    paper_texts = {}
    output = []
    for k, paper in enumerate(data):
        # article_ids = paper.get('front', {}).get('article-meta', {}).get('article-id', "")
        # pmc_key = (id["#text"] for id in article_ids if id["@pub-id-type"] == "pmc")
        if not isinstance(paper, dict):
            continue
        exclude_keywords = ["survey", "review"]
        if (
            not include_reviews
            and paper.get("@article-type").lower() in exclude_keywords
        ):
            continue
        else:
            doi_key = ""
            references = []
            summary = "No text available"
            filtered_references = (
                "References are not mentioned in the recieved information"
            )

            for element in paper["front"]["article-meta"]["article-id"]:
                if element.get("@pub-id-type") == "doi":
                    doi_key = element["#text"]
                if element.get("@pub-id-type") == "pmc":
                    pmc_key = element["#text"]
            text = ""
            if doi_key == "":
                continue
            else:
                if paper.get("body") == None:
                    paper_texts[doi_key] = ""
                else:
                    sections = paper["body"].get("sec")
                    paragraphs = paper["body"].get("p")
                    text = get_text_from_paragraphs(paragraphs=paragraphs, text=text)
                    text = get_text_from_sections(sections=sections, text=text)
                    paper_texts[doi_key] = remove_xref(text)

    return paper_texts


def get_kegg_page(item: str):  # item="map04390"
    """
    For a particular doc/item, keep only the information regarding genes, protein, pathways, publications and their interlinks.
    """
    data = REST.kegg_get(item).readlines()

    parsed_data = {
        "id": None,
        "name": None,
        "description": None,
        "references": [],
        "related_pathways": [],
    }

    # Parse the text
    current_reference = {}
    for line in data:
        line = line.strip()
        if line.startswith("ENTRY"):
            parsed_data["id"] = line.split()[1]
        elif line.startswith("NAME"):
            parsed_data["name"] = line.replace("NAME", "").strip()
        elif line.startswith("DESCRIPTION"):
            parsed_data["description"] = line.replace("DESCRIPTION", "").strip()
        elif line.startswith("REFERENCE"):
            if current_reference:  # Save the previous reference
                parsed_data["references"].append(current_reference)
            # Check if REFERENCE line has additional fields
            reference_split = line.split()
            pmid = reference_split[1] if len(reference_split) > 1 else None
            current_reference = {"PMID": pmid, "title": "", "author": ""}
        elif line.startswith("AUTHORS"):
            current_reference["author"] = (
                line.replace("AUTHORS", "").strip().split(",")[0] + ".et al"
            )
        elif line.startswith("TITLE"):
            current_reference["title"] = line.replace("TITLE", "").strip()
        elif line.startswith("///"):
            if current_reference:  # Append the last reference
                parsed_data["references"].append(current_reference)

    # Extract related pathways
    match = re.search(
        r"REL_PATHWAY\s+(.+?)\s+KO_PATHWAY",
        " ".join([item for item in data]),
        re.DOTALL,
    )
    if match:
        extracted_data = match.group(1)

        # Split the extracted data by lines and create a dictionary of {id: name}
        pathways_dict = {}
        for line in extracted_data.strip().split("\n"):
            # Strip leading and trailing spaces and split the line into id and name
            parts = line.strip().split(None, 1)  # Split only on the first space
            if len(parts) == 2:  # Ensure there are both id and name
                id, name = parts
                pathways_dict[id] = name

        parsed_data["related_pathways"] = pathways_dict

    return parsed_data


def get_gene_page(item: str):
    search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {
        "db": "gene",
        "id": item,
        "rettype":"xml"
        # "rettype":"vcv",
    }
    response = requests.get(search_url, params=params)
    if response.status_code == 400:
        return (
            "This is a bad request to the API, kindly check if the identifier is valid."
        )
    doc = xml_to_json(response.content)
    doc = json.dumps(doc)
    return doc


def get_clinvar_page(item: str):
    search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {
        "db": "clinvar",
        "id": item,
        "rettype": "vcv",
    }
    response = requests.get(search_url, params=params)
    if response.status_code == 400:
        return (
            "This is a bad request to the API, kindly check if the identifier is valid."
        )
    doc = xml_to_json(response.content)
    doc = json.dumps(doc)
    return doc



def chat_with_llama(full_text, question, thought):
    user_prompt = {
        "role": "user",
        "content": (
            "Analyze the following full text of a systems biology/computational biology research paper and answer the sub-question and in relevance to the question strictly based on the text content.\n\n"
            f"{full_text}\n\n"
            f"sub-question: {thought}\n\n"
            f"Question: {question}\n\n"
            "Important Instructions:\n"
            "- Use only the information explicitly present in the full text.\n"
            "- Do not rely on external knowledge or make any assumptions.\n"
            "- If it doesnt provide any relevant information, briefly in one line say how the paper deflects from the original question."
            "- Your response should be in plain text (not JSON), and should summarize all relevant parts from the full text that directly answer the question."
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


def fetch_info_any_source(question, thought, search_string):
    search_string = search_string.strip()
    items = search_string.split(",")
    if len(items) < 2:
        return "The syntax is incorrect, please provide both <database>,<identfier>", ""
    source = items[0]
    identifier = items[1]
    full_text = ""
    if "pubmed" in source.lower():
        full_text = fetch_pmcid_xmls(identifier) #get_full_text_from_pmc(include_reviews=True, pmc_ids=[identifier])
    elif "uniprot" in source.lower():
        fetched_page = UniProt.search(identifier)
        if len(fetched_page) == 0:
            return "The search query did not found any resutls. Try another!", ""
        full_text = fetched_page[0]
    elif "kegg" in source.lower():
        full_text = get_kegg_page(identifier)
    elif "genes" in source.lower():
        full_text = get_gene_page(identifier)
    elif "clinvar" in source.lower():
        full_text = get_clinvar_page(identifier)

    if isinstance(full_text, str) and "bad request" in full_text:
        return full_text, ""

    if not isinstance(full_text, str):
        full_text = str(full_text)

    if "pubmed" in source.lower():
        response = chat_with_llama(
        full_text=full_text, question=question, thought=thought
        )
    else:
        response = get_rel_info_by_llm(
            json_input=full_text, question=question, thought=thought
        )

    output = [{"id": identifier, "summary": response, "references": ""}]
    return "", json.dumps(output)

fetch_info_any_source_tool = StructuredTool.from_function(
    func=fetch_info_any_source,
    name="Fetch_Info_Any_Source",
    description="use this tool, to find a particular information based on the thought from any of the sources namely NCBI Genes, UniProt, KEGG, PubMedCentral, NCBI ClinVar.",
    args_schema=FetchInfoAnySourcePydanticInput,
)

fetch_info_any_source_tool.description = """
    Use this tool, to find a particular information based on the thought from any of the sources namely NCBI Genes, UniProt, KEGG, PubMedCentral, NCBI ClinVar.
    The tool takes the following arguments:
    - search_string: The search string is the name of source and an identifier separated by comma for a particular datasource that has to be queried.
    - question: The original question to find the answer for.
    - thought: The current thought.
    Example input:
    {{"search_string": "UniProt, AQ34U1", "question": 'What is the function of the PDZ domain of the DLG1 protein in humans?', "thought":"I would like to know the gene or list of gene related to the protein AQ34U1."}}
"""
