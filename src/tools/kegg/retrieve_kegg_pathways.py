import json
import requests
from bs4 import BeautifulSoup
from Bio.KEGG import REST
import re
from sentence_transformers import SentenceTransformer
from torch.nn.functional import cosine_similarity
import pandas as pd
from langchain.tools import StructuredTool
from src.tools.pydantic_inputs import RetKeggPydanticInput
from src.tools.kegg.summarize_kegg_page import clean_doc
from urllib.parse import quote_plus
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


def fetch_docs_from_API(search_string: str):
    """
    For the searched keywords, fetch the list of docs/items found via the API
    """
    search_string = quote_plus(search_string)
    url = f"https://www.kegg.jp/kegg-bin/search_pathway_text?map=map&keyword={search_string}&mode=1&viewImage=false"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    }

    # Create a session with retries
    session = requests.Session()
    retries = Retry(connect=5, backoff_factor=0.5)
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    try:
        response = session.get(url, headers=headers, timeout=10)
        response.raise_for_status()  # Raises error for HTTP errors (e.g. 404)
        soup = BeautifulSoup(response.text, "html.parser")
        rows = soup.find_all("tr")
        # Continue your processing here...
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        rows = []

    entries = {}
    if len(rows) > 0:
        for row in rows[1:]:  # Skip the header row
            entry_cell = row.find("td")
            if entry_cell:
                a = entry_cell.find("a")
                if a == None:
                    entries = "No entries found. Maybe this search is too specific. Try again with another search or broader search!"
                    break
                entry_id = entry_cell.find("a").text.strip()  # Get the entry ID
                name_cell = row.find_all("td")[1]  # Get the Name cell
                name = name_cell.text.strip().replace("\n", " ")  # Clean up text
                entries[entry_id] = name

    return entries


def retrieve_rel_docs(question, all_docs_info):
    """
    From the list of docs/items recieved from the API, filter down the relevant items based on organism type, description, etc.
    """
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    pathway_ids = [d["id"] for d in all_docs_info if "id" in d]
    descriptions = [d["description"] for d in all_docs_info if "description" in d]
    # Generate embeddings for the question
    question_embedding = model.encode(question, convert_to_tensor=True)

    # Generate embeddings for the descriptions
    description_embeddings = model.encode(descriptions, convert_to_tensor=True)

    # Compute similarities
    similarities = [
        cosine_similarity(question_embedding, desc_embedding.unsqueeze(0)).item()
        for desc_embedding in description_embeddings
    ]

    # Pair descriptions with their similarities
    results = list(zip(pathway_ids, descriptions, similarities))

    # Sort by similarity (highest first)
    results = sorted(results, key=lambda x: x[2], reverse=True)

    top_k = results[:2]
    return top_k


def retrieve_kegg_pathways(search_string, question):
    search_string = search_string.strip()
    if question is None or question == "":
        print("Error: The question is None or empty")

    entries = fetch_docs_from_API(search_string=search_string)
    if isinstance(entries, str):
        return entries, ""
    all_docs_info = []
    for item in entries.keys():
        all_docs_info.append(clean_doc(item, 3))

    df_filtered = pd.DataFrame()

    # if len(all_docs_info) > 5:
    #     top_k_results = retrieve_rel_docs(question=question, all_docs_info=all_docs_info)
    #     for id, function, score in top_k_results:
    #         row_to_add = get_dict_by_key_value(all_docs_info, "id", id)
    #         df_filtered = pd.concat([df_filtered, pd.DataFrame([row_to_add])], ignore_index=True)
    # else:
    df_filtered = pd.DataFrame(all_docs_info)

    df_filtered = df_filtered.drop(columns=["references"], axis=1)

    # df_filtered['references'] = df_filtered['references'].apply(lambda x: str(x))
    df_filtered["related_pathways"] = df_filtered["related_pathways"].apply(
        lambda x: str(x)
    )

    return (
        json.dumps(
            df_filtered[:10].to_dict(orient="records"), ensure_ascii=False
        ).replace("'", ""),
        "",
    )


retrieve_kegg_pathways_tool = StructuredTool.from_function(
    func=retrieve_kegg_pathways,
    name="Retrieve_KEGG",
    description="use this tool, to retrieve pathways from KEGG Pathways based on the search string",
    args_schema=RetKeggPydanticInput,
)

retrieve_kegg_pathways_tool.description = """
    Use this tool, to retrieve pathways from KEGG Pathways based on the search string.
    The tool takes the following arguments:
    - search_string: The search string are keyowrds for search on the KEGG Pathways data source. This can be sometimes be separated by AND, OR operators.
    - question: The original question to find the answer for.
    Example input:
    {{"search_string": "DLG1 AND Humans", "question": 'What is the function of the PDZ domain of the DLG1 protein in humans?'}}
"""

if __name__ == "__main__":
    a, b = retrieve_kegg_pathways(
        search_string="Oxidative Phosphorylation AND NADH",
        question="Why is NADH redox cycle essential for mitochondrial energy production?",
    )
    print(a, b)
