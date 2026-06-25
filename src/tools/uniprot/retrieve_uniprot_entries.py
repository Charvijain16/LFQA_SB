from Bio import UniProt
from sentence_transformers import SentenceTransformer
import json
from torch.nn.functional import cosine_similarity
import pandas as pd
from langchain.tools import StructuredTool
from src.tools.pydantic_inputs import RetUniProtPydanticInput
from src.utils import safe_get
import http.client
import time

def get_references(uniprot_info):
    extracted_references = []
    for reference in uniprot_info["references"]:
        extracted_references.append(
            {
                "title": reference["citation"].get("title", ""),
                "pubmed_id": "PMID"
                + next(
                    (
                        x.get("id", "")
                        for x in reference["citation"].get(
                            "citationCrossReferences", [{"a": ""}]
                        )
                        if x.get("database") == "PubMed"
                    ),
                    "",
                ),
                "year": reference["citation"]["publicationDate"],
                "authors": reference["citation"].get("authors", [""])[0],
            }
        )
    return extracted_references


def get_function_value(uniprot_info):
    function_value = " ".join(
        text["value"]
        for entry in uniprot_info.get("comments", {})
        if entry.get("commentType") == "FUNCTION"
        for text in entry["texts"]
    )
    return function_value


def fetch_docs_from_uniprot(search_string, max_retries=3, delay=2):
    for attempt in range(max_retries):
        try:
            return UniProt.search(search_string)
        except http.client.IncompleteRead as e:
            print(f"IncompleteRead on attempt {attempt + 1}, retrying...")
            time.sleep(delay)
        except Exception as e:
            print(f"Unexpected error: {e}")
            raise e
    raise RuntimeError("Failed to fetch from UniProt API after retries.")

def fetch_docs_from_API(search_string: str):
    """
    For the searched keywords, fetch the list of docs/items found via the API
    """
    results = fetch_docs_from_uniprot(search_string)
    if len(results) == 0:
        return "No entries found. Maybe this search is too specific. Try again with another search or broader search!"
    uniprot_fetched_results = []
    if len(results) > 5000:
        return (
            "Too many results. Try with a specific search query with important keywords"
        )
    for item in results:
        uniprot_info = item

        uniprot_fetched_results.append(
            {
                "id": uniprot_info["primaryAccession"],
                "name": uniprot_info["uniProtkbId"],
                "organism": uniprot_info["organism"]["scientificName"],
                "description": safe_get(
                    uniprot_info,
                    ["proteinDescription", "recommendedName", "fullName", "value"],
                    default="",
                ),
                # "genes": [{
                #     "name": "NA",
                #     "id": "empty"
                # }], # shall I also store the synonyms
                "function": get_function_value(
                    uniprot_info
                ),  ## only get the one with function
                "references": get_references(uniprot_info),
            }
        )

    return uniprot_fetched_results


def retrieve_rel_docs(question, uniprot_fetched_results):
    """
    From the list of docs/items recieved from the API, filter down the relevant items based on organism type, description, etc.
    """
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    protein_ids = [d["id"] for d in uniprot_fetched_results if "id" in d]
    descriptions = [d["function"] for d in uniprot_fetched_results if "function" in d]

    question_embedding = model.encode(question, convert_to_tensor=True)
    description_embeddings = model.encode(descriptions, convert_to_tensor=True)

    # Compute similarities
    similarities = [
        cosine_similarity(question_embedding, desc_embedding.unsqueeze(0)).item()
        for desc_embedding in description_embeddings
    ]

    results = list(zip(protein_ids, descriptions, similarities))

    # Sort by similarity (highest first)
    results = sorted(results, key=lambda x: x[2], reverse=True)

    # LLM summarizer for top 3 results   --- > ask jens?
    top_k = results[:2]
    return top_k


def retrieve_uniprot_entries(search_string, question):
    search_string = search_string.strip()
    if question is None or question == "":
        print("Error: The question is None or empty")
        # raise Exception("Error: The question is None or empty")
    

    uniprot_fetched_results = fetch_docs_from_API(search_string=search_string)
    if isinstance(uniprot_fetched_results, str):
        return uniprot_fetched_results, ""

    # if len(uniprot_fetched_results) > 5:
    #     top_k_results = retrieve_rel_docs(question=question, uniprot_fetched_results=uniprot_fetched_results)
    #     df_filtered = pd.DataFrame()
    #     for id, function, score in top_k_results:
    #         row_to_add = get_dict_by_key_value(uniprot_fetched_results, "id", id)
    #         df_filtered = pd.concat([df_filtered, pd.DataFrame([row_to_add])], ignore_index=True)
    # else:
    df_filtered = pd.DataFrame(uniprot_fetched_results)
    if len(df_filtered) == 0:
        return "No entries found. Try again with a smarter and concise keyword search"

    df_filtered = df_filtered.drop(columns=["references"], axis=1)

    return json.dumps(df_filtered[:10].to_dict(orient="records")), ""


retrieve_uniprot_entries_tool = StructuredTool.from_function(
    func=retrieve_uniprot_entries,
    name="Retrieve_UniProt_Page",
    description="use this tool, to retrieve proteins from UniProt based on the search string",
    args_schema=RetUniProtPydanticInput,
)

retrieve_uniprot_entries_tool.description = """
    Use this tool, to retrieve proteins from UniProt based on the search string.
    The tool takes the following arguments:
    - search_string: The search string are keyowrds for search on the UniProt data source. This can be sometimes be separated by AND, OR operators.
    - question: The original question to find the answer for.
    Example input:
    {{"search_string": "DLG1 AND Humans", "question": 'What is the function of the PDZ domain of the DLG1 protein in humans?'}}
"""
