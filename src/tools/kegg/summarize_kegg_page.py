import os
import re
from typing import Dict, List
from urllib.error import HTTPError
import openai
import pandas as pd
from Bio.KEGG import REST
from langchain_core.prompts import ChatPromptTemplate
from http.client import http
from Bio import Entrez
from sentence_transformers import SentenceTransformer
from torch.nn.functional import cosine_similarity
from langchain.tools import StructuredTool
from src.tools.pydantic_inputs import SummKeggPydanticInput

from src.utils import get_rel_info_by_llm
from src.tools.get_filtrered_references import filter_relvant_publications
import json


def clean_doc(item: str, retries: int):  # item="map04390"
    """
    For a particular doc/item, keep only the information regarding genes, protein, pathways, publications and their interlinks.
    """
    item = item.strip()
    try:
        data = REST.kegg_get(item).readlines()
    except HTTPError:
        return (
            "This identifier is invalid. Please check or correct the identifier again."
        )

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
            pmid = (
                reference_split[1].split(":")[-1] if len(reference_split) > 1 else None
            )
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


def summarize_kegg_pathways(search_string, question):
    search_string = search_string.strip()
    list_of_pathways = search_string.split(",")
    all_docs_info = []
    for item in list_of_pathways:
        parsed_data = clean_doc(item, 3)
        if isinstance(parsed_data, str):
            return parsed_data, ""
        else:
            all_docs_info.append(parsed_data)

    df_filtered = pd.DataFrame(all_docs_info)

    output = []
    for id, row in df_filtered.iterrows():
        references = row["references"]
        row.pop("references")
        summary = get_rel_info_by_llm(question=question, json_input=row.to_dict())
        pmid_values = list(map(lambda x: x["PMID"], references))
        extracted_references = list(filter(lambda num: num != None, pmid_values))

        filtered_references = filter_relvant_publications(
            question=question, references=extracted_references
        )

        output.append(
            {"id": row["id"], "summary": summary, "references": filtered_references}
        )

    # df_filtered['references'] = df_filtered['references'].apply(lambda x: str(x))
    # df_filtered['related_pathways'] = df_filtered['related_pathways'].apply(lambda x: str(x))

    return "", json.dumps(output)


summarize_kegg_pathways_tool = StructuredTool.from_function(
    func=summarize_kegg_pathways,
    name="Summarize_KEGG_Page",
    description="use this tool, to summarize information on the KEGG Pathway page for an identifier with respect to the question",
    args_schema=SummKeggPydanticInput,
)

summarize_kegg_pathways_tool.description = """
    Use this tool, to summarize information on the KEGG Pathway page for an identifier with respect to the question.
    The tool takes the following arguments:
    - search_string: The search string is an identifier or couples of identifier separated by comma for KEGG Pathways Page(s) beginning with 'map' and followed by a numeric value.
    - question: The original question to find the answer for.
    Example input:
    {{"search_string": "map4390", "question": 'What is the function of the PDZ domain of the DLG1 protein in humans?'}}
"""

if __name__ == "__main__":
    a, b = summarize_kegg_pathways(
        search_string="map04064, map04380, map04928, map05323",
        question="What are key genes in osteoclast bone resorption?",
    )
    print(a, b)
