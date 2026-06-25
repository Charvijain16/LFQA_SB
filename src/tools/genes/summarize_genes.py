import copy
import json
import openai
import requests
import pandas as pd
from langchain.tools import StructuredTool
from src.tools.pydantic_inputs import SummGenesPydanticInput
from src.utils import get_rel_info_by_llm, xml_to_json
from src.tools.get_filtrered_references import filter_relvant_publications


def summarize_gene_page(search_string, question):
    search_string = search_string.strip()
    search_string = search_string.replace(" ", "")
    list_of_ids = search_string.split(",")
    search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {
        "db": "gene",
        "id": search_string,
        "retmode": "xml",
        # "rettype":"vcv",
    }
    response = requests.get(search_url, params=params)
    if response.status_code == 400:
        return (
            "This is a bad request to the API, kindly check if the identifier is valid.",
            "",
        )
    doc = xml_to_json(response.content)
    output = []
    sub_doc = doc.get("Entrezgene-Set", "").get("Entrezgene", "")

    if isinstance(sub_doc, str) and len(sub_doc) == 0:
        return "There is no information for this gene identifier or the identifier doesnt exist", ""

    if isinstance(sub_doc, dict):
        sub_doc = [sub_doc]

    for item in sub_doc:
        id_name = item["Entrezgene_track-info"]["Gene-track"]["Gene-track_geneid"]
        # json_string = json.dumps(item)

        reduced_item = copy.deepcopy(item)

        extracted_references = []
        if "Entrezgene_comments" in item and "Gene-commentary" in item.get("Entrezgene_comments", ""):
            for i, ref in enumerate(item["Entrezgene_comments"]["Gene-commentary"]):
                temp_reference = ref.get("Gene-commentary_refs", {}).get("Pub", {})
                if isinstance(temp_reference, dict) and len(temp_reference) > 0:
                    extracted_references.append(
                        temp_reference.get("Pub_pmid", {}).get("PubMedId")
                    )
                    reduced_item["Entrezgene_comments"]["Gene-commentary"][i].pop(
                        "Gene-commentary_refs"
                    )
                elif isinstance(temp_reference, list):
                    list_of_temp_refs = [
                        temp_ref["Pub_pmid"]["PubMedId"] for temp_ref in temp_reference
                    ]
                    extracted_references.extend(list_of_temp_refs)
                    reduced_item["Entrezgene_comments"]["Gene-commentary"][i].pop(
                        "Gene-commentary_refs"
                    )
        # remove None from the list
        extracted_references = list(
            filter(lambda num: num != None, extracted_references)
        )
        filtered_references = filter_relvant_publications(
            question=question, references=extracted_references
        )

        summary = get_rel_info_by_llm(json_input=reduced_item, question=question)

        output.append(
            {"id": id_name, "summary": summary, "references": filtered_references}
        )

    return "", json.dumps(output)


summarize_gene_page_tool = StructuredTool.from_function(
    func=summarize_gene_page,
    name="Summarize_Gene_Page",
    description="use this tool, to summarize information on the NCBI Genes page for an identifier with respect to the question",
    args_schema=SummGenesPydanticInput,
)

summarize_gene_page_tool.description = """
    Use this tool, to summarize information on the NCBI Genes page for an identifier with respect to the question.
    The tool takes the following arguments:
    - search_string: The search string is an identifier or couples of identifier separated by comma for NCBI Genes Page(s). These are usually numeric values.
    - question: The original question to find the answer for.
    Example input:
    {{"search_string": "23435", "question": 'What is the function of the PDZ domain of the DLG1 protein in humans?'}}
"""

if __name__ == "__main__":
    # a, b=summarize_gene_page(search_string="19017, 66838", question="Why does ETC flux in mouse oocytes remain constant despite changes in nutrient supply or energy demand?")
    # print(a, b)
    a, b = summarize_gene_page(
        search_string="3553",
        question="relation to Cdc42 inhibition and polar body extrusion",
    )
    print(a, b)
