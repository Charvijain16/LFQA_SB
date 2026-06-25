import json
import openai
import requests
import pandas as pd
from langchain.tools import StructuredTool
from src.tools.pydantic_inputs import SummMutationPydanticInput
from src.utils import get_rel_info_by_llm, xml_to_json
from src.tools.get_filtrered_references import filter_relvant_publications


def summarize_clinvar_page(search_string, question):
    search_string = search_string.strip()
    search_string = search_string.replace(" ", "")
    list_of_ids = search_string.split(",")
    search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {
        "db": "clinvar",
        "id": search_string,
        "rettype": "vcv",
    }
    response = requests.get(search_url, params=params)
    if response.status_code == 400:
        return (
            "This is a bad request to the API, kindly check if the identifier is valid.",
            "",
        )
    doc = xml_to_json(response.content)
    output = []
    sub_doc = doc["ClinVarResult-Set"]["VariationArchive"]
    if isinstance(sub_doc, dict):
        sub_doc = [sub_doc]
    for item in sub_doc:
        id_name = item["@Accession"]
        json_string = json.dumps(item)

        summary = get_rel_info_by_llm(json_input=json_string, question=question)

        pubmed_ids = [
            obj["ID"]["#text"]
            for obj in item.get("ClassifiedRecord", {})
                .get("Classifications", {})
                .get("GermlineClassification", {})
                .get("Citation", [])
            if isinstance(obj, dict) 
            and "ID" in obj 
            and isinstance(obj["ID"], dict) 
            and obj["ID"].get("@Source") == "PubMed"
        ]

        extracted_references = pubmed_ids

        filtered_references = filter_relvant_publications(
            question=question, references=extracted_references
        )

        output.append(
            {"id": id_name, "summary": summary, "references": filtered_references}
        )

    return "", json.dumps(output)


summarize_clinvar_page_tool = StructuredTool.from_function(
    func=summarize_clinvar_page,
    name="Summarize_CliVar_Page",
    description="use this tool, to summarize information on the ClinVar page for an identifier with respect to the question",
    args_schema=SummMutationPydanticInput,
)

summarize_clinvar_page_tool.description = """
    Use this tool, to summarize information on the ClinVar page for an identifier with respect to the question.
    The tool takes the following arguments:
    - search_string: The search string is an identifier or couples of identifier separated by comma for NCBI ClinVar Page(s) beginning with 'VCV' and followed by a numeric value.
    - question: The original question to find the answer for.
    Example input:
    {{"search_string": "VCV000012783", "question": 'What is the function of the PDZ domain of the DLG1 protein in humans?'}}
"""

if __name__ == "__main__":
    a, b = summarize_clinvar_page(
        search_string="VCV000156444.33,VCV000375891.48",
        question="What is the role of Isocitrate dehydrogenase (IDH) mutations in glioma?",
    )
    print(a, b)
