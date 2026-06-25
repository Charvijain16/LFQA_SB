import json
from typing import Dict, List
from Bio import UniProt
import pandas as pd

from langchain.tools import StructuredTool
from src.tools.pydantic_inputs import SummUniProtPydanticInput

from src.utils import get_rel_info_by_llm
from src.tools.get_filtrered_references import filter_relvant_publications


def get_references(uniprot_info):
    extracted_references = []
    for reference in uniprot_info.get("references", []):
        id_name = next(
            (
                x.get("id", "")
                for x in reference["citation"].get(
                    "citationCrossReferences", [{"a": ""}]
                )
                if x.get("database") == "PubMed"
            ),
            "",
        )
        extracted_references.append(id_name)
    return extracted_references


def summarize_uniprot_entries(search_string, question):
    search_string = search_string.strip()
    if question == "" or question == None:
        raise Exception("Question is not given")
        # return Exception("Question is not present")
    
    list_of_ids = search_string.split(",")
    output = []
    for id_name in list_of_ids:
        fetched_page = UniProt.search(id_name)
        if len(fetched_page) > 0:
            uniprot_info = fetched_page[0]

            extracted_references = get_references(uniprot_info)
            extracted_references = list(
                filter(lambda num: num != "", extracted_references)
            )
            filtered_references = filter_relvant_publications(
                question=question, references=extracted_references
            )

            summary = get_rel_info_by_llm(question=question, json_input=fetched_page[0])
            output.append(
                {"id": id_name, "summary": summary, "references": filtered_references}
            )
        else:
            return (
                "The search query did not found any resutls or the identifier is not valid. Try another!",
                "",
            )

    return "", json.dumps(output)


summarize_uniprot_entries_tool = StructuredTool.from_function(
    func=summarize_uniprot_entries,
    name="Summarize_UniProt_Page",
    description="use this tool, to summarize information on the UniProt page for an identifier with respect to the question",
    args_schema=SummUniProtPydanticInput,
)

summarize_uniprot_entries_tool.description = """
    Use this tool, to summarize information on the UniProt page for an identifier with respect to the question.
    The tool takes the following arguments:
    - search_string: The search string is an identifier or couples of identifier separated by comma for UniProt Page(s) and each identifier is a mix of alphabet letters and numbers.
    - question: The original question to find the answer for.
    Example input:
    {{"search_string": "AQ34U1", "question": 'What is the function of the PDZ domain of the DLG1 protein in humans?'}}
"""

if __name__ == "__main__":

    a, b = summarize_uniprot_entries(
        search_string="Q93LQ6, Q96330, Q9ZWQ9, Q41452, Q07512, Q9XHG2, Q9M547, Q9FL28, Q0JA29, Q7XZQ6",
        question="What are synovial fibroblasts (FLS)?",
    )

    print(a, b)
