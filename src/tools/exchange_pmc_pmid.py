import json
import requests
import pandas as pd
from langchain.tools import StructuredTool
from src.tools.pydantic_inputs import ExchangePMCPMIDPydanticInput


def exchange_pmc_pmid(question, search_string):
    search_string = search_string.strip()
    # Define the base URL and parameters
    url = "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/"
    params = {
        "ids": search_string,
        "tool": "mytool",
        "email": "myemail@example.com",
        "format": "json",
    }

    # Make the request
    response = requests.get(url, params=params)
    if response.status_code == 400:
        return (
            "This is a bad request to the API, kindly check if the identifier is valid.",
            "",
        )

    output = []
    # Check if the request was successful
    if response.status_code == 200:
        data = response.json()  # or response.text for XML
        for items in data["records"]:
            if "versions" in items:
                items.pop("versions")
            output.append({"id": "", "summary": str(items), "references": ""})
    else:
        output.append({"id": "", "summary":"Kindly enter a valid identifier!", "references": ""})

    return "", json.dumps(output)


exchange_pmc_pmid_tool = StructuredTool.from_function(
    func=exchange_pmc_pmid,
    name="Exchange_PMC_PMID",
    description="use this tool, to find the PubMed identifier corresponding to PubMedCentral identifer or vice-a-versa.",
    args_schema=ExchangePMCPMIDPydanticInput,
)

exchange_pmc_pmid_tool.description = """
    Use this tool, to find the PubMed identifier corresponding to PubMedCentral identifer or vice-a-versa.
    The tool takes the following arguments:
    - search_string: The search string is an identifier or couples of identifier separated by comma, either numeric values or begininning with 'PMC'.
    - question: The original question to find the answer for.
    Example input:
    {{"search_string": "PMC6815265", "question": 'What is the function of the PDZ domain of the DLG1 protein in humans?'}}
"""


if __name__ == "__main__":
    # a, b=summarize_gene_page(search_string="19017, 66838", question="Why does ETC flux in mouse oocytes remain constant despite changes in nutrient supply or energy demand?")
    # print(a, b)

    a, b = exchange_pmc_pmid(
        search_string="PMC",
        question="relation to Cdc42 inhibition and polar body extrusion",
    )
    print(a, b)
