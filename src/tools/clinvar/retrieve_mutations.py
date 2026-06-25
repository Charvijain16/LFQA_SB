import json
import requests
import pandas as pd
from langchain.tools import StructuredTool
from src.tools.pydantic_inputs import RetMutationPydanticInput
from src.utils import xml_to_json


def clean_doc(item: dict):
    temp = {}
    temp["VariationID"] = (
        item.get("ClinVarResult-Set", {})
        .get("VariationArchive", {})
        .get("@VariationID", "")
    )
    temp["VariationName"] = (
        item.get("ClinVarResult-Set", {})
        .get("VariationArchive", {})
        .get("@VariationName", "")
    )
    temp["VariationType"] = (
        item.get("ClinVarResult-Set", {})
        .get("VariationArchive", {})
        .get("@VariationType", "")
    )
    temp["AccessionID"] = (
        item.get("ClinVarResult-Set", {})
        .get("VariationArchive", {})
        .get("@Accession", "")
        + "."
        + item.get("ClinVarResult-Set", {})
        .get("VariationArchive", {})
        .get("@Version", "")
    )
    temp["RecordType"] = (
        item.get("ClinVarResult-Set", {})
        .get("VariationArchive", {})
        .get("@RecordType", "")
    )
    # temp["NumberOfSubmissions"] = item.get('ClinVarResult-Set',{}).get('VariationArchive', {}).get("@NumberOfSubmissions", "")
    temp["Species"] = (
        item.get("ClinVarResult-Set", {})
        .get("VariationArchive", {})
        .get("@Species", "")
    )

    genelist = (
        item.get("ClinVarResult-Set", {})
        .get("VariationArchive", {})
        .get("ClassifiedRecord", {})
        .get("SimpleAllele", {})
        .get("GeneList", {})
    )

    if isinstance(genelist.get("Gene"), list):
        total_genes_count = len(genelist.get("Gene", []))
        genes = genelist.get("Gene", [])
        if total_genes_count < 8 and total_genes_count > 1:
            temp["GeneID"] = []
            for gene in genes:
                temp["GeneID"].append(gene.get("@GeneID", ""))
        else:
            temp["GeneID"] = (
                f"Total of {total_genes_count} found. Can't be displayed here"
            )
    elif isinstance(genelist.get("Gene"), dict):
        temp["GeneID"] = genelist.get("Gene", {}).get("@GeneID", "")
    else:
        temp["GeneID"] = ""

    # temp["RelationshipTypeGene"] = item.get('ClinVarResult-Set',{}).get('VariationArchive', {}).get('ClassifiedRecord',{}).get('SimpleAllele',{}).get('GeneList',{}).get('Gene',{}).get('@@RelationshipType',"")

    temp["ProteinChange"] = (
        item.get("ClinVarResult-Set", {})
        .get("VariationArchive", {})
        .get("ClassifiedRecord", {})
        .get("SimpleAllele", {})
        .get("ProteinChange", "")
    )

    temp["OtherNames"] = (
        item.get("ClinVarResult-Set", {})
        .get("VariationArchive", {})
        .get("ClassifiedRecord", {})
        .get("SimpleAllele", {})
        .get("OtherNameList", {})
        .get("Name", "")
    )

    temp["ReviewStatus"] = (
        item.get("ClinVarResult-Set", {})
        .get("VariationArchive", {})
        .get("ClassifiedRecord", {})
        .get("Classifications", {})
        .get("GermlineClassification", {})
        .get("ReviewStatus", "")
    )

    temp["Description"] = (
        item.get("ClinVarResult-Set", {})
        .get("VariationArchive", {})
        .get("ClassifiedRecord", {})
        .get("Classifications", {})
        .get("GermlineClassification", {})
        .get("Description", "")
    )

    temp["Diseases"] = []

    diseases = (
        item.get("ClinVarResult-Set", {})
        .get("VariationArchive", {})
        .get("ClassifiedRecord", {})
        .get("TraitMappingList", {})
        .get("TraitMapping")
    )

    if isinstance(diseases, list):
        for disease in diseases:
            temp["Diseases"].append(disease.get("MedGen", {}).get("@Name", ""))
    elif isinstance(diseases, dict):
        temp["Diseases"] = diseases.get("MedGen", {}).get("@Name", "")

    return temp


def retrieve_clinvar_entries(search_string: str, question: str):
    """
    For the searched keywords, fetch the list of docs/items found via the API
    """
    search_string = search_string.strip()

    # fetch bunch of ids
    search_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=clinvar&term={search_string}&sort=relevance"
    response = requests.get(search_url)
    if response.status_code == 400:
        return (
            "",
            "This is a bad request to the API, kindly check if the identifier is valid.",
        )
    response = xml_to_json(response.content)
    idlist = response.get("eSearchResult", {}).get("IdList", {}).get("Id", [])

    idlist = idlist[:10]
    all_docs_info = []
    for item in idlist:
        item = str(item)
        item = "VCV" + item.zfill(9)
        # get info for particular id
        search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        params = {
            "db": "clinvar",
            "id": item,
            "rettype": "vcv",
        }
        response = requests.get(search_url, params=params)
        if response.status_code == 400:
            return (
                "",
                "No entries found",
            )
        doc = xml_to_json(response.content)
        all_docs_info.append(clean_doc(doc))

    df = pd.DataFrame(all_docs_info)
    df = df.rename(columns={"AccessionID": "id"})
    return json.dumps(df.to_dict(orient="records")), ""


retrieve_clinvar_entries_tool = StructuredTool.from_function(
    func=retrieve_clinvar_entries,
    name="Retrieve_ClinVar",
    description="use this tool, to retrieve mutations from NCBI ClinVar based on the search string",
    args_schema=RetMutationPydanticInput,
)

retrieve_clinvar_entries_tool.description = """
    Use this tool, to retrieve mutations from NCBI ClinVar based on the search string.
    The tool takes the following arguments:
    - search_string: The search string are keyowrds for search on the NCBI ClinVar data source. This can be sometimes be separated by AND, OR operators.
    - question: The original question to find the answer for.
    Example input:
    {{"search_string": "DLG1 AND Humans", "question": 'What is the function of the PDZ domain of the DLG1 protein in humans?'}}
"""
