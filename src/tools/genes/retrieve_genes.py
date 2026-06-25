import json
from sentence_transformers import SentenceTransformer

from torch.nn.functional import cosine_similarity
import pandas as pd
from Bio import Entrez
from langchain.tools import StructuredTool
from src.tools.pydantic_inputs import RetGenesPydanticInput

Entrez.email = "charvi.jain@tu-dresden.de"
Entrez.api_key = "ef00f25c2ba58d2c308f3f17c7b9157efe08"
Entrez.sleep_between_tries = 15
Entrez.max_tries = 3


def fetch_docs_from_API(search_string: str):
    """
    For the searched keywords, fetch the list of docs/items found via the API
    """
    handle = Entrez.esearch(
        db="gene",
        term=search_string,
        sort="relevance",  # idtype="acc"
    )
    record = Entrez.read(handle)
    id_list = record["IdList"]
    chunk_size = 5
    fetched_resuts = []
    id_list = id_list[:10]
    for chunk_i in range(0, len(id_list), chunk_size):
        chunk = id_list[chunk_i : chunk_i + chunk_size]

        try:
            handle = Entrez.efetch(
                db="gene", id=",".join(chunk), rettype="gb", retmode="xml"
            )  ##  rettype="gb" can also be "doc_sum" (document summary)
            records = Entrez.read(handle)  # Print or process the fetched data
            for record in records:
                temp = {}
                temp["geneID"] = (
                    record.get("Entrezgene_track-info", {})
                    .get("Gene-track", {})
                    .get("Gene-track_geneid", "")
                )
                temp["gene_name"] = record.get("Entrezgene_gene", {}).get(
                    "Gene-ref", {}
                ).get("Gene-ref_locus", "") or record.get("Entrezgene_gene", {}).get(
                    "Gene-ref", {}
                ).get(
                    "Gene-ref_formal-name", {}
                ).get(
                    "Gene-nomenclature", {}
                ).get(
                    "Gene-nomenclature_symbol", ""
                )
                temp["description"] = record.get("Entrezgene_gene", {}).get(
                    "Gene-ref", {}
                ).get("Gene-ref_desc", "") or record.get("Entrezgene_gene", {}).get(
                    "Gene-ref", {}
                ).get(
                    "Gene-ref_formal-name", {}
                ).get(
                    "Gene-nomenclature", {}
                ).get(
                    "Gene-nomenclature_name", ""
                )
                temp["organism"] = (
                    record.get("Entrezgene_source", {})
                    .get("BioSource", {})
                    .get("BioSource_org", {})
                    .get("Org-ref", {})
                    .get("Org-ref_taxname", "")
                )
                temp["summary"] = str(record.get("Entrezgene_summary", ""))
                # temp["references"] = ""
                fetched_resuts.append(temp)
        except Exception as e:
            print(f"An error occurred: {e}")

    return fetched_resuts


def retrieve_rel_docs(question, fetched_results):
    """
    From the list of docs/items recieved from the API, filter down the relevant items based on organism type, description, etc.
    """
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    protein_ids = [d["geneID"] for d in fetched_results if "geneID" in d]
    descriptions = [d["summary"] for d in fetched_results if "summary" in d]

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


def retrieve_gene_entries(search_string, question):
    search_string = search_string.strip()
    if question is None or question == "":
        print("Error: The question is None or empty")
        # raise Exception("Error: The question is None or empty")


    fetched_results = fetch_docs_from_API(search_string=search_string)

    if len(fetched_results) == 0:
        return (
            "",
            "No results were returned from the API. Try another search if can be.",
        )
    df_filtered = pd.DataFrame(fetched_results)
    df_filtered = df_filtered.rename(columns={"geneID": "id"})

    # df_filtered=df_filtered.drop(columns=["references"], axis=1)
    return json.dumps(df_filtered.to_dict(orient="records")), ""


retrieve_gene_entries_tool = StructuredTool.from_function(
    func=retrieve_gene_entries,
    name="Retrieve_Gene",
    description="use this tool, to retrieve genes from NCBI Genes based on the search string",
    args_schema=RetGenesPydanticInput,
)

retrieve_gene_entries_tool.description = """
    Use this tool, to retrieve genes from NCBI Genes based on the search string.
    The tool takes the following arguments:
    - search_string: The search string are keyowrds for search on the NCBI Genes data source. This can be sometimes be separated by AND, OR operators.
    - question: The original question to find the answer for.
    Example input:
    {{"search_string": "DLG1 AND Humans", "question": 'What is the function of the PDZ domain of the DLG1 protein in humans?'}}
"""
