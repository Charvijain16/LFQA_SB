import json
import pandas as pd
import http.client
from typing import List
from Bio import Entrez
import pandas as pd
from http.client import http
from Bio import Entrez
import torch
from sentence_transformers import SentenceTransformer
from enum import Enum
import faiss
import numpy as np
from langchain.tools import StructuredTool
from src.tools.pydantic_inputs import RetPubMedPydanticInput


class SimilarityType(str, Enum):
    COSINE_SIMILARITY = "cosine_similarity"


class EmbeddingModel(str, Enum):
    SENTENCE_TRANSFORMERS = "sentence-transformers"
    UNIVERSAL_SENTENCE_ENCODER = "universal-sentence-encoder"
    NOMIC_EMBEDDINGS = "nomic-embeddings"
    JINA_EMBEDDINGS = "jina-embeddings"


def cosine_similarity(A, B):
    # cosine_similarity = A . B / (||A|| ||B||)
    cosine_similarity_scores = np.dot(A, B.T) / (
        np.linalg.norm(A) * np.linalg.norm(B, axis=1)
    )
    return cosine_similarity_scores.tolist()


def get_query_and_abs_embeddings(
    embedding_model: EmbeddingModel,
    input_query: str,
    list_of_abstracts: List[str],
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_query_embedding = None
    abstracts_embeddings = None

    if embedding_model == EmbeddingModel.SENTENCE_TRANSFORMERS:
        model = SentenceTransformer(
            "paraphrase-MiniLM-L6-v2"
        )  # limitation of max_length=512
        input_query_embedding = model.encode(
            sentences=[input_query], prompt="search_query"
        )
        abstracts_embeddings = model.encode(
            sentences=list_of_abstracts, prompt="search_document"
        )
    elif (
        embedding_model == EmbeddingModel.UNIVERSAL_SENTENCE_ENCODER
    ):  # doesnt work with the code, issue with tensorflow_hub in the current environment, makes the code halt.
        # model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
        # input_query_embedding = model([input_query])
        # abstracts_embeddings = model(list_of_abstracts)
        pass
    elif embedding_model == EmbeddingModel.NOMIC_EMBEDDINGS:
        model = SentenceTransformer(
            "nomic-ai/nomic-embed-text-v1", trust_remote_code=True
        )  # able to handle sequences of length upto 8192 tokens
        input_query_embedding = model.encode(
            sentences=[input_query], prompt="search_query"
        )
        abstracts_embeddings = model.encode(
            sentences=list_of_abstracts, prompt="search_document"
        )
    elif embedding_model == EmbeddingModel.JINA_EMBEDDINGS:
        model = SentenceTransformer(
            "jinaai/jina-embeddings-v2-base-en",  # switch to en/zh for English or Chinese
            trust_remote_code=True,
        )
        model.max_seq_length = 1024
        input_query_embedding = model.encode(
            sentences=[input_query], prompt="search_query"
        )
        abstracts_embeddings = model.encode(
            sentences=list_of_abstracts, prompt="search_document"
        )
    else:
        raise ValueError("Invalid model name")

    return input_query_embedding, abstracts_embeddings


def get_scores(
    input_query_embedding, abstracts_embeddings, similarity_type: SimilarityType
):
    scores = []
    if similarity_type == SimilarityType.COSINE_SIMILARITY:
        scores = cosine_similarity(input_query_embedding, abstracts_embeddings)
    else:
        raise ValueError("Invalid similarity type")
    return scores


def get_metadata(list_of_pmed_ids: List[str], include_reviews: bool = True):
    title_list = []
    abstract_list = []
    doi_list = []
    pubdate_year_list = []
    pubdate_month_list = []
    pmed_id_list = []
    pmc_id_list = []
    doi_references_list = []
    pmed_references_list = []
    exclude_keywords = ["survey", "review"]
    chunk_size = 5000
    for chunk_i in range(0, len(list_of_pmed_ids), chunk_size):
        chunk = list_of_pmed_ids[chunk_i : chunk_i + chunk_size]
        papers = fetch_details(chunk, retries=3)

        for i, paper in enumerate(papers["PubmedArticle"]):
            flag = 0
            for element in paper["MedlineCitation"]["Article"]["PublicationTypeList"]:
                if (
                    element.lower() in exclude_keywords
                ):  # exclude papers that are reviews
                    flag = 1
                    break

            if flag == 1 and not include_reviews:
                continue
            else:
                pmed_id_list.append(
                    next(
                        (
                            element
                            for element in paper["PubmedData"]["ArticleIdList"]
                            if element.attributes["IdType"] == "pubmed"
                        ),
                        None,
                    )
                )
                # assert pmed_id_list[-1] == chunk[i]
                pmc_id_list.append(
                    next(
                        (
                            element
                            for element in paper["PubmedData"]["ArticleIdList"]
                            if element.attributes["IdType"] == "pmc"
                        ),
                        None,
                    )
                )
                doi_list.append(
                    next(
                        (
                            element
                            for element in paper["PubmedData"]["ArticleIdList"]
                            if element.attributes["IdType"] == "doi"
                        ),
                        None,
                    )
                )
                title_list.append(paper["MedlineCitation"]["Article"]["ArticleTitle"])
                try:
                    abstract_list.append(
                        str(
                            paper["MedlineCitation"]["Article"]["Abstract"][
                                "AbstractText"
                            ][0]
                        )
                    )
                except:
                    abstract_list.append("No Abstract")

                try:
                    pubdate_year_list.append(
                        paper["MedlineCitation"]["Article"]["Journal"]["JournalIssue"][
                            "PubDate"
                        ]["Year"]
                    )
                except:
                    pubdate_year_list.append("No Data")
                try:
                    pubdate_month_list.append(
                        paper["MedlineCitation"]["Article"]["Journal"]["JournalIssue"][
                            "PubDate"
                        ]["Month"]
                    )
                except:
                    pubdate_month_list.append("No Data")

                doi_ref_list_paper = []
                pmed_red_list_paper = []
                if len(paper["PubmedData"]["ReferenceList"]) != 0:
                    for ref in paper["PubmedData"]["ReferenceList"][0]["Reference"]:
                        if "ArticleIdList" in ref:
                            doi_ref_list_paper.append(
                                next(
                                    (
                                        element
                                        for element in ref["ArticleIdList"]
                                        if element.attributes["IdType"] == "doi"
                                    ),
                                    None,
                                )
                            )
                            pmed_red_list_paper.append(
                                next(
                                    (
                                        element
                                        for element in ref["ArticleIdList"]
                                        if element.attributes["IdType"] == "pubmed"
                                    ),
                                    None,
                                )
                            )

                doi_references_list.append(doi_ref_list_paper)
                pmed_references_list.append(pmed_red_list_paper)

    df = pd.DataFrame(
        list(
            zip(
                pmed_id_list,
                pmc_id_list,
                doi_list,
                title_list,
                abstract_list,
                pubdate_year_list,
                pubdate_month_list,
                doi_references_list,
                pmed_references_list,
            )
        ),
        columns=[
            "PmedId",
            "PMC_IDS",
            "DOI",
            "Title",
            "Abstract",
            "Year",
            "Month",
            "References DOIs",
            "References PmedIds",
        ],
    )
    return df


def compare_input_abstract(
    input_query: str,
    list_of_abstracts: List[str],
    threshold_score: float = 0.10,
    embedding_model: EmbeddingModel = "nomic-embeddings",
    similarity_type: SimilarityType = "cosine_similarity",
):
    input_query_embedding, abstracts_embeddings = get_query_and_abs_embeddings(
        embedding_model, input_query, list_of_abstracts
    )

    # scores = get_scores(input_query_embedding, abstracts_embeddings, similarity_type)

    dimension = len(abstracts_embeddings[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(
        abstracts_embeddings
    )  # index.ntotal is the number of vectors in the index
    assert index.ntotal == len(list_of_abstracts)
    top_k = len(list_of_abstracts) if len(list_of_abstracts) < 10 else 10
    # https://github.com/facebookresearch/faiss/wiki/MetricType-and-distances
    distance, indices = index.search(
        input_query_embedding, top_k
    )  # returns a tuple of two numpy arrays, the first one contains the distances and the second one contains the indices of the vectors in the index.

    top_k_indicies_and_distance = {}
    for d, i in zip(distance[0], indices[0]):
        if d > threshold_score:
            top_k_indicies_and_distance[i] = d
        else:
            break

    return top_k_indicies_and_distance


def fetch_details(id_list, retries):
    try:
        handle = Entrez.efetch(db="pubmed", retmode="xml", id=id_list)
        results = Entrez.read(handle)
        return results
    except http.client.IncompleteRead as e:
        if retries > 0:
            fetch_details(id_list, retries - 1)
        else:
            raise e


def screening(input_query, df):
    # get abstracts
    list_of_abstracts = df["Abstract"].tolist()
    if len(list_of_abstracts) == 0:
        raise Exception(
            "No entries found. Maybe this search is too specific. Try again with another search or broader search!"
        )
    # returning only the papers whose abstract matches with the input query
    top_k_indices_n_scores = compare_input_abstract(
        input_query=input_query, list_of_abstracts=list_of_abstracts
    )

    top_k_indices = list(top_k_indices_n_scores.keys())
    top_k_scores = list(top_k_indices_n_scores.values())

    # remove all the papers from the dataframe except the relevant papers
    df = df.loc[top_k_indices]
    df["Relevance_score"] = top_k_scores
    return df


def search_on_PubMed(llm_phrase, input_query):
    if "," in llm_phrase:
        search_string, from_date, to_date = llm_phrase.split(",")

        handle = Entrez.esearch(
            db="pubmed",
            term=search_string,
            restart=0,
            retmax=3000,
            retmode="xml",
            datetype="pdat",
            sort="pub+date",
            mindate=from_date.strip(),
            maxdate=to_date.strip(),
        )
    else:
        from_date = ""
        to_date = ""
        search_string = llm_phrase
        handle = Entrez.esearch(
            db="pubmed",
            term=search_string,
            restart=0,
            retmax=1500,
            retmode="xml",
            datetype="pdat",
            sort="relevance",
            maxdate="2025/03/31"
        )

    results = Entrez.read(handle)
    list_of_pmed_ids = results.get("IdList")
    if len(list_of_pmed_ids) > 1500:
        raise Exception(
            "Results exceed 1500 entries, please try with a more specific search query."
        )

    df = get_metadata(list_of_pmed_ids)
    filtered_df = screening(input_query, df)
    return filtered_df[:10]


def retrieve_pubmed_papers(search_string, question):
    search_string = search_string.strip()
    try:
        PUBMED_DATABASE = search_on_PubMed(
            llm_phrase=search_string, input_query=question
        )
        df = PUBMED_DATABASE.drop(
            columns=["DOI", "Relevance_score", "References DOIs", "References PmedIds"],
            axis=1,
        )
        df = df.rename(columns={"PmedId": "id"})
        df = df.rename(columns={"PMC_IDS": "pmcid"})
        return json.dumps(df.to_dict(orient="records")), ""
    except Exception as e:
        return "", str(e)


retrieve_pubmed_papers_tool = StructuredTool.from_function(
    func=retrieve_pubmed_papers,
    name="Retrieve_PubMed",
    description="use this tool, to retrieve papers from PubMed based on the search string",
    args_schema=RetPubMedPydanticInput,
)

retrieve_pubmed_papers_tool.description = """
    Use this tool, to retrieve papers from PubMed based on the search string.
    The tool takes the following arguments:
    - search_string: The search string are keyowrds for search on the PubMed data source. This can be sometimes be separated by AND, OR operators.
    Example input:
    {{"search_string": "DLG1 AND Humans", "question": 'What is the function of the PDZ domain of the DLG1 protein in humans?'}}
"""

if __name__ == "__main__":
    a, b = retrieve_pubmed_papers(
        search_string="eosinophils AND osteoclastogenesis",
        question="What causes the downregulation of osteoclastogenic genes upon interaction with eosinophils?",
    )
    print(a, b)
