from http.client import http
from Bio import Entrez
from sentence_transformers import SentenceTransformer
from torch.nn.functional import cosine_similarity


def fetch_details(id_list, retries):
    if len(id_list)==0:
        return {}
    try:
        handle = Entrez.efetch(db="pubmed", retmode="xml", id=id_list)
        results = Entrez.read(handle)
        return results
    except http.client.IncompleteRead as e:
        if retries > 0:
            fetch_details(id_list, retries - 1)
        else:
            # logging.error("Failed to fetch the data")
            raise e


def get_abstracts(results):
    list_of_dict_of_abs = []
    for paper in results["PubmedArticle"]:
        pmedid = next(
            (
                element
                for element in paper["PubmedData"]["ArticleIdList"]
                if element.attributes["IdType"] == "pubmed"
            ),
            None,
        )
        try:
            abstract = (
                paper["MedlineCitation"]["Article"]["ArticleTitle"]
                + "\n"
                + paper["MedlineCitation"]["Article"]["Abstract"]["AbstractText"][0]
            )
        except:
            abstract = paper["MedlineCitation"]["Article"]["ArticleTitle"]
        list_of_dict_of_abs.append({"id": str(pmedid), "description": str(abstract)})

    return list_of_dict_of_abs


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


def filter_relvant_publications(question, references):
    Entrez.email = "charvi.jain@tu-dresden.de"
    Entrez.api_key = "ef00f25c2ba58d2c308f3f17c7b9157efe08"
    Entrez.sleep_between_tries = 10
    Entrez.max_tries = 3
    # id_list = [item['pubmed_id'] for item in references]
    relevant_references = []
    if len(references) == 0:
        return relevant_references
    results = fetch_details(id_list=references, retries=3)

    list_of_dict_of_abs = get_abstracts(results)
    top_k = retrieve_rel_docs(question=question, all_docs_info=list_of_dict_of_abs)
    

    for paper in results["PubmedArticle"]:
        paper_id = paper["MedlineCitation"]["PMID"].strip()
        for topk_id, _, _ in top_k:
            if topk_id == paper_id:
                relevant_references.append(
                    {
                        "pubmed_id": topk_id,
                        "title": paper["MedlineCitation"]["Article"]["ArticleTitle"],
                        "year": paper["MedlineCitation"]["Article"]["Journal"][
                            "JournalIssue"
                        ]["PubDate"]["Year"],
                    }
                )
                break
            else:
                continue

    return relevant_references
