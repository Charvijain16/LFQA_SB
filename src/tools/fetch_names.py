import requests
from Bio import UniProt
from Bio.KEGG import REST


def fetch_gene_names(gene_ids):
    """
    Fetch gene symbols (and descriptions) from NCBI for a list of Gene IDs.

    :param gene_ids: List of Gene IDs (integers or strings)
    :return: A dictionary mapping each Gene ID to a dict of gene symbol and description.
    """

    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
    params = {
        "db": "gene",
        "id": ",".join(str(gid) for gid in gene_ids),
        "retmode": "json",
    }

    # Send request to NCBI E-utilities
    response = requests.get(base_url, params=params)
    response.raise_for_status()  # Raises an HTTPError if the request returned an unsuccessful status code

    data = response.json()
    # 'data["result"]' should contain the main dictionary of items
    result_dict = data.get("result", {})

    # The gene data is indexed by string versions of the IDs
    # Example structure: data["result"]["196"], data["result"]["66713"], etc.
    gene_info = []

    for gid in gene_ids:
        gid_str = str(gid)
        if gid_str in result_dict:
            gene_record = result_dict[gid_str]

            # gene_record typically has 'name' (symbol) and 'description' (full name/short description)
            gene_symbol = gene_record.get("name", "N/A")
            gene_description = gene_record.get("description", "N/A")

            gene_info.append(
                (gene_symbol + ", " + gene_description + ", " + f"GeneID:{gid}")
            )
        else:
            gene_info.append(("No Gene description" + ", " + f"GeneID:{gid}"))

    return gene_info


def fetch_protein_names(protein_ids):
    results = []

    for uniprot_id in protein_ids:
        # Search UniProt for the given ID
        query_result = UniProt.search(uniprot_id)

        if query_result:
            # Extract protein name (Field: "Protein names")
            protein_name = query_result[0].get("uniProtkbId", "N/A")
            results.append((protein_name + ", " + f"ProteinID:{uniprot_id}"))
        else:
            results.append(("No name found" + ", " + f"ProteinID:{uniprot_id}"))

    return results


def fetch_pathways_names(pathways_ids):
    results = []
    for kegg_id in pathways_ids:
        try:
            # Query KEGG for the given identifier
            response = REST.kegg_get(kegg_id).read()

            # Extract the pathway name (first line after the 'NAME' field)
            lines = response.split("\n")
            name_line = next((line for line in lines if line.startswith("NAME")), None)

            if name_line:
                pathway_name = name_line.replace("NAME", "").strip()
                results.append((pathway_name + ", " + f"KEGG_PATHWAY_ID:{kegg_id}"))
            else:
                results.append(("No name found" + ", " + f"KEGG_PATHWAY_ID:{kegg_id}"))

        except Exception as e:
            results.append((f"Error: {str(e)}" + ", " + f"KEGG_PATHWAY_ID:{kegg_id}"))
            # results = f"Error: {str(e)}"

    return results


def fetch_mutations_names(mutation_ids):
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
    results = []

    for vcv_id in mutation_ids:
        try:
            params = {
                "db": "clinvar",
                "id": vcv_id.lstrip("VCV"),  # ClinVar uses only numeric part
                "retmode": "json",
            }

            # Fetch data from ClinVar
            response = requests.get(base_url, params=params)
            response.raise_for_status()

            data = response.json()
            result_dict = data.get("result", {})
            variant_data = result_dict.get(params["id"], {})

            # Extract variant title (clinically relevant name)
            variant_name = variant_data.get("title", "Not found")
            results.append((variant_name + ", " + f"MUTATION_ID:{vcv_id}"))

        except Exception as e:
            results.append(("No name found" + ", " + f"MUTATION_ID:{vcv_id}"))
            # results[vcv_id] = f"Error: {str(e)}"

    return results


def get_pmcid_to_pmid(pmcids):
    """
    Given an iterable of PMCIDs (strings starting with 'PMC'),
    returns a dict mapping each PMC ID to its corresponding PMID (string),
    or to None if no PMID is found.
    """
    if not pmcids:
        return {}

    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
    # strip the “PMC” prefix for the query
    numeric_ids = [pmcid.replace("PMC", "", 1) for pmcid in pmcids]
    params = {"db": "pmc", "id": ",".join(numeric_ids), "retmode": "json"}
    resp = requests.get(base_url, params=params)
    resp.raise_for_status()
    data = resp.json().get("result", {})
    mapping = {}
    # “uids” is the list of numeric IDs you asked for
    for uid in data.get("uids", []):
        pmcid = "PMC" + uid
        article = data.get(uid, {})
        pmid = None
        # ESummary for PMC returns an “articleids” list of dicts
        # each with keys like {"idtype":"pmid", "value":"12345678"}
        for aid in article.get("articleids", []):
            if aid.get("idtype", "").lower() == "pmid":
                pmid = aid.get("value")
                break
        mapping[pmcid] = pmid
    return mapping


def dedupe_article_ids(article_ids):
    """
    Input: iterable of mixed IDs, e.g. ["12345","PMC67890","PMC11111","22222"]
    Output: list where, if both “PMC67890”→“12345” and “12345” were present,
            only “12345” remains; any PMCIDs without a mapped PMID stay as PMC###
    """
    # split out what you’ve already got
    pmids = {aid for aid in article_ids if not aid.startswith("PMC")}
    pmcs = {aid for aid in article_ids if aid.startswith("PMC")}

    # fetch the PMC→PMID mapping
    mapping = get_pmcid_to_pmid(pmcs)

    final = set(pmids)
    for pmc in pmcs:
        pmid = mapping.get(pmc)
        if pmid:
            # if that PMID was already in your list, drop the PMC
            if pmid in pmids:
                continue
            # otherwise, add the PMID instead of the PMC
            final.add(pmid)
        else:
            # no PMID found for that PMC, so keep the PMC
            final.add(pmc)

    return list(final)


def fetch_paper_names(article_ids):
    results = []

    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"

    for article_id in article_ids:
        # Determine database type (PubMed = "pubmed", PMC = "pmc")
        db = "pmc" if str(article_id).startswith("PMC") else "pubmed"

        params = {
            "db": db,
            "id": article_id.lstrip("PMC"),  # PMC IDs require numeric only
            "retmode": "json",
        }

        try:
            # Fetch data from PubMed/PMC
            response = requests.get(base_url, params=params)
            response.raise_for_status()

            data = response.json()
            result_dict = data.get("result", {})
            article_data = result_dict.get(params["id"], {})

            # Extract title
            title = article_data.get("title", "Title not found")
            results.append((title + ", " + f"ID:{article_id}"))

        except Exception as e:
            results.append(("error occured" + ", " + f"ID:{article_id}"))
            # results[article_id] = f"Error: {str(e)}"

    return results
