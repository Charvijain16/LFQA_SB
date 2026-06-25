import os
from Bio import Entrez

Entrez.email = os.environ.get("ENTREZ_EMAIL")
Entrez.api_key = os.environ.get("ENTREZ_API_KEY")
Entrez.sleep_between_tries = 15
Entrez.max_tries = 3

