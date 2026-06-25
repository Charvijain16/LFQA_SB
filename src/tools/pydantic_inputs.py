from pydantic import BaseModel, Field


class RetMutationPydanticInput(BaseModel):
    search_string: str = Field(
        description="The search string are keyowrds for search on the NCBI ClinVar data source. This can be sometimes be separated by AND, OR operators."
    )
    question: str = Field(
        description="The question to which you need the answer from the NCBI ClinVar data source"
    )
    
    
class SummMutationPydanticInput(BaseModel):
    search_string: str = Field(
        description="The search string is an identifier or couples of identifier separated by comma for NCBI ClinVar Page(s) beginning with 'VCV' and followed by a numeric value "
    )
    question: str = Field(
        description="The question to which you need the answer from the NCBI ClinVar Page"
    )
    
    
class RetGenesPydanticInput(BaseModel):
    search_string: str = Field(
        description="The search string are keyowrds for search on the NCBI Genes data source. This can be sometimes be separated by AND, OR operators."
    )
    question: str = Field(
        description="The question to which you need the answer from the NCBI Genes data source"
    )


class SummGenesPydanticInput(BaseModel):
    search_string: str = Field(
        description="The search string is an identifier or couples of identifier separated by comma for NCBI Genes Page(s). These are usually numeric values. "
    )
    question: str = Field(
        description="The question to which you need the answer from the NCBI Genes Page"
    )

class RetKeggPydanticInput(BaseModel):
    search_string: str = Field(
        description="The search string are keyowrds for search on the KEGG Pathways data source. This can be sometimes be separated by AND, OR operators."
    )
    question: str = Field(
        description="The question to which you need the answer from the KEGG Pathways data source"
    )

class SummKeggPydanticInput(BaseModel):
    search_string: str = Field(
        description="The search string is an identifier or couples of identifier separated by comma for KEGG Pathways Page(s) beginning with 'map' and followed by a numeric value"
    )
    question: str = Field(
        description="The question to which you need the answer from the KEGG Pathways Page"
    )

class RetPubMedPydanticInput(BaseModel):
    search_string: str = Field(
        description="The search string are keyowrds for search on the PubMed data source. This can be sometimes be separated by AND, OR operators."
    )
    question: str = Field(
        description="The question to which you need the answer from the PubMed data source"
    )

class SummPubmedPydanticInput(BaseModel):
    search_string: str = Field(
        description="The search string is an identifier or couples of identifier separated by comma for PubMedCentral Page(s) beginning with 'PMC'."
    )
    question: str = Field(
        description="The question to which you need the answer from the PubMedCentral Page"
    )

class RetUniProtPydanticInput(BaseModel):
    search_string: str = Field(
        description="The search string are keyowrds for search on the UniProt data source. This can be sometimes be separated by AND, OR operators."
    )
    question: str = Field(
        description="The question to which you need the answer from the UniProt data source"
    )

class SummUniProtPydanticInput(BaseModel):
    search_string: str = Field(
        description="The search string is an identifier or couples of identifier separated by comma for UniProt Page(s) and each identifier is a mix of alphabet letters and numbers."
    )
    question: str = Field(
        description="The question to which you need the answer from the UniProt Page"
    )

class ExchangePMCPMIDPydanticInput(BaseModel):
    search_string: str = Field(
        description="The search string is an identifier or couples of identifier separated by comma, either numeric values or begininning with 'PMC'."
    )
    question: str = Field(
        description="The question to which you need the answer."
    )
    
class FetchInfoAnySourcePydanticInput(BaseModel):
    question: str = Field(
        description="The question to which you need the answer."
    )
    thought: str = Field(
        description="The current thought"
    )
    search_string: str = Field(
        description="The search string is the name of source from the list --NCBI Genes, UniProt, KEGG, PubMedCentral, NCBI ClinVar-- and an identifier separated by comma for a particular datasource that has to be queried."
    )
    


