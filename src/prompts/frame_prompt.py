from src.tools.clinvar.retrieve_mutations import retrieve_clinvar_entries_tool
from src.tools.clinvar.summarize_mutations import summarize_clinvar_page_tool
from src.tools.genes.retrieve_genes import retrieve_gene_entries_tool
from src.tools.genes.summarize_genes import summarize_gene_page_tool
from src.tools.kegg.retrieve_kegg_pathways import retrieve_kegg_pathways_tool
from src.tools.kegg.summarize_kegg_page import summarize_kegg_pathways_tool
from src.tools.pubmed.retrieve_pubmed_papers import retrieve_pubmed_papers_tool
from src.tools.pubmed.summarize_pubmed_paper import summarize_pubmed_papers_tool
from src.tools.uniprot.retrieve_uniprot_entries import retrieve_uniprot_entries_tool
from src.tools.uniprot.summarize_uniprot_page import summarize_uniprot_entries_tool
from src.tools.exchange_pmc_pmid import exchange_pmc_pmid_tool
from src.tools.fetch_any_info import fetch_info_any_source_tool
from langchain.tools.convert_to_openai import format_tool_to_openai_function
from transformers import AutoTokenizer

# System message to be placed at the beginning
SYSTEM_MESSAGE = """You have no prior knowledge. Your goal is to gather information and evidences for the question from different data sources namely NCBI Genes, UniProt, KEGG, PubMed, and NCBI ClinVar. Follow multiple iterations of thought/relevance/action/actioninput/observation (the solution-process). Each thought or argument prompts an action that, using the provided input, returns an observation. Based on that observation, a relevance identifier is stored, leading to the next thought. The thought doesn't summarize the entire previous observation; it simply advances the logic. Thoughts should begin simple and progressively evolve into more complex and specific reasoning over iterations. If a chosen direction does not yield sufficient evidence, you are allowed to change its approach logically. For actions, only use one among {tool_names} tools in one iteration. Action inputs should remain concise and well-targeted to maximize the relevance and breadth of search results. You must perform a thorough exploration across available sources, ensuring every aspect of the question is adequately addressed. End the response with <<<LLAMA3_3_EOS>>>."""

PREFIX = """You have access to the following tools: """
# Define the tool instruction section
TOOL_INSTRUCTION = """PubMed can only retrieve the list of papers and provide metadata information. To get the full text of a paper, use PubMedCentral. For finding or summarizing proteins use UniProt, for pathways use KEGG, for genes use Genes and for mutations use ClinVAR.
For tools with retrieve functionality, always provide the search_string with apt keywords and make use of AND, OR operators for better results. Also, pass the model organism in the search_string if possible, to make the search narrow. For example: '(DLG1 OR CDC42) AND "Home Sapiens"'. For tools with summarize functionality, only give the identifier or list of identifiers as action input. If you find an identifier and its description relevant, and, if necessary, extract key information from the full page by using the summarize functions. Remember: For Summarize_PubMedCentral_Page, always give the 'pmcid' as the action input, else always use 'id' for other tools."""

# Format instructions section
FORMAT_INSTRUCTIONS = """Use the following format always:
Thought: you should always think about what to do
Relevance: if a previous observation exits, enter the identifer from it, if the information claimed with the identifier is relevant in the context of the previous thought and question otherwise, enter a hyphen; The identifier will always be a field 'id' in the previous observation and you have to take its value. If multiple identifers are relevant, then separate them by comma.
Action: the action to take, should be one of the [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Relevance/Action/Action Input/Observation/Thought can repeat N times)
Thought: I have now collected sufficient evidences and information for the question from the available data sources.
Relevance: relevant identifiers from previous observation.
<<<LLAMA3_3_EOS>>>"""

SUFFIX = """
Question: {input}
Thought:{agent_scratchpad}"""

openai_retrieve_clinvar_entries_tool = {
    'name': 'Retrieve_ClinVar',
    'description': 'Use this tool, to retrieve mutations from NCBI ClinVar based on the search string.',
    'parameters': {
        'properties': {
            'search_string': {
                'description': 'The search string are keyowrds for search on the NCBI ClinVar data source. This can be sometimes be separated by AND, OR operators.',
                'type': 'string',
            },
            'question': {
                'description': 'The question to which you need the answer from the NCBI ClinVar data source',
                'type': 'string',
            },
        },
        'required': ['search_string', 'question'],
        'type': 'object',
    },
}
openai_summarize_clinvar_page_tool = {
    'name': 'Summarize_CliVar_Page',
    'description': 'Use this tool, to summarize information on the ClinVar page for an identifier with respect to the question.',
    'parameters': {
        'properties': {
            'search_string': {
                'description': 'The search string is an identifier or couples of identifier separated by comma for NCBI ClinVar Page(s) beginning with VCV and followed by a numeric value ',
                'type': 'string',
            },
            'question': {
                'description': 'The question to which you need the answer from the NCBI ClinVar Page',
                'type': 'string',
            },
        },
        'required': ['search_string', 'question'],
        'type': 'object',
    },
}
openai_retrieve_gene_entries_tool = {
    'name': 'Retrieve_Gene',
    'description': 'Use this tool, to retrieve genes from NCBI Genes based on the search string.',
    'parameters': {
        'properties': {
            'search_string': {
                'description': 'The search string are keyowrds for search on the NCBI Genes data source. This can be sometimes be separated by AND, OR operators.',
                'type': 'string',
            },
            'question': {
                'description': 'The question to which you need the answer from the NCBI Genes data source',
                'type': 'string',
            },
        },
        'required': ['search_string', 'question'],
        'type': 'object',
    },
}
openai_summarize_gene_page_tool = {
    'name': 'Summarize_Gene_Page',
    'description': 'Use this tool, to summarize information on the NCBI Genes page for an identifier with respect to the question.',
    'parameters': {
        'properties': {
            'search_string': {
                'description': 'The search string is an identifier or couples of identifier separated by comma for NCBI Genes Page(s). These are usually numeric values. ',
                'type': 'string',
            },
            'question': {
                'description': 'The question to which you need the answer from the NCBI Genes Page',
                'type': 'string',
            },
        },
        'required': ['search_string', 'question'],
        'type': 'object',
    },
}
openai_retrieve_kegg_pathways_tool = {
    'name': 'Retrieve_KEGG',
    'description': 'Use this tool, to retrieve pathways from KEGG Pathways based on the search string.',
    'parameters': {
        'properties': {
            'search_string': {
                'description': 'The search string are keyowrds for search on the KEGG Pathways data source. This can be sometimes be separated by AND, OR operators.',
                'type': 'string',
            },
            'question': {
                'description': 'The question to which you need the answer from the KEGG Pathways data source',
                'type': 'string',
            },
        },
        'required': ['search_string', 'question'],
        'type': 'object',
    },
}
openai_summarize_kegg_pathways_tool = {
    'name': 'Summarize_KEGG_Page',
    'description': 'Use this tool, to summarize information on the KEGG Pathway page for an identifier with respect to the question.',
    'parameters': {
        'properties': {
            'search_string': {
                'description': 'The search string is an identifier or couples of identifier separated by comma for KEGG Pathways Page(s) beginning with map and followed by a numeric value',
                'type': 'string',
            },
            'question': {
                'description': 'The question to which you need the answer from the KEGG Pathways Page',
                'type': 'string',
            },
        },
        'required': ['search_string', 'question'],
        'type': 'object',
    },
}
openai_retrieve_pubmed_papers_tool = {
    'name': 'Retrieve_PubMed',
    'description': 'Use this tool, to retrieve papers from PubMed based on the search string.',
    'parameters': {
        'properties': {
            'search_string': {
                'description': 'The search string are keyowrds for search on the PubMed data source. This can be sometimes be separated by AND, OR operators.',
                'type': 'string',
            },
            'question': {
                'description': 'The question to which you need the answer from the PubMed data source',
                'type': 'string',
            },
        },
        'required': ['search_string', 'question'],
        'type': 'object',
    },
}
openai_summarize_pubmed_papers_tool = {
    'name': 'Summarize_PubMedCentral_Page',
    'description': 'Use this tool, to summarize information on the PubMedCentral paper with respect to the question.',
    'parameters': {
        'properties': {
            'search_string': {
                'description': 'The search string is an identifier or couples of identifier separated by comma for PubMedCentral Page(s) beginning with PMC',
                'type': 'string',
            },
            'question': {
                'description': 'The question to which you need the answer from the PubMedCentral Page',
                'type': 'string',
            },
        },
        'required': ['search_string', 'question'],
        'type': 'object',
    },
}
openai_retrieve_uniprot_entries_tool = {
    'name': 'Retrieve_UniProt_Page',
    'description': '\n    Use this tool, to retrieve proteins from UniProt based on the search string.',
    'parameters': {
        'properties': {
            'search_string': {
                'description': 'The search string are keyowrds for search on the UniProt data source. This can be sometimes be separated by AND, OR operators.',
                'type': 'string',
            },
            'question': {
                'description': 'The question to which you need the answer from the UniProt data source',
                'type': 'string',
            },
        },
        'required': ['search_string', 'question'],
        'type': 'object',
    },
}
openai_fetch_info_any_source_tool = {
    'name': 'Fetch_Info_Any_Source',
    'description': 'Use this tool, to find a particular information based on the thought from any of the sources namely NCBI Genes, UniProt, KEGG, PubMedCentral, NCBI ClinVar.',
    'parameters': {
        'properties': {
            'question': {
                'description': 'The question to which you need the answer.',
                'type': 'string',
            },
            'thought': {'description': 'The current thought', 'type': 'string'},
            'search_string': {
                'description': "The search string is the name of source from the list --NCBI Genes, UniProt, KEGG, PubMedCentral, NCBI ClinVar-- and an identifier separated by comma for a particular datasource that has to be queried, e.g., 'UniProt, AQ34U1'",
                'type': 'string',
            },
        },
        'required': ['question', 'thought', 'search_string'],
        'type': 'object',
    },
}
openai_exchange_pmc_pmid_tool = {
    'name': 'Exchange_PMC_PMID',
    'description': 'Use this tool, to find the PubMed identifier corresponding to PubMedCentral identifer or vice-a-versa.',
    'parameters': {
        'properties': {
            'search_string': {
                'description': 'The search string is an identifier or couples of identifier separated by comma, either numeric values or begininning with PMC',
                'type': 'string',
            },
            'question': {
                'description': 'The question to which you need the answer.',
                'type': 'string',
            },
        },
        'required': ['search_string', 'question'],
        'type': 'object',
    },
}
openai_summarize_uniprot_entries_tool = {
    'name': 'Summarize_UniProt_Page',
    'description': 'Use this tool, to summarize information on the UniProt page for an identifier with respect to the question.',
    'parameters': {
        'properties': {
            'search_string': {
                'description': 'The search string is an identifier or couples of identifier separated by comma for UniProt Page(s) and each identifier is a mix of alphabet letters and numbers.',
                'type': 'string',
            },
            'question': {
                'description': 'The question to which you need the answer from the UniProt Page',
                'type': 'string',
            },
        },
        'required': ['search_string', 'question'],
        'type': 'object',
    },
}

tools = [
    openai_retrieve_clinvar_entries_tool,
    openai_summarize_clinvar_page_tool,
    openai_retrieve_gene_entries_tool,
    openai_summarize_gene_page_tool,
    openai_retrieve_kegg_pathways_tool,
    openai_summarize_kegg_pathways_tool,
    openai_retrieve_pubmed_papers_tool,
    openai_summarize_pubmed_papers_tool,
    openai_retrieve_uniprot_entries_tool,
    openai_summarize_uniprot_entries_tool,
    openai_exchange_pmc_pmid_tool,
    openai_fetch_info_any_source_tool,
]

tool_names = ", ".join([tool["name"] for tool in tools])


def format_prompt_per_model(model_name: str, tool_names=tool_names, tools=tools):
    # load the tokenizer and the model
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    system_message = SYSTEM_MESSAGE.format(tool_names=tool_names)
    format_instructions = FORMAT_INSTRUCTIONS.format(tool_names=tool_names)

    system_prompt = "\n\n".join(
        [
            system_message,
            "TOOL INSTRUCTION:",
            TOOL_INSTRUCTION,
            "FORMAT INSTRUCTION:",
            format_instructions,
        ]
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "{{{input}}}"},
        {"role": "assistant", "content": "Thought:{{{agent_scratchpad}}}"},
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,  # Switches between thinking and non-thinking modes. Default is True.
    )

    return text
