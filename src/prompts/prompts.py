
def get_prompt_by_llm(llm:str):
    
    if "gpt" in llm or "claude" in llm:
        ## GPT4-o
        prepend_template2 = """You have no prior knowledge. Your goal is to gather information and evidences for the question from different data sources namely NCBI Genes, UniProt, KEGG, PubMed, and NCBI ClinVar. Follow multiple iterations of thought/relevance/action/actioninput/observation (the solution-process). Each thought or argument prompts an action that, using the provided input, returns an observation. Based on that observation, a relevance identifier is stored, leading to the next thought. The thought doesn’t summarize the entire previous observation; it simply advances the logic. Thoughts should begin simple and progressively evolve into more complex and specific reasoning over iterations. If a chosen direction does not yield sufficient evidence, you are allowed to change its approach logically. Relevance should always be either identifier or list of identifiers separated by comma based on previous observation or hyphen and never a running text. For actions, only use one among {tool_names} tools in one iteration. Action inputs should remain concise and well-targeted to maximize the relevance and breadth of search results. You must perform a thorough exploration across availavle sources, ensuring every aspect of the question is adequately addressed. End the response with <<<LLAMA3_3_EOS>>>."""

        # react style prompting
        PREFIX = """You have access to the following tools: """
        TOOL_INSTRUCTION = """PubMed can only retrieve the list of papers and provide metadata information. To get the full text of a paper, use PubMedCentral.  For finding or summarizing proteins use UniProt, for pathways use KEGG, for genes use Genes and for mutations use ClinVAR. 
        For tools with retrieve functionality, always provide the search_string with apt keywords and make use of AND, OR operators for better results. Also, pass the model organism in the search_string if possible, to make the search narrow. For example: '(DLG1 OR CDC42) AND "Home Sapiens"'. For tools with summarize functionality, only give the identifier or list of identifiers as action input. If you find an identifier and its description relevant, and, if necessary, extract key information from the full page by using the summarize functions. Remember: For Summarize_PubMedCentral_Page, always give the 'pmcid' as the action input, else always use 'id' for other tools."""

        FORMAT_INSTRUCTIONS = """Use the following format always:
        Thought: you should always think about what to do
        Relevance: if a previous observation exits, enter the identifer from it, if the information claimed with the identifier is relevant in the context of the previous thought and question otherwise, enter a hyphen; The identifier will always be a field 'id' in the previous observation and you have to take its value. If multiple identifers are relevant, then separate them by comma.
        Action: the action to take, should be one of the [{tool_names}]
        Action Input: the input to the action
        Observation: the result of the action
        ... (this Thought/Relevance/Action/Action Input/Observation/Thought can repeat N times)
        Thought: I have now collected sufficient evidences and information for the question from the available data sources. 
        Relevance: Relevant ids from previous observation. 
        <<<LLAMA3_3_EOS>>> """

        SUFFIX = """
        Question: {input}
        Thought:{agent_scratchpad}"""
        
    elif "gemini" in llm:
        prepend_template2 = """You have no prior knowledge. Your goal is to gather information and evidences for the question from different data sources namely NCBI Genes, UniProt, KEGG, PubMed, and NCBI ClinVar. Follow multiple iterations of thought/relevance/action/actioninput/observation (the solution-process). Each thought or argument prompts an action that, using the provided input, returns an observation. Based on that observation, a relevance identifier is stored, leading to the next thought. The thought doesn’t summarize the entire previous observation; it simply advances the logic. Thoughts should begin simple and progressively evolve into more complex and specific reasoning over iterations. If a chosen direction does not yield sufficient evidence, you are allowed to change its approach logically. Relevance should always be either identifier or list of identifiers separated by comma based on previous observation or hyphen and never a running text. For actions, only use one among {tool_names} tools in one iteration. Action inputs should remain concise and well-targeted to maximize the relevance and breadth of search results. You must perform a thorough exploration across availavle sources, ensuring every aspect of the question is adequately addressed. End the response with <<<LLAMA3_3_EOS>>>."""

        # react style prompting
        PREFIX = """You have access to the following tools: """
        TOOL_INSTRUCTION = """PubMed can only retrieve the list of papers and provide metadata information. To get the full text of a paper, use PubMedCentral.  For finding or summarizing proteins use UniProt, for pathways use KEGG, for genes use Genes and for mutations use ClinVAR. 
        For tools with retrieve functionality, always provide the search_string with apt keywords and make use of AND, OR operators for better results. Also, pass the model organism in the search_string if possible, to make the search narrow. For example: '(DLG1 OR CDC42) AND "Home Sapiens"'. For tools with summarize functionality, only give the identifier or list of identifiers as action input. If you find an identifier and its description relevant, and, if necessary, extract key information from the full page by using the summarize functions. Remember: For Summarize_PubMedCentral_Page, always give the 'pmcid' as the action input, else always use 'id' for other tools."""

        FORMAT_INSTRUCTIONS = """Use the following format always:
        Thought: you should always think about what to do
        Relevance: if a previous observation exits, enter the identifer from it, if the information claimed with the identifier is relevant in the context of the previous thought and question otherwise, enter a hyphen; The identifier will always be a field 'id' in the previous observation and you have to take its value. If multiple identifers are relevant, then separate them by comma.
        Action: the action to take, should be one of the [{tool_names}]
        Action Input: the input to the action. Do NOT wrap in code blocks or backticks.
        Observation: the result of the action
        Thought: you should write the next thought for line of action 
        ... (this Thought/Relevance/Action/Action Input/Observation/Thought can repeat N times)
        Thought: I have now collected sufficient evidences and information for the question from the available data sources. 
        Relevance: Relevant ids from previous observation. 
        <<<LLAMA3_3_EOS>>> """

        SUFFIX = """
        Question: {input}
        Thought:{agent_scratchpad}"""
        
    elif llm == "meta-llama/Llama-3.3-70B-Instruct":
        prepend_template2 = """[INSTRUCTION] 
        You have no prior knowledge. Your goal is to gather information and evidences for the question from different data sources namely NCBI Genes, UniProt, KEGG, PubMed, and NCBI ClinVar. Follow multiple iterations of thought/relevance/action/actioninput/observation (the solution-process). Each thought or argument prompts an action that, using the provided input, returns an observation. Based on that observation, a relevance identifier is stored, leading to the next thought. The thought doesn’t summarize the entire previous observation; it simply advances the logic. Thoughts should begin simple and progressively evolve into more complex and specific reasoning over iterations. If a chosen direction does not yield sufficient evidence, you are allowed to change its approach logically. For actions, only use one among <<tool_names>> tools in one iteration. Action inputs should remain concise and well-targeted to maximize the relevance and breadth of search results. You must perform a thorough exploration across availavle sources, ensuring every aspect of the question is adequately addressed. End the response with <<<LLAMA3_3_EOS>>>."""

        # react style prompting
        PREFIX = """You have access to the following tools: """
        TOOL_INSTRUCTION = """PubMed can only retrieve the list of papers and provide metadata information. To get the full text of a paper, use PubMedCentral.  For finding or summarizing proteins use UniProt, for pathways use KEGG, for genes use Genes and for mutations use ClinVAR. 
        For tools with retrieve functionality, always provide the search_string with apt keywords and make use of AND, OR operators for better results. Also, pass the model organism in the search_string if possible, to make the search narrow. For example: '(DLG1 OR CDC42) AND "Home Sapiens"'. For tools with summarize functionality, only give the identifier or list of identifiers as action input. If you find an identifier and its description relevant, and, if necessary, extract key information from the full page by using the summarize functions. Remember: For Summarize_PubMedCentral_Page, always give the 'pmcid' as the action input, else always use 'id' for other tools."""
        
        FORMAT_INSTRUCTIONS = """Use the following format always, beginning with the Thought:
        Thought: you should always think about what to do
        Relevance: if a previous observation exits, enter the identifer from it if the information claimed with the identifier is relevant in the context of the previous thought and question otherwise, enter a hyphen; The identifier will always be a field 'id' in the previous observation and you have to take its value. If multiple identifers are relevant, then separate them by comma.
        Action: the action to take, should be one of the [{tool_names}]
        Action Input: the input to the action
        Observation: the result of the action
        ... (this Thought/Relevance/Action/Action Input/Observation/Thought can repeat N times)
        Thought: I have now collected sufficient evidences and information for the question from the available data sources. 
        Relevance: relevant identifiers from previous observation.
        <<<LLAMA3_3_EOS>>> """

        SUFFIX = """[BEGIN REASONING]
        Question: {input}
        {agent_scratchpad}"""

    elif llm == "":
        prepend_template2= ""
        PREFIX = "" 
        TOOL_INSTRUCTION = "" 
        FORMAT_INSTRUCTIONS = ""
        SUFFIX = ""

    return prepend_template2, PREFIX, TOOL_INSTRUCTION, FORMAT_INSTRUCTIONS, SUFFIX


