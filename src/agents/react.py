import json
import logging
from typing import Any, Dict, List, Union
import copy
import pydantic_core
import regex as re

from src.agents.template_construction import TemplateConstruction
from src.models import initialize_models
from src.prompts.frame_prompt import format_prompt_per_model
from src.prompts.prompts import get_prompt_by_llm
from src.tools.fetch_names import (
    dedupe_article_ids,
    fetch_gene_names,
    fetch_mutations_names,
    fetch_paper_names,
    fetch_pathways_names,
    fetch_protein_names,
)
import tiktoken
from langchain.agents import (
    AgentExecutor,
    AgentOutputParser,
    BaseSingleActionAgent,
)
from langchain.tools import StructuredTool
from langchain.callbacks.manager import Callbacks
from langchain_community.callbacks.manager import get_openai_callback
from langchain.chains.llm import LLMChain
from langchain.prompts import StringPromptTemplate
from langchain.schema import (
    AgentAction,
    AgentFinish,
    OutputParserException,
)
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
from src.utils import extract_entries_by_ids, extract_entries_by_pubmed_id, filter_json_by_ids, is_plain_text, parse_input
from langsmith import traceable


class CustomPromptTemplate(StringPromptTemplate):
    template: str
    tools: List[StructuredTool]

    def format(self, **kwargs) -> str:
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        # defining the intermediate step to gather observations and thoughts to guess the final reply
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\n\nThought: "

        kwargs["agent_scratchpad"] = thoughts
        kwargs["tools"] = "\n".join(
            [f"{tool.name}: {tool.description}" for tool in self.tools]
        )
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        if len(intermediate_steps) != 0:
            log_last_action = intermediate_steps[-1]
            log_last_observation = json.dumps(log_last_action[1], indent=4)
            logging.info(
                f"Logging LLM thougths and outputs: \nThought: {log_last_action[0].log}\n     \n Observation: {str(log_last_action[1])} \n\n\n\n"
            )
        else:
            logging.info(f"Initial prompt for the LLM: {str(self.template)}")

        return self.template.format(**kwargs) #selective_format(self.template, kwargs)  


class LLMSingleActionAgentCustom(BaseSingleActionAgent):
    llm_chain: LLMChain
    output_parser: AgentOutputParser
    stop: List[str]

    @property
    def input_keys(self) -> List[str]:
        return list(set(self.llm_chain.input_keys) - {"intermediate_steps"})

    def dict(self, **kwargs: Any) -> Dict:
        """Return dictionary representation of agent."""
        _dict = super().dict()
        del _dict["output_parser"]
        return _dict

    def plan(
        self,
        intermediate_steps,
        callbacks: Callbacks = None,
        **kwargs: Any,
    ):
        """Given input, decided what to do.

        Args:
            intermediate_steps: Steps the LLM has taken to date,
                along with observations
            callbacks: Callbacks to run.
            **kwargs: User inputs.

        Returns:
            Action specifying what tool to use.
        """
        output = self.llm_chain.run(
            intermediate_steps=intermediate_steps,
            stop=self.stop,
            callbacks=callbacks,
            **kwargs,
        )
        user_question = kwargs.pop("input")
        return self.output_parser.parse(output, user_question)

    async def aplan(
        self,
        intermediate_steps,
        callbacks: Callbacks = None,
        **kwargs,
    ) -> Union[AgentAction, AgentFinish]:
        """Given input, decided what to do.

        Args:
            intermediate_steps: Steps the LLM has taken to date,
                along with observations
            callbacks: Callbacks to run.
            **kwargs: User inputs.
        Who was the longest-serving president in U.S. history? # longest-serving president in U.S. history

        Returns:
            Action specifying what tool to use.
        """
        output = await self.llm_chain.arun(
            intermediate_steps=intermediate_steps,
            stop=self.stop,
            callbacks=callbacks,
            **kwargs,
        )
        user_question = kwargs.pop("input")
        return self.output_parser.parse(output, user_question)

    def tool_run_logging_kwargs(self):
        return {
            "llm_prefix": "",
            "observation_prefix": "" if len(self.stop) == 0 else self.stop[0],
        }


class CustomOutputParser(AgentOutputParser):
    # defining the parse method with multiple arguments to return answers using the Agent performing tasks
    def parse(self, llm_output: str, user_question) -> Union[AgentAction, AgentFinish]:
        if "<<<LLAMA3_3_EOS>>>" in llm_output:
            # ----------------- Preprocess final answer ------------
            """
            #### preprocess(llm_output.split("Final Report:")[-1].strip())
            """
            logging.info(llm_output)
            print("################################# \nI am in the last step to produce the final answer", llm_output)
            return AgentFinish(
                return_values={"output": llm_output.split("Final Report:")[-1].strip()},
                log=llm_output,
            )

        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)

        # Returning AgentFinish object that gets the input/question from the user and the AgentAction starts running and the loops keeps repeating
        if not match:
            raise OutputParserException(f"Could not parse LLM output: `{llm_output}`")

        action = match.group(1).strip()
        action_input = match.group(2)


        if action in [
            "Retrieve_ClinVar",
            "Summarize_CliVar_Page",
            "Retrieve_Gene",
            "Summarize_Gene_Page",
            "Retrieve_KEGG",
            "Summarize_KEGG_Page",
            "Retrieve_PubMed",
            "Summarize_PubMedCentral_Page",
            "Retrieve_UniProt_Page",
            "Summarize_UniProt_Page",
            "Exchange_PMC_PMID",
            "Fetch_Info_Any_Source",
        ]:

            extracted_thought = ""
            parse_thought = False
            if action == "Fetch_Info_Any_Source":
                parse_thought = True
                thought_match = re.search(
                    r"Thought:\s*'(.*?)'\s*Relevance:", llm_output, re.DOTALL
                )

                if thought_match:
                    extracted_thought = thought_match.group(1).strip()
                else:
                    # Assume everything before "Relevance:" is the thought
                    relevance_match = re.search(
                        r"(.*?)Relevance:", llm_output, re.DOTALL
                    )
                    if relevance_match:
                        extracted_thought = (
                            relevance_match.group(1).strip().strip("'")
                        )
                
            cleaned_action_input = parse_input(
                s=action_input, 
                user_question=user_question, 
                user_thought=extracted_thought, 
                parse_thought=parse_thought
            )

            return AgentAction(
                tool=action, tool_input=cleaned_action_input, log=llm_output
            )
        else:
            raise Exception("No such action available. ")


class CustomReactPipeline:
    def __init__(self, prompts_style, num_few_shot_example, MODEL_NAME, main_ip, api_key, dynamic, DPP):
        self.prompts_style = prompts_style
        self.num_few_shot_example = num_few_shot_example
        self.dynamic = dynamic
        self.DPP = DPP
        self.few_shot_library = "all_datasets/my_dataset/few_shots_with_action_seq.json"
        self.selected_few_shot_questions = []  # Store selected questions

        initialize_models.init(model_name=MODEL_NAME, main_ip=main_ip, api_key=api_key)

        self.llm = initialize_models.LLM    

    def _build_few_shots(self, question: str) -> tuple[str, List[str]]:
        logging.info("Few-shot construction for question: %s", question)
        logging.info("Flags -> DPP: %s | Dynamic (diversity): %s", self.DPP, self.dynamic)

        blocks = []
        selected_questions = []
        if self.DPP:
            blocks.append("### DPP-Selected Examples")
            template, selected_questions = TemplateConstruction(question=question, dataset=self.few_shot_library).few_shot_with_dpp()
            blocks.append(template)
        elif self.dynamic:
            blocks.append("### Diversity-Selected Examples")
            template, selected_questions = TemplateConstruction(question=question, dataset=self.few_shot_library, cos=True).full_shot_with_diversity()
            blocks.append(template)
        else:
            blocks.append("### Static Examples")
            template, selected_questions = TemplateConstruction(question=question, dataset=self.few_shot_library).static_prompt_construction()
            blocks.append(template)

        return "\n\n".join(blocks), selected_questions

    def get_tools(self):

        tools = [
            retrieve_clinvar_entries_tool,
            summarize_clinvar_page_tool,
            retrieve_gene_entries_tool,
            summarize_gene_page_tool,
            retrieve_kegg_pathways_tool,
            summarize_kegg_pathways_tool,
            retrieve_pubmed_papers_tool,
            summarize_pubmed_papers_tool,
            retrieve_uniprot_entries_tool,
            summarize_uniprot_entries_tool,
            exchange_pmc_pmid_tool,
            fetch_info_any_source_tool,
        ]

        return tools

    def get_prompt(self, question, prompt_type, tools):

        if self.num_few_shot_example > 0:
            few_shot_examples, self.selected_few_shot_questions = self._build_few_shots(question)
        else:
            few_shot_examples = ""
            self.selected_few_shot_questions = []
        
        if self.llm.model_name == "meta-llama/Llama-3.3-70B-Instruct" or "gpt" in self.llm.model_name: #isinstance(self.llm, ChatAnthropic) or isinstance(self.llm, ChatGoogleGenerativeAI): #or self.llm.model_name == "meta-llama/Llama-3.3-70B-Instruct" or "gpt" in self.llm.model_name:
            prepend_template2, PREFIX, TOOL_INSTRUCTION, FORMAT_INSTRUCTIONS, SUFFIX = (
                get_prompt_by_llm(self.llm.model_name)
            )
            # Framing the prompt
            tool_strings = "\n ".join(
                [f"{tool.name}: {tool.description}" for tool in tools]
            )
            tool_names = ", ".join([tool.name for tool in tools])
            format_instructions = FORMAT_INSTRUCTIONS.format(tool_names=tool_names)
            react_template = "\n\n".join(
                [PREFIX, tool_strings, TOOL_INSTRUCTION, format_instructions, SUFFIX]
            )

            complete_prompt = (
                f"{prepend_template2}{few_shot_examples}\n{react_template.strip()}"
            )
            
            
        else:
            complete_prompt = format_prompt_per_model(model_name=self.llm.model)

        prompt = CustomPromptTemplate(
            template=complete_prompt,  # complete_prompt.strip("\n"),
            tools=tools,
            input_variables=["input", "intermediate_steps"],
        )
        # prompt = CustomChatPromptTemplate(
        #     messages=[
        #         SystemMessage(content="{text}"),
        #         HumanMessage(content="{text}")
        #     ],
        #     template=complete_prompt,
        #     tools=tools,
        #     input_variables=["input", "intermediate_steps"],
        # )

        return prompt

    def answer_ques(self, question):
        if self.num_few_shot_example > 0:
            return ("", 0, 0)
        
        final_answer_gen_prompt = f"""Your task is to generate a comprehensive report for the given question.
        Question: {question}
        Report: """

        with get_openai_callback() as cb_3:
            report = self.llm.invoke(final_answer_gen_prompt)
            wot_completion_tokens = cb_3.completion_tokens
            wot_total_tokens = cb_3.total_tokens
            
        return (
            report.content, wot_completion_tokens, wot_total_tokens)

    def clean_observation(self, relevance, prev_observation):
        revised_prev_obs = ""
        relevant_ids = relevance.split(",")
        relevant_ids = [id.strip() for id in relevant_ids]

        if '"summary":' in str(prev_observation):
            for item in relevant_ids:
                item = item.strip()
                if item in str(prev_observation):
                    relevant_item = extract_entries_by_ids(prev_observation, item)
                    if relevant_item == None or relevant_item == "":
                        relevant_item = extract_entries_by_pubmed_id(prev_observation, [item])
                    revised_prev_obs += relevant_item + "\n"
        else:
            revised_prev_obs = str(filter_json_by_ids(prev_observation, relevant_ids))
            
        prev_observation = revised_prev_obs

        return revised_prev_obs

    def filter_ref_identifiers(self, ids, dict_of_identifiers):
        ids = ids.split(",")
        ids = list(set(ids))
        existing_ids_in_dict = {
            i for values in dict_of_identifiers.values() for i in values
        }
        for id in ids:
            id = id.strip()
            if id in existing_ids_in_dict:
                continue
            if str(id).startswith("map") and "pathways" in dict_of_identifiers:
                dict_of_identifiers["pathways"].append(id)

            elif (
                str(id).isnumeric()
                and len(list(str(id))) <= 6
                and "genes" in dict_of_identifiers
            ):
                dict_of_identifiers["genes"].append(id)

            elif str(id).startswith("VCV") and "mutations" in dict_of_identifiers:
                dict_of_identifiers["mutations"].append(id)

            elif (
                (str(id).startswith("PMC") or str(id).isnumeric())
                and len(list(str(id))) > 6
                and "papers" in dict_of_identifiers
            ):
                dict_of_identifiers["papers"].append(id)

            elif (
                re.fullmatch(r"[A-Za-z0-9]+", id)
                and any(c.isdigit() for c in id)
                and any(c.isalpha() for c in id)
                and "proteins" in dict_of_identifiers
            ):
                dict_of_identifiers["proteins"].append(id)

            else:
                print("Doesnt fit in any of the categories")

        dict_of_identifiers["papers"] = dedupe_article_ids(
            dict_of_identifiers["papers"]
        )
        return dict_of_identifiers

    def preprocess_thought_process(self, formatted_thought_process):
        clean_thought_process = []
        dict_of_identifiers = {
            "genes": [],
            "proteins": [],
            "mutations": [],
            "pathways": [],
            "papers": [],
        }
        previous_iteration = {}
        current_iteration = {}
        revised_pre_observation = {}

        first = 0
        second = 1
        while first < (len(formatted_thought_process) - 1):
            previous_iteration = formatted_thought_process[first]
            current_iteration = formatted_thought_process[second]
            revised_pre_observation = copy.deepcopy(previous_iteration)
            relevant_ids = current_iteration["Relevance"]
            prev_obsevation = previous_iteration["Observation"]
            if relevant_ids == None or relevant_ids == "-":
                revised_pre_observation["Observation"] = prev_obsevation if (is_plain_text(prev_obsevation[0]) and is_plain_text(prev_obsevation[1])) else ""
            else:
                dict_of_identifiers = self.filter_ref_identifiers(
                    relevant_ids, dict_of_identifiers
                )
                revised_pre_observation["Observation"] = self.clean_observation(
                    relevance=relevant_ids, prev_observation=prev_obsevation
                )
            first += 1
            second += 1
            clean_thought_process.append(revised_pre_observation)
        clean_thought_process.append(current_iteration)
        return clean_thought_process, dict_of_identifiers

    def format_reference_list(self, dict_of_identifiers):
        formatted_references = []
        keys = ["genes", "proteins", "mutations", "pathways", "papers"]
        for key in keys:
            if key == "genes" and len(dict_of_identifiers["genes"]) > 0:
                list_of_genes = dict_of_identifiers["genes"]
                formatted_references.extend(fetch_gene_names(list_of_genes))
            elif key == "proteins" and len(dict_of_identifiers["proteins"]) > 0:
                list_of_proteins = dict_of_identifiers["proteins"]
                formatted_references.extend(fetch_protein_names(list_of_proteins))
            elif key == "mutations" and len(dict_of_identifiers["mutations"]) > 0:
                list_of_mutations = dict_of_identifiers["mutations"]
                formatted_references.extend(fetch_mutations_names(list_of_mutations))
            elif key == "pathways" and len(dict_of_identifiers["pathways"]) > 0:
                list_of_pathways = dict_of_identifiers["pathways"]
                formatted_references.extend(fetch_pathways_names(list_of_pathways))
            elif key == "papers" and len(dict_of_identifiers["papers"]) > 0:
                list_of_papers = dict_of_identifiers["papers"]
                formatted_references.extend(fetch_paper_names(list_of_papers))
            else:
                print("the key doesnt exist")
        references = "\n"
        for i in range(len(formatted_references)):
            references += f"[{i+1}]. " + formatted_references[i] + ".\n"
        return references + "\n"

    def formatiing_thought_process(self, thoughtprocess):
        formatted_thought_process = []
        for item in thoughtprocess:
            if isinstance(item, tuple):
                temp_thought = item[0].log
                current_thought = {
                    "Thought": "",
                    "Relevance": "",
                    "Action": "",
                    "ActionInput": "",
                    "Observation": "",
                }

                # Identify if "Thought:" is explicitly mentioned
                thought_match = re.search(
                    r"Thought:\s*'(.*?)'\s*Relevance:", temp_thought, re.DOTALL
                )

                if thought_match:
                    current_thought["Thought"] = thought_match.group(1).strip()
                else:
                    # Assume everything before "Relevance:" is the thought
                    relevance_match = re.search(
                        r"(.*?)Relevance:", temp_thought, re.DOTALL
                    )
                    if relevance_match:
                        current_thought["Thought"] = (
                            relevance_match.group(1).strip().strip("'")
                        )

                # Extract other fields
                relevance_match = re.search(
                    r"Relevance:\s*(.*?)(?:\n|(?=Action:|<<<LLAMA3_3_EOS>>>))",
                    temp_thought,
                    re.DOTALL,
                )
                if relevance_match:
                    current_thought["Relevance"] = relevance_match.group(1).strip()

                action_match = re.search(r"Action:\s*(\S+)", temp_thought)
                if action_match:
                    current_thought["Action"] = action_match.group(1)

                current_thought["ActionInput"] = item[0].tool_input
                # action_input_match = re.search(r'Action Input:\s*"([^"]+)"', temp_thought, re.DOTALL)
                # if action_input_match:
                #     try:
                #         current_thought["ActionInput"] = json.loads(action_input_match.group(1))
                #     except (SyntaxError, ValueError):
                #         current_thought["ActionInput"] = action_input_match.group(1).strip()
                if current_thought["ActionInput"] == "Invalid or incomplete response":
                    continue
                else:
                    current_thought["ActionInput"].pop("question", None)
                current_thought["Observation"] = item[1]

                formatted_thought_process.append(current_thought)
            elif isinstance(item, str):
                try:
                    eos_match = re.search(r'"<<<LLAMA3_3_EOS>>>"', item)
                except TypeError as e:
                    print("EOS token not found. Printing the item")
                    print(item)
                thought_match = re.search(
                    r"^(.*?)(?=\nRelevance:)", item, re.DOTALL
                )  # or item
                relevance_match = re.search(r"Relevance:\s*(.*)", item)

                # Create the dictionary
                current_thought = {
                    "Thought": (
                        thought_match.group(1).strip() if thought_match else item
                    ),
                    "Relevance": (
                        relevance_match.group(1).strip() if relevance_match else None
                    ),
                    "<<<LLAMA3_3_EOS>>>": eos_match.group(0) if eos_match else None,
                }
                formatted_thought_process.append(current_thought)
        return formatted_thought_process
    
    def final_answer_with_preprocessing_generation(self, question, formatted_thought_process):

        clean_thought_process, dict_of_identifiers = self.preprocess_thought_process(
            formatted_thought_process
        )

        formatted_references = self.format_reference_list(
            dict_of_identifiers=dict_of_identifiers
        )

        
        encoding = tiktoken.encoding_for_model('gpt-3.5-turbo')
        formatted_TP_length = len(encoding.encode(str(formatted_thought_process)))
        clean_TP_length = len(encoding.encode(str(clean_thought_process)))
        

        logging.info(
            f"Length before preprocessing was {len(str(formatted_thought_process))}, and after preprocessing became {len(str(clean_thought_process))}. Total reduction of {len(str(formatted_thought_process)) - len(str(clean_thought_process))} characters"
        )

        final_answer_gen_prompt = f"""You have no knowlede about the world. Your task is to generate a comprehensive report for the given question. As a foundation for report generation, you are provided with multiple iterations of thought/relevance/action/actioninput/observation (also called as solution process). This mechanism illustrates how a question is thought through and navigated via different sources to collect evidences important to answer the question. You are also provided with the list of references that were marked as relevant in the solution process. You are required to use the index of the reference from the reference list as in-text citations. For example : 'In [3] the authors incorporate multiple modalities other than text into the encoder-decoder framework.' where [3] marks the index for reference in the reference list, and the information corresponding to it can be found in the solution process.
        
        The generated report is written in academic style of writing and framing all the collected information in a logical sense to answer the question.
        
        Question: {question}
        Solution Process: {json.dumps(clean_thought_process)}
        References: {formatted_references}  
        Report: """

        with get_openai_callback() as cb_1:
            report = self.llm.invoke(final_answer_gen_prompt)
            wtp_completion_tokens = cb_1.completion_tokens
            wtp_total_tokens = cb_1.total_tokens
            

        return (
            clean_thought_process,
            report.content,
            formatted_references,
            dict_of_identifiers,
            wtp_completion_tokens,
            wtp_total_tokens,
            formatted_TP_length,
            clean_TP_length,
        )

    def get_tools_used_count(self, thought_process):
        total_tools = []
        for step in thought_process["intermediate_steps"]:
            total_tools.append(step[0].tool)

        return len(total_tools), len(set(total_tools)), list(set(total_tools)), total_tools

    @traceable
    def execute_agent(self, input, prompt_type):
        logging.info(f"Question: {input}\n")
        tools = self.get_tools()
        prompt = self.get_prompt(input, prompt_type, tools)
        llm_chain = LLMChain(llm=self.llm, prompt=prompt)

        tool_names = [tool.name for tool in tools]
        output_parser = CustomOutputParser()

        agent = LLMSingleActionAgentCustom(
            llm_chain=llm_chain,
            output_parser=output_parser,
            stop=["\nObservation:"],
            allowed_tools=tool_names,
        )

        agent_executor = AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=tools,
            verbose=True,
            handle_parsing_errors=True,
            return_intermediate_steps=True,
        )

        formatted_thought_process = ""

        output = {}
        tries = 3
        with get_openai_callback() as cb:
            while tries > 0:
                try:
                    output = agent_executor.invoke(input)
                    break
                except pydantic_core._pydantic_core.ValidationError as e:
                    print(f"Validation error: {e}")
                    tries -= 1
                    if tries == 0:
                        raise  # re-raise after final try
            total_thought_process_completion_tokens = cb.completion_tokens
            total_thought_process_total_tokens = cb.total_tokens    

        print("total_thought_process_completion_tokens: ", total_thought_process_completion_tokens, "\n",
              "total_thought_process_total_tokens: ", total_thought_process_total_tokens, "\n",)
        
        print("Intermediate Steps: ", True if output.get("intermediate_steps") else False, len(output.get("intermediate_steps")),"\n")
        # collection of the thought process
        if (
            "intermediate_steps" in output
            and len(output.get("intermediate_steps", [])) > 0
        ):
            thoughtprocess = output["intermediate_steps"]

            if type(output.get("intermediate_steps")[-1]) != str and isinstance(
                (output.get("intermediate_steps")[-1][0]), AgentAction
            ):
                thoughtprocess.append(output["output"])

        else:
            thoughtprocess = []
            raise Exception(
                "The output does not contain any intermediate steps. Please check the output."
            )
            
        formatted_thought_process = self.formatiing_thought_process(thoughtprocess)
        

        clean_thought_process, wtp_answer, wtp_references, dict_of_references, wtp_completion_tokens, wtp_total_tokens, formatted_TP_length, clean_TP_length = self.final_answer_with_preprocessing_generation(
            question=input, formatted_thought_process=formatted_thought_process
        )

        logging.info(f"Final Report: \n {wtp_answer}")

        wtp_thought_process = clean_thought_process
        wtp_answer = wtp_answer.split("References:")[0]
        return (

            wtp_thought_process,
            wtp_answer,
            wtp_references
        )
