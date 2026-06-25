import logging
import os
import re
import xml.etree.ElementTree as ET
import copy
import tiktoken 
from langchain_text_splitters import RecursiveJsonSplitter
from langchain.docstore.document import Document
from langchain_core.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
import json
import numpy as np
import re
import pandas as pd
import pickle

from typing import Optional

from src.human_IQ2 import initialize_models

def parse_input(s: str, user_question: Optional[str] = None, user_thought: Optional[str] = None,
    parse_thought: bool = False) -> dict:
    """
    Parse input into a dict with:
      - 'search_string' (always)
      - 'question'      (from payload or, if missing, from user_question)
      - 'thought'       (if present in payload, or if parse_thought=True then user_thought)

    
    Handles any quoting style for keys/values:
      • Bare string: 
          "DDL4 AND BRCA1"
      • Single‐quoted mini‐dict: 
          "'search_string': '...', 'question': '...'"
      • Double‐quoted JSON‐style: 
          '{"search_string": "...", "question": "..."}'
      • Mixed quoting (e.g. JSON keys + single-quoted value)
    """
    s = s.strip()
    result: dict = {}

    # 1) Try to extract any 'search_string' or 'question' pairs,
    #    where keys and values may each be in ' or " quotes.
              
    pattern = r"""['"](?P<key>search_string|question|thought)['"]\s*:\s*(['"])(?P<val>.*?)\2"""
    pairs = re.findall(pattern, s, flags=re.DOTALL)
    if pairs:
        for key, _quote, val in pairs:
            result[key] = val
        result = ensure_question(result, user_question)
        # ensure 'thought' if requested
        if parse_thought and "thought" not in result:
            result["thought"] = user_thought or ""
        return result

    # 2) No explicit pairs → the whole thing is the search string.
    result["search_string"] = s.strip("'\"")
    result = ensure_question(result, user_question)
    if parse_thought:
        result["thought"] = user_thought or ""
    return result



def ensure_question(result: dict, user_question: Optional[str]) -> dict:
    """
    Guarantee there's always a 'question' key.
    If it wasn't in the payload, use user_question (or empty string).
    """
    if "question" not in result:
        result["question"] = user_question or ""
    return result



# Function to convert XML to a dictionary
def xml_to_dict(element):
    """
    Recursively converts an XML element and its children into a dictionary.

    Parameters:
        element (xml.etree.ElementTree.Element): The XML element.

    Returns:
        dict: The converted dictionary representation of the XML.
    """
    result = {}
    # Process element's attributes
    for key, value in element.attrib.items():
        result[f"@{key}"] = value
    
    # Process element's children
    for child in element:
        child_dict = xml_to_dict(child)
        if child.tag not in result:
            result[child.tag] = child_dict
        else:
            if not isinstance(result[child.tag], list):
                result[child.tag] = [result[child.tag]]
            result[child.tag].append(child_dict)
    
    # Process element's text
    if element.text and element.text.strip():
        text = element.text.strip()
        if result:
            result["#text"] = text
        else:
            result = text
    
    return result

# Convert XML string to JSON
def xml_to_json(xml_content):
    """
    Converts XML content into a JSON string.

    Parameters:
        xml_content (str): The XML content as a string.

    Returns:
        str: The JSON representation of the XML.
    """
    try:
        root = ET.fromstring(xml_content)
        xml_dict = {root.tag: xml_to_dict(root)}
        return xml_dict
    except ET.ParseError as e:
        return f"Error parsing XML: {e}"
    
    


def split_list(lst, num_parts):
    return np.array_split(lst, num_parts)
    
def get_dict_by_key_value(lst, key, value):
    result = next((item for item in lst if item.get(key) == value), None)
    return result


def get_rel_info_by_llm(json_input: dict, question: str,  thought:str=""):

    llm = initialize_models.LLM
    #logging.info(f"The LLM used by the tool is : {llm.model_name}")

    # Refine Summarization
    SUMMARIZE_PROPMT_NO_THOUGHT = """Given the following JSON data and the question, summarize the information relevant to the question.
             
Question: {question}
            
JSON_data: {text}
            
Output:"""
    SUMMARIZE_PROMPT_WITH_THOUGHT = """You are given a JSON document, a thought, and a question.
Your job is to extract or summarize information **strictly** based on the JSON content and **only** as per the instruction in the thought.
Use the question only as additional context. Do not include any information not explicitly present in the JSON.

Thought: {thought}
Question: {question}
            
JSON_data: {text}
            
Output:"""

    REFINE_PROMPT = (
    "Your job is to produce a final summary\n"
    "We have provided an existing summary up to a certain point: {existing_answer}\n"
    "We have the opportunity to refine the existing summary"
    "(only if needed) with some more context below.\n"
    "------------\n"
    "{text}\n"
    "------------\n"
    "Given the additional information of the json data, refine the original summary"
    "If the additional context isn't useful, return the original summary."
    )    

    

    REFINE_PROMPT_WITH_THOUGHT = (
        "Your job is to produce a final summary based on information collected from the provided text with reagrds to the thought: {thought}\n"
        "We have provided an existing summary up to a certain point: {existing_answer} \n"
         "We have the opportunity to refine the existing summary"
        "(only if needed) with some more context below.\n"
        "------------\n"
        "{text}\n"
        "------------\n"
        "Given the additional information of the json data, refine the original summary"
        "If the additional context isn't useful, return the original summary."
    )
    
    SYNTHESIZE_PROMPT_WITH_THOUGHT = """You are given text and a specific thought guiding what information to extract.

    Thought: {thought}

    Using only the information in the text below, provide a clear, concise summary that **directly and specifically answers the thought**. Do not include any information not explicitly present in the text. Avoid repetition and unnecessary elaboration—focus strictly on what the text reveals about the thought.

    Text:
    ------------
    {text}
    ------------

    Output a synthesized, precise summary answering the thought:
    """
    SYNTHESIZE_PROMPT = """You are given text and a specific question guiding what information to extract.

    Question: {question}

    Using only the information in the text below, provide a clear, concise summary that **directly and specifically answers the question**. Do not include any information not explicitly present in the text. Avoid repetition and unnecessary elaboration—focus strictly on what the text reveals about the question.

    Text:
    ------------
    {text}
    ------------

    Output a synthesized, precise summary answering the question:
    """

    if thought != "":
       SUMMARIZE_PROPMT = SUMMARIZE_PROMPT_WITH_THOUGHT
    else:
        SUMMARIZE_PROPMT = SUMMARIZE_PROPMT_NO_THOUGHT
    json_string = str(copy.deepcopy(json_input))
    encoding = tiktoken.encoding_for_model("gpt-4o-mini")
    length = len(encoding.encode(json_string))
    if length > 40_000:
        "Given the additional information of the json data, refine the original summary"
        "If the additional context isn't useful, return the original summary."
        
        splitter = RecursiveJsonSplitter(max_chunk_size=1000)
        try:
            json_chunks = splitter.split_json(json_data=json_input)
        except:
            parsed = json.loads(json_input)
            json_chunks = splitter.split_json(json_data=parsed)
        temp_json_chunks = json_chunks.copy()
        for idx, chunk in enumerate(temp_json_chunks):
            if len(encoding.encode(str(chunk))) > 40_000:
               json_chunks.remove(chunk)

        split_docs = [
            Document(page_content=str(t)) for t in json_chunks[: len(json_chunks)]
        ]
        # new_split_docs = [{"text": doc.page_content} for doc in split_docs]


        prompt_template = SUMMARIZE_PROPMT
        
        if thought!="":
            prompt = PromptTemplate(
                input_variables=["question", "text", "thought"], template=prompt_template
            )
            prompt = prompt.partial(thought=thought)
            refine_prompt = PromptTemplate.from_template(REFINE_PROMPT_WITH_THOUGHT).partial(thought=thought)
        else:
            prompt = PromptTemplate(
                input_variables=["question", "text"], template=prompt_template
            )
            refine_prompt = PromptTemplate.from_template(REFINE_PROMPT)
        summarize_prompt = prompt.partial(question=question)
        # summarize_prompt = prompt.format(input_query=INPUT_QUERY, text = split_docs[0].page_content)
        chain = load_summarize_chain(
            llm=llm,
            chain_type="refine",
            question_prompt=summarize_prompt,
            refine_prompt=refine_prompt,
            return_intermediate_steps=False,
            input_key="text",
            output_key="output_text",
        )
        result = chain.invoke({"text": split_docs}, return_only_outputs=True)
        summary = result["output_text"]
        if thought!="":
            synthesize_prompt = SYNTHESIZE_PROMPT_WITH_THOUGHT.format(thought=thought, text=summary)
            final_summary = llm.invoke(synthesize_prompt)
            final_summary = final_summary.content
        else:
            synthesize_prompt = SYNTHESIZE_PROMPT.format(question=question, text=summary)
            final_summary = llm.invoke(synthesize_prompt)
            final_summary = final_summary.content
        
        return final_summary

    else:
        if thought!="":
            final_prompt = SUMMARIZE_PROPMT.format(thought=thought, question=question, text=json.dumps(json_input))
        else:
            final_prompt = SUMMARIZE_PROPMT.format(
                question=question, text=json.dumps(json_input)
            )
        messages = [
            {
                "role": "system",
                "content": "You are an assistant that extract information from the structured data.",
            },
            {"role": "user", "content": final_prompt},
        ]
        response = llm.invoke(messages)
        response_text = response.content
        
        return response_text 


def safe_get(d, keys, default=""):
    """
    Safely navigate a nested dictionary.
    :param d: The dictionary to search.
    :param keys: A list of keys to navigate in order.
    :param default: The default value to return if any key is missing.
    :return: The value found or the default.
    """
    for key in keys:
        if not isinstance(d, dict) or key not in d:
            return default
        d = d.get(key)
    return d if d is not None else default

def load_GT(file_path: str) -> pd.DataFrame:
    # Read the raw text
    with open(file_path, "r") as f:
        content = f.read()

    # Manually split into JSON objects
    # (Assuming objects are just stacked without commas/newlines)
    objects = content.strip().split("}\n{")
    objects = [o + "}" if not o.endswith("}") else o for o in objects]  # Fix ending }
    objects = [
        "{" + o if not o.startswith("{") else o for o in objects
    ]  # Fix starting {

    # Parse each JSON object
    data = [json.loads(obj) for obj in objects]

    # Convert to DataFrame
    gt = pd.DataFrame(data)
    return gt

def load_model_output(file_path: str) -> pd.DataFrame:
    output_rows = []
    with open(file_path, "rb") as in_f:
        while True:
            try:
                output_rows.append(pickle.load(in_f))
            except EOFError:
                break

    output_df = pd.concat(output_rows, ignore_index=True)
    return output_df


def is_plain_text(s):
    """
    Heuristically checks if a string is plain text (not JSON or JSON-like).
    
    Args:
        s (str): The string to check.
    
    Returns:
        bool: True if it's plain text, False if it looks like JSON or JSON-like.
    """
    s = s.strip()
    
    # Empty string is considered plain
    if not s:
        return True

    # Check if it starts with common JSON structures
    if s.startswith('{') or s.startswith('['):
        return False

    # Check for presence of key-value like patterns: "key": "value"
    if re.search(r'"\s*[\w\- ]+\s*"\s*:\s*', s):
        return False

    # Check for presence of curly braces or square brackets inside
    if '{' in s or '}' in s or '[' in s or ']' in s:
        return False

    # Check for JSON-style escaped quotes (\"), which usually don't appear in normal text
    if r'\"' in s:
        return False

    return True



# def extract_entries_by_pubmed_id(text, target_ids):
#     """
#     Extract entire entries where the PubMed ID in relevant references matches target_ids
    
#     Args:
#         text (str): The input text containing entries
#         target_ids (list): List of PubMed IDs to match
    
#     Returns:
#         str: String containing all matching entries in their original format
#     """
    
#     # Remove the surrounding quotes if present
#     clean_text = text.strip("'\"")
    
#     # Split the text by **Id**: to get individual entries
#     # We need to be careful to preserve the **Id**: marker for non-first entries
#     parts = re.split(r'(\*\*Id\*\*:)', clean_text)
    
#     # Reconstruct entries properly
#     entries = []
#     if len(parts) > 1:
#         # First part before any **Id**: (usually empty or header)
#         if parts[0].strip():
#             entries.append(parts[0])
        
#         # Combine **Id**: markers with their content
#         for i in range(1, len(parts), 2):
#             if i + 1 < len(parts):
#                 entry = parts[i] + parts[i + 1]  # **Id**: + content
#                 entries.append(entry)
#     else:
#         entries = [clean_text]  # No **Id**: markers found
    
#     matching_entries = []
    
#     for entry in entries:
#         # Skip empty entries
#         if not entry.strip() or '**Id**:' not in entry:
#             continue
            
#         # Look for pubmed_id in this entry
#         pubmed_match = re.search(r'"pubmed_id":\s*"([^"]+)"', entry)
        
#         if pubmed_match:
#             pubmed_id = pubmed_match.group(1)
#             print(f"Debug: Found entry with pubmed_id: {pubmed_id}")
            
#             # Check if this pubmed_id matches any target_id
#             if pubmed_id in target_ids:
#                 print(f"Debug: Match found for pubmed_id: {pubmed_id}")
#                 matching_entries.append(entry.strip())
    
#     # Join all matching entries with double newlines for separation
#     result = '\n\n'.join(matching_entries)
    
#     print(f"Debug: Total matching entries: {len(matching_entries)}")
#     return result


def extract_entries_by_pubmed_id(text_tuple, target_pubmed_ids):
    raw = text_tuple[0] if not is_plain_text(text_tuple[0]) else text_tuple[1]  # Extract the JSON string part

    # Match each entry in the outer list
    entry_pattern = re.compile(r'{\s*"id":\s*"(?P<id>[^"]+)",\s*"summary":\s*"(?P<summary>.*?)",\s*"references":\s*(?P<references>\[.*?\])\s*}', re.DOTALL)

    matching_entries = []

    for match in entry_pattern.finditer(raw):
        entry_id = match.group("id")
        summary = match.group("summary")
        references = match.group("references")

        # Search for target pubmed_id in the references JSON string
        pubmed_matches = re.findall(r'"pubmed_id":\s*"(\d+)"', references)
        if any(pubmed_id in target_pubmed_ids for pubmed_id in pubmed_matches):
            formatted = (
                f"**Id**: {entry_id}\n"
                f"**Summary:**\n{summary.strip()}\n"
                f"**References:**\n{references.strip()}"
            )
            matching_entries.append(formatted)

    return "\n\n".join(matching_entries)


# def extract_entries_by_ids(text, target_ids):
#     # Regex to find all entries
#     # Each entry ends right before the next "**Id**" or at the end of the string
#     pattern = re.compile(
#         r'\*\*Id\*\*: (?P<id>PMC\d+)\*\*Summary:\*\*\n(?P<summary>.*?)(?=\*\*Id\*\*: PMC\d+|$)',
#         re.DOTALL
#     )
    
#     extracted = []
#     for match in pattern.finditer(text):
#         entry_id = match.group('id')
#         if entry_id in target_ids:
#             full_text = match.group(0).strip()

#             # Ensure "Relevant References" is preserved even if empty
#             if '"references":' not in full_text:
#                 full_text += "\nreferences:"

#             extracted.append(full_text)

#     return "\n\n".join(extracted)

import re

def extract_entries_by_ids(text_tuple, target_ids):
    raw = text_tuple[0] if text_tuple[0].strip() else text_tuple[1]

    # Match individual JSON-like objects with id, summary, references
    entry_pattern = re.compile(
        r'{\s*"id":\s*"(?P<id>[^"]+)",\s*'
        r'"summary":\s*"(?P<summary>.*?)",\s*'
        r'"references":\s*(?P<references>\[[^\]]*?\])\s*}',
        re.DOTALL
    )

    extracted = []
    for match in entry_pattern.finditer(raw):
        entry_id = match.group("id")
        if entry_id in target_ids:
            summary = match.group("summary").strip()
            references = match.group("references").strip()
            formatted = (
                f"**Id**: {entry_id}\n"
                f"**Summary:**\n{summary}\n"
                f"**References:**\n{references}"
            )
            extracted.append(formatted)

    return "\n\n".join(extracted)



def filter_json_by_ids(json_input, id_list):
    """
    Filters JSON string or tuple of strings by matching IDs.

    Args:
        json_input (str or tuple): JSON string or a tuple containing JSON string(s).
        id_list (list): List of string IDs to retain.

    Returns:
        list: Filtered list of dictionaries.
    """
    # If input is a tuple, extract the first non-empty string
    if isinstance(json_input, tuple):
        json_input = next((x for x in json_input if isinstance(x, str) and x.strip()), '')

    if not isinstance(json_input, str):
        raise TypeError(f"Expected str or tuple of str, got {type(json_input)}")

    try:
        data = json.loads(json_input)
        if not isinstance(data, list):
            raise ValueError("Expected a list of dictionaries inside the JSON.")
        return [entry for entry in data if entry.get("id") in id_list]
    except Exception as e:
        print(f"[ERROR] Invalid JSON input: {e}")
        return json_input
