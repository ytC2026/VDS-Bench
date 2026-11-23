# Reference functions from Yunkaidang/uncertainty/utils.py
import re

SYSTEM_MSG_MC = '\n    You are an AI assistant who will help me to answer the question based on the given picture.\n    Your output answer must be strictly one of the following: A, B or other capital letter given in the options.\n    We hope you strictly follow the following format: only output "A" or "B" or "Other letter".\n'

def construct_query_base_MC(question, choices):
    prompt = f'\n    {SYSTEM_MSG_MC}\n    {question}\n    {dictionary_to_text(choices)}\n    '
    return prompt

def dictionary_to_text(dictionary):
    sorted_keys = sorted(dictionary.keys())
    lines = []
    for key in sorted_keys:
        lines.append(f'{key}: {dictionary[key]}')
    return '\n'.join(lines)

def extract_option(response, question):
    options_pattern = '\\b([A-D]): ([^,]+)(?:,|$)'
    options = re.findall(options_pattern, question)
    for option in options:
        option_letter = option[0]
        option_text = option[1].strip()
        if option_text in response:
            return option_letter
    letter_pattern = '\\b([A-D])\\b'
    match = re.search(letter_pattern, response)
    if match:
        return match.group(1)
    return response.replace(' ', '').lower()
