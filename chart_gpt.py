import pandas as pd
import matplotlib.pyplot as plt
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
import re
from pprint import pprint
from typing import Tuple


import tiktoken

def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0301"):
    # copied from https://github.com/openai/openai-cookbook/blob/main/examples/How_to_format_inputs_to_ChatGPT_models.ipynb
    """Returns the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model == "gpt-3.5-turbo":
        print("Warning: gpt-3.5-turbo may change over time. Returning num tokens assuming gpt-3.5-turbo-0301.")
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0301")
    elif model == "gpt-4":
        print("Warning: gpt-4 may change over time. Returning num tokens assuming gpt-4-0314.")
        return num_tokens_from_messages(messages, model="gpt-4-0314")
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif model == "gpt-4-0314":
        tokens_per_message = 3
        tokens_per_name = 1
    else:
        raise NotImplementedError(f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens.""")
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens

def _sanitize(content:str) -> str:
    code_regex = r'```(?:python)?\n([\s\S]*)```'
    match = re.search(code_regex, content)
    code = match.group(1)
    return code

def _pandas_dtype_str(df: pd.DataFrame) -> str:
    return " " .join([f"{col}({dtype})" for col, dtype in df.dtypes.items()])


_SYSTEM_TEMPLATE = "You are an expert Python data analyst. You have been given a dataframe with the following columns: `{dtype_str}`"
_MAX_TOKENS = 4096

class ChartGPT:

    def __init__(self) -> None:
        self._llm = ChatOpenAI(model_name="gpt-3.5-turbo")
        self._history = []
        self._system_prompt = None

    def _clear_history(self) -> None:
        self._history = []
        self._system_prompt = None

    def inspect(self, df) -> None:
        self._system_prompt = SystemMessage(content = _SYSTEM_TEMPLATE.format(dtype_str=_pandas_dtype_str(df)))
        self._df = df

    def _get_response(self, prompt) -> Tuple[AIMessage, HumanMessage]:
        assert self._system_prompt is not None, "Inspect a dataframe first!"

        new_msg = HumanMessage(content=f"Give me the only python code and nothing else for the following. Assume I have the dataframe preloaded : {prompt}")
        # TODO: remove hardcoding and use token count
        messages = [self._system_prompt] + self._history[:-4] + [new_msg]
        pprint(messages)
        resp = self._llm(messages)
        return resp, new_msg
    
    def ask(self, prompt) -> None:
        ai_response, msg = self._get_response(prompt)
        self._history += [msg, ai_response]
        print(ai_response.content)
        code = _sanitize(ai_response.content)
        print(code)
        # TODO: try/ except plot
        exec(code, {'df': self._df})