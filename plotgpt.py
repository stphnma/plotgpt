import pandas as pd
from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
import re
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
        print(
            "Warning: gpt-3.5-turbo may change over time. Returning num tokens assuming gpt-3.5-turbo-0301."
        )
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0301")
    elif model == "gpt-4":
        print(
            "Warning: gpt-4 may change over time. Returning num tokens assuming gpt-4-0314."
        )
        return num_tokens_from_messages(messages, model="gpt-4-0314")
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = (
            4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        )
    elif model == "gpt-4-0314":
        tokens_per_message = 3
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
        )
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        num_tokens += len(encoding.encode(message.content))
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens


def _parse_code(content: str) -> str:
    code_regex = r"```(?:python)?\n([\s\S]*)```"
    match = re.search(code_regex, content)
    code = match.group(1)
    return code


def _pandas_dtype_str(df: pd.DataFrame) -> str:
    return " ".join([f"{col}({dtype})" for col, dtype in df.dtypes.items()])


_SYSTEM_TEMPLATE = "You are an expert Python data analyst. You have been given a dataframe with the following columns: `{dtype_str}`"
_MAX_TOKENS = 4096
_MAX_HISTORY_CONTEXT = 50
_MODEL = "gpt-3.5-turbo"


class PlotGPT:
    def __init__(self, show_code: bool = True) -> None:
        self._llm = ChatOpenAI(model_name=_MODEL)
        self._history = []
        self._system_prompt = None
        self._show_code = show_code

    def _clear_history(self) -> None:
        self._history = []
        self._system_prompt = None

    def inspect(self, df: pd.DataFrame) -> None:
        self._clear_history()
        self._system_prompt = SystemMessage(
            content=_SYSTEM_TEMPLATE.format(dtype_str=_pandas_dtype_str(df))
        )
        self._df = df

    def _construct_messages(self, new_msg: HumanMessage):
        num_remaining_tokens = _MAX_TOKENS - num_tokens_from_messages(
            [self._system_prompt, new_msg], model=_MODEL
        )
        if num_remaining_tokens < 0:
            raise ValueError(f"prompt is too long for {_MAX_TOKENS} limit!")

        history = self._history[-1 * _MAX_HISTORY_CONTEXT :]

        while num_tokens_from_messages(history) > num_remaining_tokens:
            history = history[2:]

        return [self._system_prompt] + history + [new_msg]

    def _get_response(self, prompt) -> Tuple[AIMessage, HumanMessage]:
        assert self._system_prompt is not None, "Inspect a dataframe first!"

        new_msg = HumanMessage(
            content=f"Give me the only python code and nothing else for the following. Only use matplotlib and seaborn. Assume I have the dataframe preloaded as `df`: {prompt}"
        )
        messages = self._construct_messages(new_msg)
        resp = self._llm(messages)
        return resp, new_msg

    def ask(self, prompt) -> None:
        assert self._system_prompt is not None, "Need to inspect a dataframe first!"

        ai_response, msg = self._get_response(prompt)
        self._history += [msg, ai_response]
        code = _parse_code(ai_response.content)
        if self._show_code:
            print(code)
        # TODO: try/ except plot
        exec(code, {"df": self._df})


# ai.ask("plot sepal width vs sepal length")

# ai.ask("now color it by species")

# ai.ask("make separate sepal width vs sepal length scatterplot subplots per species. Combine it into a single figure")
