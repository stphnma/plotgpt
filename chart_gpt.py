import pandas as pd
# import openai
import matplotlib.pyplot as plt
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

def _sanitize(content):
    assert content.partition('\n')[0] == '```python' and content.partition('\n')[1] == '```', ValueError("Not a python block!")
    sanitized = content.replace("```python", "").replace("```","")
    return sanitized

class ChartGPT:
    
    def __init__(self, df: pd.DataFrame):
        
        dtype_str = " " .join([f"{col}({dtype})" for col, dtype in df.dtypes.items()])
        self._llm = ChatOpenAI(model_name="gpt-3.5-turbo")
        
        self._system_template = f"You are an expert Python data analyst. You have been given a dataframe with the following columns: `{dtype_str}`"
    
    def _generate_llm_response(self, prompt):
        
        messages = [
            SystemMessage(content = self._system_template),
            HumanMessage(content=f"Give me the only python code and nothing else for the following. Assume I have the dataframe preloaded : {prompt}")
        ]
        
        resp = self._llm(messages)
        import ipdb; ipdb.set_trace()
        plotting_code = _sanitize(resp.content)
        print(plotting_code)
        exec(plotting_code)
        
        



if __name__ == "__main__":
    import seaborn as sns
    df = sns.load_dataset('iris')
    
    
    chart = ChartGPT(df)
    chart._generate_llm_response("Plot sepal_width vs sepal_length")
    
