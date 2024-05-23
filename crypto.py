import os
import requests
import warnings
from typing import Any, List

from langchain.agents import Tool, initialize_agent
from langchain.tools import DuckDuckGoSearchRun
from langchain.llms.base import BaseLLM
from pydantic import BaseModel, Field
from openfabric_pysdk.helper import Proxy
from openfabric_pysdk.utility import SchemaUtil
from langchain.schema import Generation, LLMResult

# Set API keys
os.environ["COINMARKETCAP_API_KEY"] = "1491e542-5af4-49b6-93bb-77df1a196765"

warnings.filterwarnings("ignore")

# Retrieve the LLM proxy link from the environment or config
LLM_PROXY_LINK = "wss://7d7f3f9f1b62466b8f4e07120eccad7a.openfabric.network", 
# "wss://419df75747a0432d97b60b870d6c0535.openfabric.network"  # Replace with your actual proxy link
llm_proxy = Proxy(LLM_PROXY_LINK, "LLM proxy")

# Define functions to get cryptocurrency data
def get_crypto_price(crypto_symbol, history=5):
    url = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest"
    params = {
        "symbol": crypto_symbol,
        "count": history,
        "interval": "daily",
        "convert": "USD"
    }
    headers = {
        "X-CMC_PRO_API_KEY": os.getenv("COINMARKETCAP_API_KEY")
    }
    
    response = requests.get(url, headers=headers, params=params)
    data = response.json()
    
    if 'data' in data:
        result = []
        for crypto in data['data']:
            if crypto['symbol'] == crypto_symbol:
                quote = crypto['quote']['USD']
                date = crypto['last_updated'].split('T')[0]
                price = quote['price']
                volume = quote['volume_24h']
                result.append(f"Date: {date}, Close: {price:.2f}, Volume: {volume:.2f}")
        return "\n".join(result)
    else:
        return "No data found"

def get_crypto_details(crypto_symbol):
    url = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/info"
    params = {
        "symbol": crypto_symbol
    }
    headers = {
        "X-CMC_PRO_API_KEY": os.getenv("COINMARKETCAP_API_KEY")
    }
    
    response = requests.get(url, headers=headers, params=params)
    data = response.json()
    
    if 'data' in data:
        crypto_data = list(data['data'].values())[0]  # Extract the first (and presumably only) item
        details = {
            "Name": crypto_data['name'],
            "Symbol": crypto_data['symbol'],
            "Category": crypto_data['category'],
            "Description": crypto_data['description']
        }
        details_str = "\n".join([f"{key}: {value}" for key, value in details.items()])
        return details_str
    else:
        return "No data found"

def get_latest_crypto_data(crypto_symbol):
    url = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest"
    params = {
        "start": "1",
        "limit": "100",
        "convert": "USD"
    }
    headers = {
        "X-CMC_PRO_API_KEY": os.getenv("COINMARKETCAP_API_KEY")
    }
    
    response = requests.get(url, headers=headers, params=params)
    data = response.json()
    
    if 'data' in data:
        for crypto in data['data']:
            if crypto['symbol'] == crypto_symbol:
                market_data = {
                    "Price (USD)": crypto['quote']['USD']['price'],
                    "Volume 24h (USD)": crypto['quote']['USD']['volume_24h'],
                    "Market Cap (USD)": crypto['quote']['USD']['market_cap'],
                    "Max Supply": crypto['max_supply'],
                    "Circulating Supply": crypto['circulating_supply'],
                    "Total Supply": crypto['total_supply']
                }
                return market_data
        return "Cryptocurrency not found"
    else:
        return "No data found"

# Define the tools
search = DuckDuckGoSearchRun()

tools = [
    Tool(
        name="get latest crypto data",
        func=get_latest_crypto_data,
        description="Use this when you are asked to get details on current price and other metrics about a cryptocurrency. You should input the cryptocurrency symbol to it."
    ),
    Tool(
        name="get crypto data",
        func=get_crypto_price,
        description="Use when you are asked to evaluate or analyze a cryptocurrency. This will output historic price data. You should input the cryptocurrency symbol to it."
    ),
    Tool(
        name="get crypto details",
        func=get_crypto_details,
        description="Use this to get details and metrics about a cryptocurrency. You should input the cryptocurrency symbol to it."
    ),
    Tool(
        name="DuckDuckGo Search",
        func=search.run,
        description="Use this to fetch recent news about cryptocurrencies. You can input the cryptocurrency name to it."
    ),

]

new_prompt = """
You are a financial advisor. Give cryptocurrency recommendations for given query.
Every time first you should identify the cryptocurrency name and get its symbol. Remember symbol is always in uppercase.
Answer the following questions as best you can. You have access to the following tools:

get latest crypto data: Use this when you are asked to get details on current price and other metrics about a cryptocurrency. You should input the cryptocurrency symbol to it.

DuckDuckGo Search: Use this to fetch recent news about cryptocurrencies. You can input the cryptocurrency name to it.

Steps:
1) Get the cryptocurrency name and symbol. Output - cryptocurrency symbol
2) Use "get latest crypto data" tool to gather cryptocurrency info. Output - Crypto data
3) Use "DuckDuckGo Search" tool to search for latest cryptocurrency-related news. Output - Crypto news
4) Analyze the cryptocurrency based on gathered data and give detailed analysis for investment choice only when asked by user. Provide numbers and reasons to justify your answer. Output - Detailed Crypto Analysis

Use the following format:

Question: the input question you must answer unless if it's not related to crypto you can stop there
Thought: you should always think about what to do, also try to follow steps mentioned above
Action: the action to take, should be one of [get latest crypto data, DuckDuckGo Search]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat maximum 5 times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question
If the observation returns "Cryptocurrency not found" or any error, move to the next step.
If you receive the same response multiple times, ensure to avoid re-entering the same action.

Begin!

Question: {input}
Thought:{agent_scratchpad}
"""

class LlmsInput:
    inputText: str = None
    language: str = None
    link: str = None
    file: str = None
    length_summary: str = None
    operationType: str = None
    systemPrompt: str = None


# Function to make the LLM request
def make_llm_request(user_input: str):
    return SchemaUtil.create(
        LlmsInput(),
        dict(
            inputText=user_input,
            operationType="generate",
            systemPrompt=new_prompt,
        )
    )

def perform_llm_request(proxy: Proxy, user_input: str, uid: str) -> str:
    proxy_response: str = ""
    result_key = "outputText"
    llm_request = make_llm_request(user_input)
    try:
        llm_output = proxy.request(vars(llm_request), uid)
        llm_output.wait(timeout=30)

        if llm_output.status() == "COMPLETED":
            proxy_response = llm_output.data()[result_key]
            if "Final Answer:" in proxy_response:
                return proxy_response

        else:
            proxy_response = "Error: LLM did not complete successfully."

    except Exception as e:
        proxy_response = f"Error: {str(e)}"

    return proxy_response

# Making custom LLM class for calling Openfabric LLM 
class CustomLLM(BaseLLM, BaseModel):
    proxy: Proxy

    def __init__(self, proxy: Proxy):
        super().__init__(proxy=proxy)

    def _call(self, prompt: str, stop: Any = None, run_manager: Any = None) -> str:
        return perform_llm_request(self.proxy, prompt, "example-uid")

    @property
    def _llm_type(self) -> str:
        return "custom_llm"

    def _generate(self, prompts: List[str], stop: Any = None, run_manager: Any = None) -> LLMResult:
        responses = [self._call(prompt, stop, run_manager) for prompt in prompts]
        generations = [Generation(text=response) for response in responses]

        # Check for the final answer and stop further generation
        final_answer_found = False
        for response in responses:
            if "Final Answer:" in response:
                final_answer_found = True
                break

        if final_answer_found:
            return LLMResult(generations=[generations])

        return LLMResult(generations=[generations])

# Initialize the custom LLM
llm = CustomLLM(proxy=llm_proxy)

# Initialize the zero-shot agent
zero_shot_agent = initialize_agent(
    llm=llm,
    agent="zero-shot-react-description",
    tools=tools,
    verbose=True,
    max_iteration=4,
    return_intermediate_steps=True,
    handle_parsing_errors=True
)

# Update the agent's prompt template
zero_shot_agent.agent.llm_chain.prompt.template = new_prompt

# Define a function to get the crypto recommendation and print the final answer
def get_crypto_recommendation(question):
    result = zero_shot_agent({"input": question})
    final_answer = result['output']
    print(final_answer)

# Example usage
get_crypto_recommendation("What is bitcoin price today?")

