import os
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file (if present)
load_dotenv()

# Initialize client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def call_gpt4(prompt: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o",   # you can use "gpt-4o" or "gpt-4o-mini"
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

def process_dataframe(df: pd.DataFrame):
    """
    Receives a pandas DataFrame and sends a summary request to GPT-4.
    """
    prompt = f"Summarize the following dataframe and why are they considered fraud in one paragraph don't explane what the columns are just say why do you think it is fraud:\n{df.head().to_string()}"
    summary = call_gpt4(prompt)
    print(summary)

if __name__ == "__main__":
    df1 = pd.read_csv("3are1.csv")
    process_dataframe(df1)