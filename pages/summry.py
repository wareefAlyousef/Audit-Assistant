import os
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
from flask import Flask, render_template, request, send_file, jsonify

# Load environment variables from .env file
load_dotenv()

# Initialize client 
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
summary = ""

# Initialize Flask app
app = Flask(
    __name__,
    template_folder="../templates",
    static_folder="../static"
)

string = (
    'predict_fraud(df, model_path="catboost_model.joblib") '
    "Rule-Based Checks check_org check_dest substractAmount substractAmountDest "
    "Flag negative balances for origin or destination "
    "Flag transactions above a given threshold "
    "Flag transactions where sender and recipient are the same "
    "Flag transactions with zero or negative amounts "
    "Flag accounts that make multiple transactions within a short window (velocity check) "
    "window: number of consecutive steps to check "
    "1. Origin balance inconsistency "
    "2. Destination balance inconsistency "
    "3. Negative balances "
    "4. Large transactions "
    "5. Self-transfers "
    "6. Zero or negative amounts "
    "7. High frequency / velocity checks"
)

def call_gpt4(prompt: str, max_tokens = None) -> str:
    response = None
    if (max_tokens is not None):
        response = client.chat.completions.create(
            model="gpt-4",  
            messages=[{"role": "developer", "content": prompt}],
            max_completion_tokens = max_tokens
        )
    else:
        response = client.chat.completions.create(
            model="gpt-4",  
            messages=[{"role": "developer", "content": prompt}]
        )
    return response.choices[0].message.content


# def call_gpt4(prompt: str) -> str:
#     response = client.chat.completions.create(
#         model="gpt-4o",   # you can use "gpt-4o" or "gpt-4o-mini"
#         messages=[{"role": "user", "content": prompt}]
#     )
#     return response.choices[0].message.content

def process_dataframe(df: pd.DataFrame):
    """
    Receives a pandas DataFrame and sends a summary request to GPT-4.
    """
    prompt = f"Summarize the following dataframe and why are they considered fraud in one paragraph don't explane what the columns are just say why do you think it is fraud:\n{df.head().to_string()}"
    summary = call_gpt4(prompt,7000)

    prompt = f"Present the findings from the summary in a clear and concise manner, put them in points:\n{summary}"
    summary = call_gpt4(prompt)

    prompt = f"Keep the way the text is presented, but shorten each point and make it clearer:\n{summary}"
    summary = call_gpt4(prompt)

    # print(summary+"\n")

    return summary.strip()

def process_analysis(df: pd.DataFrame):
    prompt = f"giveing this summary to the user of why are these instances considered Fraud: "+summary+"\n{df.head().to_string()} Given these functions that were appled to the data to determen Froud: "+string+" \n Answer This to the user How was the data analyzed?"
    analysis = call_gpt4(prompt)

    prompt = f"Present the findings from the analysis in a clear and concise manner, put them in points:\n{analysis}"
    analysis = call_gpt4(prompt)

    prompt = f"Keep the way the text is presented, but shorten each point and make it clearer for a non technical audience:\n{analysis}"
    analysis = call_gpt4(prompt)

    prompt = f"rewrite each point in a non technical manner and no need no dive in deep in each point:\n{analysis}"
    analysis = call_gpt4(prompt)

    prompt = f"make it only max to 5 point and less is preferred:\n{analysis}"
    analysis = call_gpt4(prompt)

    print(analysis)

    return analysis.strip()


@app.route("/")
def index():
    return render_template("summary.html")


@app.route("/get_summary", methods=["GET"])
def get_summary():
    try:
        csv_path = "data/3are1.csv"
        print("Loading CSV from:", csv_path)
        df1 = pd.read_csv(csv_path)
        summary_text = process_dataframe(df1)
        return jsonify({"summary": summary_text})
    except Exception as e:
        print("Error:", e)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)


# if __name__ == "__main__":
#     df1 = pd.read_csv("C:\\Users\\warif\\Documents\\GitHub\\Audit-Assistant\\data\\3are1.csv")
#     process_dataframe(df1)



