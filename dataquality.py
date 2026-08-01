import json
from pydantic import BaseModel
from litellm import completion
from colorama import Fore

class Score(BaseModel):
    score:int
    explanation: str

class Rank(BaseModel):
    accuracy: Score
    style: Score

def llm_call(record:str)->dict:
    stream=completion(
        model="ollama_chat/nativebot:latest",
        messages=[
            {
                "role":"user",
                "content":f"""You are an expert evaluator for instruction-tuning datasets.

                    Your task is to evaluate the following Question-Answer pair on two dimensions: Accuracy and Style.
                    Definitions:

                    1. Accuracy (Score: 1-10)
                    Evaluate whether the answer correctly and completely answers the question.
                    Scoring guide:
                    - 10: Factually correct and completely answers the question.
                    - 8-9: Correct but slightly incomplete or missing minor details.
                    - 5-7: Partially correct but contains omissions or small inaccuracies.
                    - 2-4: Mostly incorrect or answers a different question.
                    - 1: Completely incorrect, nonsensical, irrelevant, or harmful.

                    2. Style (Score: 1-10)
                    Evaluate the quality of the answer, NOT its factual correctness.

                    Consider:
                    - Clarity
                    - Grammar
                    - Fluency
                    - Helpfulness
                    - Politeness
                    - Appropriate level of detail

                    Scoring guide:
                    - 10: Exceptionally clear, natural, helpful and well-written.
                    - 8-9: Clear and natural with only minor issues.
                    - 5-7: Understandable but awkward, incomplete or poorly phrased.
                    - 2-4: Difficult to understand or poorly written.
                    - 1: Harmful, offensive, incoherent, empty, or unreadable.

                    Important Rules:
                    - Evaluate the answer in the language it is written. Do NOT penalize Hindi, Telugu, English, or any other language.
                    - Do NOT penalize short answers if they correctly answer the question.
                    - A concise but correct answer can receive Accuracy = 10 and Style = 10.
                    - Do NOT invent missing information.
                    - If the answer is factually correct, give a high Accuracy score.
                    - Base your judgment only on the provided question and answer.

                    Question:
                    {record["question"]}

                    Answer:
                    {record["answer"]}

                    Return ONLY valid JSON matching this schema:

                    {{
                    "accuracy": {{
                        "score": integer,
                        "explanation": "brief explanation"
                    }},
                    "style": {{
                        "score": integer,
                        "explanation": "brief explanation"
                    }}
                    }}""",
            }
        ],
        stream=True,
        options={"num_predict":2000,"temperature":0.2},
        format=Rank.model_json_schema()
    )
    data=""
    for x in stream:
        delta=x['choices'][0]["delta"]["content"]
        if delta is not None:
            print(Fore.LIGHTBLUE_EX +delta+ Fore.RESET,end="")
            data+=delta
    return json.loads(data)

if __name__=="__main__":
    with open("data/instruction.jsonl", "r", encoding="utf-8") as infile, \
        open("data/instruction_quality.jsonl", "w", encoding="utf-8") as instruction_out, \
        open("quality_results.jsonl", "w", encoding="utf-8") as quality_out:

        for line in infile:
            line = line.strip()
            if not line:
                continue

            pair = json.loads(line)

            print(Fore.YELLOW + str(pair) + Fore.RESET)

            result = llm_call(pair)

            if result["accuracy"]["score"] >= 6 and result["style"]["score"] >= 6:
                # Write instruction immediately
                instruction_out.write(
                    json.dumps(pair, ensure_ascii=False) + "\n"
                )
                instruction_out.flush()   # Optional: ensures it's written immediately

                # Write quality result immediately
                quality_out.write(
                    json.dumps(
                        {**pair, "quality": result},
                        ensure_ascii=False
                    ) + "\n"
                )
                quality_out.flush()       # Optional