def prompt_template(data: str, num_records: int = 5):
    return f"""You are an expert data curator assisting a machine learning engineer in creating a high-quality instruction-tuning dataset for Indic languages.

Your task: transform the provided data into {num_records} diverse question-and-answer (Q&A) pairs that will be used to fine-tune a language model.

Guidelines:
- Generate {num_records} distinct Q&A pairs, each reflecting a different aspect of the source data.
- Mix question lengths: some short (1-2 sentences), some longer (3-4 sentences).
- Keep each answer concise but informative, capturing key insights from the source data.
- IMPORTANT: Write questions and answers in the SAME language and script as the source data. Do not translate into English unless the source data itself is in English. If the source mixes a regional language with Romanized/transliterated text, preserve that same style in your output.
- Avoid sensitive, biased, or toxic content. If the source data itself contains a toxic or unsafe prompt, the answer should reflect a safe, appropriate response rather than repeating or amplifying the harmful content.
- Do not include any commentary, explanation, or text outside the JSON output.

Output strictly as a JSON object matching this structure:
{{
    "generated": [
        {{"question": "...", "answer": "..."}},
        {{"question": "...", "answer": "..."}}
    ]
}}

Source data:
{data}
"""


if __name__ == "__main__":
    print(prompt_template("Sahithi Akula", 10))