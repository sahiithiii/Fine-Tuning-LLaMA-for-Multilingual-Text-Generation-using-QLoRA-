"""
Key differences from the pdf version:
- No chunking
- Rows are already instruction/response (or multi-turn) data, so we first
try to extract {question, answer} directly. Only if a row doesn't have
a clean single Q/A shape (e.g. multi-turn "messages"/"conversations",
or you explicitly want the LLM to synthesize *additional* Q&A pairs
from the content) do we fall back to your original llm_call().
"""

import json
from typing import List,Optional
from colorama import Fore
from datasets import load_dataset
from pydantic import BaseModel
from litellm import completion
from generated_prompt import prompt_template



import json
from datasets import load_dataset
from colorama import Fore,init

init(autoreset=True)

CONFIGS = [
    "Anudesh",
    "Dolly_T",
    "HHRLHF_T",
]


# for config in CONFIGS:
#     print(Fore.GREEN + f"\n========== {config} ==========" + Fore.RESET)

#     ds = load_dataset(
#         "ai4bharat/indic-align",
#         config,
#         split="train"
#     )

#     print(Fore.CYAN + "Number of rows:" + Fore.RESET, len(ds))
#     print(Fore.CYAN + "Column names:" + Fore.RESET)
#     print(ds.column_names)

#     print(Fore.YELLOW + "\nFirst row:\n" + Fore.RESET)
#     print(json.dumps(ds[0], indent=2, ensure_ascii=False))

#     print("\n" + "=" * 80)

dataset={}
idx=0

for config in CONFIGS:
    print(Fore.GREEN+f"\n=======Loading {config} ==========")
    ds=load_dataset(
        "ai4bharat/indic-align",
        config,
        split="train" 
    )

    print(Fore.CYAN+f"Rows:{len(ds)}")
    print(Fore.CYAN+f"Columns:{len(ds.column_names)}")

    if config=="Anudesh":
        for row in ds:
            generated=[]

            for pair in row["interactions"]:
                if len(pair)!=2:
                    continue
                question,answer=pair
                if question is None or answer is None:
                    continue
                question=str(question).strip()
                answer=str(answer).strip()
                if question=="" or answer=="":
                    continue
                generated.append({
                    "question":question,
                    "answer":answer
                })
            dataset[idx]={
                "generated":generated,
                "source":{
                    "dataset":"indic-align",
                    "config":config,
                    "id":row["id"]
                }
            }

            idx+=1
    else:
        language_columns=[
            col for col in ds.column_names
            if col not in(
                "doc_id",
                "num_turns",
                "__index_level_0__"
            )
        ]

        print(Fore.YELLOW+f"Languages found:{len(language_columns)}")

        for row in ds:
            generated=[]
            for lang in language_columns:
                conversations=row.get(lang)
                if not conversations:
                    continue
                for pair in conversations:
                    if len(pair)!=2:
                        continue
                    question,answer=pair

                    if not question or not answer:
                        continue
                    generated.append({
                        "question":question.strip(),
                        "answer":answer.strip(),
                        "language":lang
                    })

            if generated:
                dataset[idx]={
                    "generated":generated,
                    "source":{
                        "dataset":"indic-align",
                        "config":config,
                        "doc_id":row["doc_id"]
                    }
                }
                idx+=1

print(Fore.GREEN+f"\n Total examples saved:{idx}")
with open("indic_align_multilingual.json","w",encoding="utf-8") as f:
    json.dump(
        dataset,
        f,
        ensure_ascii=False,
        indent=2
    )
print(Fore.GREEN+"\nSaved to indic_align_multilingual.json")














# #which subsets to pull
# # CONFIG_SPLIT_MAP={
# #     "IndicAlign-Instruct":["Anudesh","Dolly_T"],
# #     "IndicAlign-Toxic":["HHRLHF_T"],
# # }

# CONFIGS = [
#     "Anudesh",
#     "Dolly_T",
#     "HHRLHF_T",
# ]


# class Record(BaseModel):
#     question:str
#     answer:str

# class Response(BaseModel):
#     generated: List[Record]

# def llm_call(data:str,num_records:int=5) -> dict:
#     stream=completion(
#         model="ollama_chat/llama2:latest",
#         messages=[
#             {
#                 "role":"user",
#                 "content":prompt_template(data,num_records),
#             }
#             ],
#         stream=True,
#         options={"num_predict":2000},
#         format=Response.model_json_schema(),
#     )
#     data_out=""
#     for x in stream:
#         delta=x['choices'][0]['delta']['content']
#         if delta is not None:
#             print(Fore.LIGHTBLACK_EX+delta+Fore.RESET,end="")
#             data_out+=delta
#     return json.loads(data_out)


# def extract_qa_from_row(row:dict) -> Optional[dict]:
#     """
#     Try to pull a clean {questio,answer,source_text} straight out of 
#     a row without calling the LLM. Handles the common indic-align shapes. 
#     Returns None if the row doesn't match the known shape (caller can then 
#     decide to fall back to llm_call on the raw text).
#     """

#     #Shape A: instrcution/context/response (Dolly style)
#     if "instruction" in row and "response" in row:
#         ctx=(row.get("context") or "").strip()
#         question=row["instruction"] if not ctx else f'{row["instruction"]}\n\n{ctx}'
#         return {"question":question,"answer":row["response"]}

#     #Shape B: prompt/response
#     if "prompt" in row and "response" in row:
#         return {"question":row["prompt"],"answer":row["response"]}

#     #Shape C: messages/conversations list of {role,content}
#     turns=row.get("messages") or row.get("conversations")
#     if turns:
#         user_turns=[t.get("content","") for t in turns if t.get("role")=="user"]
#         asst_turns=[t.get("content","") for t in turns if t.get("role") in ("assistant","model")]
#         if user_turns and asst_turns:
#             return {"questions":user_turns[0],"answer":asst_turns[0]}

#     # Shape D: interactions
#     if "interactions" in row:
#         interactions = row["interactions"]

#         if interactions:
#             return {
#                 "question": interactions[0][0],
#                 "answer": interactions[0][1]
#             }
#     return None

# def row_to_text(row:dict)->str:
#     """Flatten a row into plain text, for feeding to llm_call as a fallback."""
#     parts=[]
#     for key in ("instruction","context","prompt","response","question","answer"):
#         if row.get(key):
#             parts.append(str(row[key]))
#     turns=row.get("messages") or row.get("conversations")
#     if turns:
#         for t in turns:
#             parts.append(f'{t.get("role","")}:{t.get("content","")}')
#     return "\n\n".join(parts) if parts else json.dumps(row,ensure_ascii=False)

# if __name__=="__main__":
#     dataset={}
#     idx=0
#     for config_name in CONFIGS:
#             print(Fore.GREEN+f"\n===Loading {config_name} ==="+Fore.RESET)
#             ds=load_dataset("ai4bharat/indic-align",config_name,split="train")

#             for row in ds:
#                 print(Fore.YELLOW+f"Row Keys: {list(row.keys())}"+Fore.RESET)
#                 direct=extract_qa_from_row(row)
#                 if direct is not None:
#                     #already structured q/a
#                     dataset[idx]={
#                         "generated":[direct],
#                         "context":row_to_text(row),
#                         "source":{"config":config_name,"split":"train"},
#                     }
#                 else:
#                     #fall back to your original approach: ask llm to synthesize qa pairs from the flattened text
#                     enriched_text=row_to_text(row)
#                     print(Fore.LIGHTMAGENTA_EX+f"Text:\n{enriched_text[:300]}..."+Fore.RESET)
#                     generated=llm_call(enriched_text)
#                     dataset[idx]={
#                         "generated":generated["generated"],
#                         "context":enriched_text,
#                         "source":{"config":config_name,"split":"train"}
#                     }

#                 idx+=1

# with open("indic_align_data.json","w",encoding="utf-8") as f:
#     json.dump(dataset,f,ensure_ascii=False,indent=2)
