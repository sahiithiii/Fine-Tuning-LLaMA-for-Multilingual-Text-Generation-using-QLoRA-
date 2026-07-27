import json
from colorama import Fore,init
import os

os.makedirs("data", exist_ok=True)
init(autoreset=True)

instructions=[]
with open('indic_align_multilingual.json','r',encoding="utf-8") as f:
    data=json.load(f)
    for key,chunk in data.items():
        for pairs in chunk['generated']:
            instructions.append(pairs)
print("Collected:",len(instructions))

with open('data/instruction.json','w',encoding="utf-8") as f:
    json.dump(instructions,f)

with open('data/instruction.json','r',encoding="utf-8") as f:
    data=json.load(f)
    print(Fore.LIGHTBLACK_EX+str(data[:10]))