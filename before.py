from litellm import completion
from colorama import Fore

def llm_call(prompt:str) -> None:
    stream=completion(
        model="ollama_chat/llama2:latest",
        #top_k=1,
        messages=[
            {
                "role":"user",
                "content":prompt,
            }
        ],
        stream=True,
    )
    data=""
    for x in stream:
        delta=x['choices'][0]['deltha']['content']
        if delta is not None:
            print(Fore.LIGHTBLACK_EX+delta+Fore.RESET,end="")
            data+=delta

if __name__=="__main__":
    llm_call("How do you How are you in Hindi?")
