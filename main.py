import os
import pdb

# input(f"The current process ID is {os.getpid()}. Press enter to continue")

print("importing vllm")
from vllm import LLM
print("Setting environment params")
import vllm.envs as venvs; venvs.VLLM_ENABLE_V1_MULTIPROCESSING = False
print("Setting pdb trace")
pdb.set_trace()
# llm = LLM(model="facebook/opt-125m")
llm = LLM(model="openai/gpt-oss-20b")