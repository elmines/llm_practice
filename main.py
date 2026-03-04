from vllm import LLM
from vllm.sampling_params import BeamSearchParams

def main():
    llm = LLM(model="facebook/opt-125m")
    params = BeamSearchParams(beam_width=5, max_tokens=50)
    # outputs = llm.generate("Hello, my name is")
    outputs = llm.beam_search([{"prompt": "Hello, my name is "}], params)
    for output in outputs:
        generated_text = output.sequences[0].text
        print(f"Prompt: {generated_text}")


if __name__ == "__main__":
    main()
