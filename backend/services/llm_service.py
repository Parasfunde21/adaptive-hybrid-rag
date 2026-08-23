import ollama


class LLMService:

    def __init__(self):

        self.model = "qwen2.5:7b"

    def generate(
        self,
        prompt: str
    ):

        response = ollama.chat(
            model=self.model,

            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],

            options={
                # Keep responses detailed but avoid
                # unnecessarily long generation.
                "num_predict": 400,

                # Low temperature improves grounding
                # and reduces unsupported information.
                "temperature": 0.1,

                "top_p": 0.9,

                # Explicit context size.
                "num_ctx": 4096,
            }
        )

        return response["message"]["content"]


llm_service = LLMService()