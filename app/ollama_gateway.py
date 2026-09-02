from ollama import AsyncClient


class OllamaGateway:
    def __init__(self, client: AsyncClient | None = None) -> None:
        self.client = client or AsyncClient()

    async def generate_json(self, model: str, system: str, user: str) -> str:
        response = await self.client.chat(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            format="json",
            options={"temperature": 0.1},
        )
        return response["message"]["content"]

