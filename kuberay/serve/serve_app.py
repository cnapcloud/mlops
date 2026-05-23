from ray import serve
from fastapi import FastAPI
from starlette.requests import Request
from starlette.responses import StreamingResponse
from openai import AsyncOpenAI
import socket

app = FastAPI()

@serve.deployment(
    num_replicas=2,
    ray_actor_options={"num_cpus": 1}
)
@serve.ingress(app)
class LLMService:
    def __init__(self):
        self.hostname = socket.gethostname()
        self.client = AsyncOpenAI(
            base_url="http://192.168.0.75:11434/v1",
            api_key="dummy"
        )

    @app.post("/v1/chat/completions")
    async def chat(self, request: Request):
        body = await request.json()
        messages = body["messages"]
        hostname = self.hostname
        client = self.client

        async def generate():
                stream = await client.chat.completions.create(
                    model="qwen3:8b",
                    messages=messages,
                    temperature=body.get("temperature", 0.7),
                    stream=True,
                    extra_body={"think": True}
                )

                def _sse_message(data: str, event: str | None = None, id: str | None = None) -> str:
                    parts = []
                    if id is not None:
                        parts.append(f"id: {id}")
                    if event is not None:
                        parts.append(f"event: {event}")
                    # splitlines preserves multiline payloads and sends each line as its own data: entry
                    for line in str(data).splitlines() or [""]:
                        parts.append(f"data: {line}")
                    # ensure a blank line (\n\n) terminates the event per SSE spec
                    return "\n".join(parts) + "\n\n"

                async for chunk in stream:
                    delta = chunk.choices[0].delta

                    if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                        yield _sse_message(delta.reasoning_content, event="reasoning")
                    if getattr(delta, "content", None):
                        yield _sse_message(delta.content, event="message")
        
        headers = {
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }

        return StreamingResponse(generate(), media_type="text/event-stream", headers=headers)


deployment = LLMService.bind()

if __name__ == "__main__":
    serve.start(
        http_options={"host": "0.0.0.0", "port": 8000}
    )
    print("Ray Serve running on http://127.0.0.1:8000")
    serve.run(deployment, blocking=True)