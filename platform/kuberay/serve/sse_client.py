"""Simple async SSE client for the Ray Serve LLM endpoint.

Usage:
    python sse_client.py
    python sse_client.py http://127.0.0.1:8000/v1/chat/completions '{"messages":[{"role":"user","content":"Hello"}]}'

Requires: `httpx`
"""
import asyncio
import json
import sys
from typing import Optional

import httpx


async def sse_client(url: str, payload: dict):
    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream("POST", url, json=payload) as resp:
            resp.raise_for_status()
            buffer = ""
            event: Optional[str] = None
            data_lines: list[str] = []

            prev_char: str = ""
            async for chunk in resp.aiter_bytes():
                text = chunk.decode(errors="replace")
                buffer += text

                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    line = line.rstrip("\r")

                    if line == "":
                        # end of event — clear event name
                        event = None
                        data_lines = []
                        continue

                    if line.startswith("event:"):
                        event = line[len("event:"):].strip()
                        continue

                    if line.startswith("data:"):
                        # preserve token-leading whitespace: remove only a single separator space after the colon
                        payload = line[len("data:"):]
                        if payload.startswith(" "):
                            payload = payload[1:]
                        # try to parse JSON-ish payload; if it's a string, print raw
                        try:
                            parsed = json.loads(payload)
                        except Exception:
                            parsed = payload
                        if isinstance(parsed, str):
                            out = parsed
                        else:
                            out = json.dumps(parsed, ensure_ascii=False)

                        # insert a space only when both boundary chars are ASCII alphanumeric
                        if prev_char and out:
                            first_char = out[0]
                            if prev_char.isascii() and first_char.isascii() and prev_char.isalnum() and first_char.isalnum():
                                print(" ", end="", flush=False)

                        # print inline (no newline) to preserve streaming sentence flow
                        print(out, end="", flush=True)
                        if out:
                            # if the token ends with </think>, add a blank line separator
                            if out.strip().endswith("</think>"):
                                print("\n\n", end="", flush=True)
                                prev_char = "\n"
                            else:
                                prev_char = out[-1]
                        continue

            # flush any remaining buffered data (treat as a final data: line)
            if buffer:
                line = buffer.rstrip("\r\n")
                if line.startswith("data:"):
                    payload = line[len("data:"):]
                    if payload.startswith(" "):
                        payload = payload[1:]
                    try:
                        parsed = json.loads(payload)
                    except Exception:
                        parsed = payload
                    if isinstance(parsed, str):
                        out = parsed
                    else:
                        out = json.dumps(parsed, ensure_ascii=False)

                    if prev_char and out:
                        first_char = out[0]
                        if prev_char.isascii() and first_char.isascii() and prev_char.isalnum() and first_char.isalnum():
                            print(" ", end="", flush=False)

                    print(out, end="", flush=True)
                    if out:
                        if out.strip().endswith("</think>"):
                            print("\n\n", end="", flush=True)
                            prev_char = "\n"
                        else:
                            prev_char = out[-1]

            # print final newline after stream completes
            print()


async def main():
    url = "http://127.0.0.1:8000/v1/chat/completions"
    payload = {"messages": [{"role": "user", "content": "안녕하세요! 자기소개 해주세요."}], "temperature": 0.7}

    if len(sys.argv) >= 2:
        url = sys.argv[1]
    if len(sys.argv) >= 3:
        try:
            payload = json.loads(sys.argv[2])
        except Exception as e:
            print("Failed to parse payload JSON:", e)
            return

    print(f"Connecting to {url} with payload:\n{json.dumps(payload, ensure_ascii=False)}\n")
    try:
        await sse_client(url, payload)
    except httpx.HTTPStatusError as e:
        print("HTTP error:", e)
    except Exception as e:
        print("Error:", e)


if __name__ == "__main__":
    asyncio.run(main())
