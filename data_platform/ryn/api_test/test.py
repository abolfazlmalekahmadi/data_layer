import aiohttp
import asyncio

async def send_request(session, url, payload):
    async with session.post(url, json=payload) as response:
        return await response.json()

async def main():
    url = "http://127.0.0.1:6990/huggingface-to-pvc"
    payload = {
        "dataset_name": "fka/awesome-chatgpt-prompts",
        "dataset_config": None,
        "dataset_name_override": "fka/awesome-chatgpt-prompts",
        "user_name": "Ali",
        "private": False
    }
    async with aiohttp.ClientSession() as session:
        tasks = [send_request(session, url, payload) for _ in range(30)]
        responses = await asyncio.gather(*tasks)
        for i, resp in enumerate(responses, 1):
            print(f"Response {i}: {resp}")

asyncio.run(main())