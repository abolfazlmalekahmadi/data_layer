import aiohttp
import asyncio

async def send_request(session, url, payload):
    async with session.post(url, json=payload) as response:
        return await response.json()

async def main():
    # url = "http://127.0.0.1:6991/huggingface"
    # payload = {
    #     "dataset_name": "glue",
    #     "dataset_config": "axx",
    #     "dataset_name_override": "fka/awesome-chatgpt-prompts",
    #     "user_name": "Ali",
    #     "private": False
    # }
    url = "http://127.0.0.1:6992/openml"
    payload = {
        "dataset_name": "daaaaa",
        "dataset_id": 31,
        "user_name": "Alis",
        "private": False
    }
    async with aiohttp.ClientSession() as session:
        tasks = [send_request(session, url, payload) for _ in range(10)]
        responses = await asyncio.gather(*tasks)
        for i, resp in enumerate(responses, 1):
            print(f"Response {i}: {resp}")

asyncio.run(main())