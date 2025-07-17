import os
import httpx
import asyncio
import math

first_sentence = input("Enter the first sentence: ")
second_sentence = input("Enter the second sentence: ")

async def get_embedding(sentence: str):
    url = "http://ai.thewcl.com:6502/embedding_vector"
    headers = {
        "Authorization": f"Bearer {os.environ['BEARER_TOKEN']}"
    }
    async with httpx.AsyncClient() as client:
        response = await client.post(url, headers=headers, json={"text": sentence})
        response.raise_for_status()
        return response.json()["embedding"]

async def calculate_cosine_similarity(embedding1: list[float], embedding2: list[float]):
    dot_product = sum(a * b for a, b in zip(embedding1, embedding2))
    magnitude1 = math.sqrt(sum(a * a for a in embedding1))
    magnitude2 = math.sqrt(sum(b * b for b in embedding2))
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0
    return dot_product / (magnitude1 * magnitude2)

async def main():
    first_embedding = await get_embedding(first_sentence)
    second_embedding = await get_embedding(second_sentence)
    print("First embedding:", first_embedding[:5])
    print("Second embedding:", second_embedding[:5])
    similarity = await calculate_cosine_similarity(first_embedding, second_embedding)
    print(f"\nCosine similarity: {similarity:.4f}")

if __name__ == "__main__":
    asyncio.run(main())