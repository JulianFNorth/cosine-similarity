import os
import httpx
import asyncio
import math

def get_number_of_sentences():
    while True:
        try:
            num = int(input("How many sentences would you like to enter? "))
            if num > 0:
                return num
        except ValueError:
            pass
        print("Enter a valid positive number!\n")

def make_questions(number_of_sentences):
    all_sentences = []
    for i in range(number_of_sentences):
        sentence = input(f"Please enter sentence number {i+1}: ")
        all_sentences.append(sentence)
    return all_sentences

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

def sort_similarities(all_sentences):
    sentence_pairs = []
    for sentence in all_sentences:
        score = float(sentence.split(": ")[-1])
        sentence_pairs.append((sentence, score))

    sorted_sentences = sorted(sentence_pairs, key=lambda x: x[1], reverse=True)

    for sentence, _ in sorted_sentences:
        print(sentence)

async def loop():
    num = get_number_of_sentences()
    sentences = make_questions(num)
    all_sentences = []

    print(sentences)
    for i in range(len(sentences)):
        for j in range(i + 1, len(sentences)):
            emb1 = await get_embedding(sentences[i])
            emb2 = await get_embedding(sentences[j])
            similarity = await calculate_cosine_similarity(emb1, emb2)
            all_sentences.append(f"Sentences {i+1} and {j+1} cosine similarity: {similarity:.4f}")
    
    return all_sentences

def main():
    all_sentences = asyncio.run(loop())
    sort_similarities(all_sentences)

if __name__ == "__main__":
    main()