import time

import ollama


def main() -> None:
    texts = ["embedding performance test sentence"] * 16
    start = time.perf_counter()
    result = ollama.embed(model="qwen3-embedding:4b", input=texts)
    elapsed = time.perf_counter() - start
    embeddings = result["embeddings"]
    print("count", len(embeddings))
    print("dim", len(embeddings[0]))
    print("seconds", round(elapsed, 3))
    print("items_per_sec", round(len(texts) / elapsed, 2))


if __name__ == "__main__":
    main()
