import requests
import time
import threading
import json

URL = "http://localhost:8000/rag"

# Define query sets
test_queries_easy = [
    "What are common pet animals?",
    "Which animals can hover in the air?",
    "Name a pet fish.",
    "What is a low-maintenance pet?",
    "Which animals are kept in tanks or bowls?",
    "Which pets are rodents?",
    "What animal is known for mimicking speech?",
    "Which pets are herbivores?",
    "Which pets live in burrows?",
    "What kind of animals are ferrets?"
]
test_queries_general = [
    "What is the lifespan of an elephant?",
    "How do whales communicate?",
    "What are the main differences between reptiles and amphibians?",
    "Why do birds migrate?",
    "What are endangered species?",
    "How does photosynthesis work?",
    "What is natural selection?",
    "What is the role of a keystone species in an ecosystem?",
    "What causes animal extinction?",
    "What is the food chain in a forest ecosystem?"
]
test_queries_mixed = [
    "Why are dogs considered loyal animals?",
    "What makes parrots capable of mimicking human speech?",
    "How do turtles differ from amphibians?",
    "What are the behavioral differences between cats and dogs?",
    "Are rabbits suitable pets for children?",
    "What are the advantages of keeping fish as pets?",
    "How do ferrets behave compared to other pets?",
    "What skills make dogs useful as assistance animals?",
    "Why are guinea pigs considered sociable animals?",
    "What roles do birds play in ecosystems and as pets?"
]

TEST_QUERIES = test_queries_easy + test_queries_general + test_queries_mixed

# Answer parser
def answer_reader(s):
    result = {}
    current_key = None
    lines = s.strip().split("\n")

    for line in lines:
        if ":" in line:
            key, value = line.split(":", 1)
            key = key.strip().lower()
            value = value.strip()
            result[key] = value
            current_key = key
        else:
            if current_key:
                result[current_key] += "\n" + line.strip()
    return result

# Request sender
def send_request(query, results):
    payload = json.dumps({"query": query, "k": 2})
    headers = {"Content-Type": "application/json"}

    start_time = time.time()
    response = requests.post(URL, headers=headers, data=payload)
    elapsed_time = time.time() - start_time
    results.append(elapsed_time)

    if response.status_code == 200:
        response_json = response.json()
        answer = response_json.get("result", "").strip()
        parsed = answer_reader(answer)
        print(f"Query: {query} | Time: {elapsed_time:.2f}s | Status: {response.status_code} | Answer: {parsed.get('answer', '')}")
    else:
        print(f"Query: {query} | Time: {elapsed_time:.2f}s | Status: {response.status_code} | Error: {response.text}")

# Load test runner
def run_load_test(requests_per_second):
    results = []
    interval = 1.0 / requests_per_second
    threads = []

    for query in TEST_QUERIES:
        thread = threading.Thread(target=send_request, args=(query, results))
        thread.start()
        threads.append(thread)
        time.sleep(interval)

    for thread in threads:
        thread.join()

    avg_time = sum(results) / len(results) if results else 0
    print(f"\nCompleted {len(TEST_QUERIES)} requests at {requests_per_second} req/s")
    print(f"Average response time: {avg_time:.2f}s")

# Main
if __name__ == "__main__":
    for rate in [1, 2, 5, 10, 20]:
        print(f"\n=== Testing {rate} requests per second ===")
        run_load_test(rate)