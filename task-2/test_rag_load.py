import requests
import time
import threading
import json

URL = "http://localhost:8000/rag"

# Defining and merging query sets
test_queries_easy = [
    "What are common pet animals?",                             # OK
    "Which animals can hover in the air?",                      # INCORRECT (sometimes: parrots)
    "Name a pet fish.",                                         # OK
    "What is a low-maintenance pet?",                           # OK
    "Which animals are kept in tanks or bowls?",                # INCORRECT (always: hamsters)
    "Which pets are rodents?",                                  # OK
    "What animal can mimick speech?",                           # OK
    "Which pets are herbivores?",                               # OK
    "Which pets live in burrows?",                              # INCORRECT (sometimes: burrows)
    "What kind of animals are ferrets?"                         # INCORRECT (sometimes: ferrets)
]
test_queries_general = [
    "What is the lifespan of an elephant?",                     # OK        (60-70yy)
    "How do whales communicate?",                               # OK
    "How reptiles and amphibians are different?",               # INCORRECT (sometimes: compares dogs and other domesticated animals; sometimes: activates reason-based selection)
    "Why do birds migrate?",                                    # INCORRECT (sometimes: talks of hunting techniques of birds of prey)
    "What are endangered species?",                             # OK        (sometimes: activates multi-answer selection)
    "How does photosynthesis work?",                            # OK
    "What is natural selection?",                               # OK
    "What is a keystone species' role in an ecosystem?",        # OK
    "What causes animal extinction?",                           # OK        (sometimes: activates multi-answer selection)
    "What is the food chain in a forest ecosystem?"             # INCORRECT (sometimes: talks about rabbits' diet; sometimes: activates multi-answer selection)
]
test_queries_mixed = [
    "Why are dogs considered loyal animals?",                   # OK
    "What makes parrots capable of mimicking speech?",          # OK
    "How do turtles differ from amphibians?",                   # INCORRECT (always: compares turtles to dogs)
    "Are there behavioral differences between cats and dogs?",  # OK
    "Are rabbits suitable pets for children?",                  # INCORRECT (sometimes: explains how they are not suitable due to high aggressivity and size)
    "What are the advantages of keeping fish as pets?",         # INCORRECT (sometimes: compares hamsters and dogs; sometimes: activates reason-based selection)
    "How do ferrets behave compared to other pets?",            # OK
    "What skills make dogs useful as assistance animals?",      # INCORRECT (sometimes: present parrots' mimicking ability; always: activates multi-answer selection)
    "Why are guinea pigs considered sociable animals?",         # OK        (sometimes: activates reason-based selection)
    "What roles do birds play in ecosystems and as pets?"       # OK
]
TEST_QUERIES = test_queries_easy + test_queries_general + test_queries_mixed
maxQueryLen = max([len(query) for query in TEST_QUERIES])+2

def answer_reader(s):
    """
    Answer parser.
    """
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

def answer_printer(s):
    """
    Answer output formatter.
    """
    lines = s.strip().split("\n")
    first_line = lines[0].strip()
    l = 0
    if (len(first_line) < 3 or 'answer' in first_line.lower()) and len(lines)>1: # In case the multi-answer mode is activated and the first line does not contain the answer
        first_line = first_line+"\\n"+lines[1]
        l += 1
    first_line_sentences = first_line.split(". ")
    first_sentence = first_line_sentences[0].strip()
    s = 0
    if (len(first_sentence) < 5 or 'answer' in first_sentence.lower()) and len(first_line_sentences)>1: # In case the multi-answer mode is activated and the first sentence does not contain the answer
        first_sentence = first_sentence+first_line_sentences[1]
        s += 1
    if len(first_line_sentences) > 1+l:
        return first_sentence + ". [continues...]"
    if len(lines) > 1+s:
        return first_sentence + " [continues in other line...]"
    return first_sentence

def send_request(query, results):
    """
    Request sender.
    """
    payload = json.dumps({"query": query, "k": 2})
    headers = {"Content-Type": "application/json"}

    start_time = time.time()
    response = requests.post(URL, headers=headers, data=payload)
    elapsed_time = time.time() - start_time
    results.append(elapsed_time)

    if response.status_code == 200:
        response_json = response.json()
        answer = response_json.get("result", "").strip()
        print(f"{query.ljust(maxQueryLen)}|{elapsed_time:>6.2f}s |  {answer_printer(answer_reader(answer).get('answer', ''))}")
    else:
        print(f"{query.ljust(maxQueryLen)}|{elapsed_time:>6.2f}s |  X (status: {response.status_code}) Error: {response.text}")

def run_load_test(requests_per_second):
    """
    Tester for specific requests rate.
    """
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
    print("-"*maxQueryLen+"------------------- - - -")
    print(f"Completed {len(TEST_QUERIES)} requests at {requests_per_second:>5.2f} requests/s!")
    print(f"Average response time: {avg_time:>6.2f}s")

# Testing service at different requests-rate loads
if __name__ == "__main__":
    for rate in [0.10, 0.25, 0.5, 1, 2, 5, 10, 20, 50, 100]:
        testTitle = f"\n===== TESTING {rate:>5.2f} requests/s (one request every {(1/rate):>5.2f}s)"
        print(f"{testTitle.ljust(maxQueryLen)} ================== = = =")
        queryLabel = "Query".ljust(maxQueryLen)
        print(f"{queryLabel}|  Time  |  Answer")
        print("-"*maxQueryLen+"------------------- - - -")
        run_load_test(rate)
        print("="*maxQueryLen+"=================== = = =\n")