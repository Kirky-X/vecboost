import requests
import time
import sys

BASE_URL = "http://localhost:9002"

def wait_for_server(timeout=300):
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"{BASE_URL}/health")
            if response.status_code == 200:
                print("Server is ready!")
                return True
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(1)
        print("Waiting for server...")
    print("Timeout waiting for server.")
    return False

def test_health():
    print("Testing /health...")
    response = requests.get(f"{BASE_URL}/health")
    assert response.status_code == 200
    # The actual implementation returns plain text "OK", not JSON
    assert response.text == "OK"
    print("Health check passed.")

def test_embed():
    print("Testing /api/v1/embed...")
    payload = {"text": "Hello, world!"}
    response = requests.post(f"{BASE_URL}/api/v1/embed", json=payload)
    if response.status_code != 200:
        print(f"Embed failed: {response.text}")
    assert response.status_code == 200
    data = response.json()
    assert "embedding" in data
    assert "dimension" in data
    assert len(data["embedding"]) == data["dimension"]
    print(f"Embed check passed. Dimension: {data['dimension']}")

def test_similarity():
    print("Testing /api/v1/similarity...")
    payload = {
        "source": "Hello world",
        "target": "Hi world"
    }
    response = requests.post(f"{BASE_URL}/api/v1/similarity", json=payload)
    if response.status_code != 200:
        print(f"Similarity failed: {response.text}")
    assert response.status_code == 200
    data = response.json()
    assert "score" in data
    print(f"Similarity check passed. Score: {data['score']}")

def test_batch_embed():
    print("Testing /api/v1/embed/batch...")
    payload = {"texts": ["Hello", "World"]}
    response = requests.post(f"{BASE_URL}/api/v1/embed/batch", json=payload)
    if response.status_code != 200:
        print(f"Batch embed failed: {response.text}")
    assert response.status_code == 200
    data = response.json()
    assert "embeddings" in data
    assert len(data["embeddings"]) == 2
    print("Batch embed check passed.")

if __name__ == "__main__":
    if not wait_for_server():
        sys.exit(1)

    try:
        test_health()
        test_embed()
        test_similarity()
        test_batch_embed()
        print("\nAll integration tests passed!")
    except Exception as e:
        print(f"\nTests failed: {e}")
        sys.exit(1)
