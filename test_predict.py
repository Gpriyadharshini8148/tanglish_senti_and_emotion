
import requests
import json
import time

url_health = 'http://127.0.0.1:5000/health'
url_predict = 'http://127.0.0.1:5000/predict'
data = {'text': 'This movie is super good!'}
headers = {'Content-Type': 'application/json'}

def wait_for_server(url, timeout=60):
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = requests.get(url, timeout=1)
            if resp.status_code == 200:
                print("Server is up!")
                return True
        except Exception:
            time.sleep(1)
            print(".", end="", flush=True)
    print("\nServer timed out.")
    return False

if wait_for_server(url_health):
    print(f"Sending predict request to {url_predict}...")
    start = time.time()
    try:
        response = requests.post(url_predict, json=data, headers=headers, timeout=600)
        print(f"Response Status: {response.status_code}")
        print(f"Response Text: {response.text}")
        print(f"Total predict time: {time.time() - start:.2f}s")
    except Exception as e:
        print(f"Predict request failed: {e}")
else:
    print("Could not connect to server health check.")
