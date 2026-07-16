import os
import time

import requests

start_id = 875
end_id = 4353

base_url = "https://dccam.net/documents/{}/pdf"

script_dir = os.path.dirname(os.path.abspath(__file__))

headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

print(f"--- Starting downloads for range {start_id} to {end_id} ---")

for doc_id in range(start_id, end_id + 1):
    url = base_url.format(doc_id)

    filename = os.path.join(script_dir, f"document_{doc_id}.pdf")

    if os.path.exists(filename):
        print(f"Skipping (already exists): {filename}")
        continue

    time.sleep(3)

    print(f"Downloading {url}...")
    try:
        response = requests.get(url, headers=headers, stream=True)

        if response.status_code == 200:
            with open(filename, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Saved: {filename}")
        elif response.status_code == 404:
            print(f"404 Not Found: {url}")
        else:
            print(f"Failed (Status Code {response.status_code}): {url}")

    except Exception as e:
        print(f"Error downloading {url}: {e}")

print("\nDownload process complete.")
