import os
import time

import requests

chancellors = {
    "wir": 409,
    "cun": 250,
    "str": 282,
    "ma1": 388,
    "ma3": 476,
    "lut": 364,
    "bru": 774,
}

base_url = "https://aktenreichskanzlei.bundesarchiv.de/resources/pdf/{}-D{}.pdf"
base_output_dir = "downloaded_pdfs"


def download_pdf(prefix, identifier, output_dir):
    url = base_url.format(prefix, identifier)
    filename = os.path.join(output_dir, f"{prefix}-D{identifier}.pdf")

    print(f"Downloading {url}...")
    try:
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(filename, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Saved: {filename}")
            return True
        elif response.status_code == 404:
            print(f"404 Not Found: {url}")
            return False
        else:
            print(f"Failed (Status Code {response.status_code}): {url}")
            return False
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False


for prefix, max_range in chancellors.items():
    prefix_dir = os.path.join(base_output_dir, prefix)
    os.makedirs(prefix_dir, exist_ok=True)

    print(f"\n--- Starting downloads for prefix: {prefix} (1 to {max_range}) ---")

    for i in range(1, max_range + 1):
        success = download_pdf(prefix, str(i), prefix_dir)
        time.sleep(3)

        if not success:
            print(
                f"Base file {prefix}-D{i}.pdf missing. Checking suffixes 'a' and 'b'..."
            )

            for suffix in ["a", "b"]:
                download_pdf(prefix, f"{i}{suffix}", prefix_dir)
                time.sleep(1)

print("\nDownload process complete.")
