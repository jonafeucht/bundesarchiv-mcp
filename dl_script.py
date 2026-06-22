import os
import time

import requests

chancellors = {
    "arh01": 384,
    "arh02": 286,
    "arh03": 212,
    "arh04": 208,
    "arh05": 287,
    "arh06": 222,
}

base_url = "https://aktenreichskanzlei.bundesarchiv.de/resources/pdf/{}d{}.pdf"
base_output_dir = "downloaded_pdfs"


def download_pdf(prefix, identifier, output_dir):
    url = base_url.format(prefix, identifier)
    filename = os.path.join(output_dir, f"{prefix}d{identifier}.pdf")

    if os.path.exists(filename):
        print(f"Skipping (already exists): {filename}")
        return True

    time.sleep(1)

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
        identifier = str(i).zfill(3)
        success = download_pdf(prefix, identifier, prefix_dir)

        if not success:
            print(
                f"Base file {prefix}-D{identifier}.pdf missing. Checking suffixes 'a' and 'b'..."
            )

            for suffix in ["a", "b"]:
                download_pdf(prefix, f"{identifier}{suffix}", prefix_dir)
                time.sleep(1)

print("\nDownload process complete.")
