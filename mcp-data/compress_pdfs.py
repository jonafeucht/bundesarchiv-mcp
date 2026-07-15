import os
import time
from multiprocessing import Pool, cpu_count

try:
    import fitz

    HAS_PYMUPDF = True
    fitz.TOOLS.mupdf_display_errors(False)
except ImportError:
    HAS_PYMUPDF = False
    print("❌ PyMuPDF (fitz) is required. Run: pip install pymupdf")

MANIFEST_FILE = ".compressed_manifest.txt"

# === SAFETY FILTERS ===
MAX_FILE_SIZE_MB = 150  # Skip files larger than 150MB to prevent memory exhaustion
MAX_PAGES = 500  # Skip files with more than 500 pages (high processing time)
MAX_VECTOR_PATHS_PER_PAGE = (
    5000  # Skip CAD drawings/blueprints with massive vector lines
)


def load_manifest():
    if os.path.exists(MANIFEST_FILE):
        with open(MANIFEST_FILE, "r", encoding="utf-8") as f:
            return set(line.strip() for line in f if line.strip())
    return set()


def mark_as_processed(file_path):
    with open(MANIFEST_FILE, "a", encoding="utf-8") as f:
        f.write(file_path + "\n")


def is_high_risk_file(file_path):
    try:
        size_mb = os.path.getsize(file_path) / (1024 * 1024)
        if size_mb > MAX_FILE_SIZE_MB:
            return (
                True,
                f"File is too large ({size_mb:.1f}MB > {MAX_FILE_SIZE_MB}MB limit)",
            )

        doc = fitz.open(file_path)
        page_count = len(doc)

        if page_count > MAX_PAGES:
            doc.close()
            return True, f"Too many pages ({page_count} pages > {MAX_PAGES} limit)"

        pages_to_test = min(5, page_count)
        for i in range(pages_to_test):
            page = doc[i]
            paths_count = len(page.get_drawings())
            if paths_count > MAX_VECTOR_PATHS_PER_PAGE:
                doc.close()
                return (
                    True,
                    f"Extreme vector complexity detected on page {i+1} ({paths_count} paths)",
                )

        doc.close()
        return False, ""
    except Exception as e:
        return True, f"Corrupted or unreadable PDF: {e}"


def compress_single_pdf(file_path, max_dim=1500, jpeg_quality=35):
    if not HAS_PYMUPDF:
        return 0, 0, "No PyMuPDF"

    is_risk, reason = is_high_risk_file(file_path)
    if is_risk:
        return 0, 0, f"SKIPPED: {reason}"

    try:
        orig_size = os.path.getsize(file_path)
        temp_output = file_path + ".tmp"
        doc = fitz.open(file_path)

        doc.rewrite_images(
            dpi_threshold=120,
            dpi_target=72,
            quality=jpeg_quality,
            lossy=True,
            lossless=False,
            set_to_gray=False,
        )

        doc.save(
            temp_output,
            garbage=4,
            deflate=True,
            deflate_images=True,
            deflate_fonts=True,
            use_objstms=True,
            clean=True,
            pretty=False,
        )
        doc.close()

        new_size = os.path.getsize(temp_output)

        if new_size < orig_size * 0.98:
            os.replace(temp_output, file_path)
            return orig_size, new_size, "SUCCESS"
        else:
            if os.path.exists(temp_output):
                os.remove(temp_output)
            return orig_size, orig_size, "NO_GAIN"

    except Exception as e:
        if os.path.exists(file_path + ".tmp"):
            os.remove(file_path + ".tmp")
        return 0, 0, f"FAILED: {e}"


def _worker(full_path):
    orig, new, status = compress_single_pdf(full_path)
    return full_path, orig, new, status


def batch_compress_directory(root_folder):
    if not HAS_PYMUPDF:
        return

    raw_pdf_files = []
    print(f"🔍 Scanning '{root_folder}' for PDFs...")
    for dirpath, _, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.lower().endswith(".pdf"):
                raw_pdf_files.append(os.path.abspath(os.path.join(dirpath, filename)))

    processed_files = load_manifest()
    pdf_files = [f for f in raw_pdf_files if f not in processed_files]

    skipped_count = len(raw_pdf_files) - len(pdf_files)
    if skipped_count > 0:
        print(
            f"♻️  Found {skipped_count} already handled files. Resuming and skipping them!"
        )

    file_count = len(pdf_files)
    if file_count == 0:
        print("\nAll files are processed or skipped.")
        return

    total_orig_size = 0
    total_new_size = 0
    skipped_files = 0
    start_time = time.time()

    num_processes = max(1, cpu_count() - 1)
    print(f"🚀 Starting parallel execution across {num_processes} CPU cores...")

    with Pool(processes=num_processes) as pool:
        iterator = pool.imap_unordered(_worker, pdf_files)
        completed = 0

        while True:
            try:
                full_path, orig, new, status = iterator.next(timeout=90)
                completed += 1
                filename = os.path.basename(full_path)

                if "SKIPPED" in status or "FAILED" in status:
                    print(f"[{completed}/{file_count}] ⚠️  {filename} -> {status}")
                    skipped_files += 1
                else:
                    total_orig_size += orig
                    total_new_size += new
                    pct = ((orig - new) / orig * 100) if orig > 0 else 0
                    print(
                        f"[{completed}/{file_count}] ✓ {filename} : Reduced by {pct:.1f}%"
                    )

                mark_as_processed(full_path)

            except StopIteration:
                break
            except Exception as e:
                print(f"\n⚠️  A process worker hung or crashed: {e}. Moving forward...")
                continue

    # 4. Process Summary
    orig_mb = total_orig_size / (1024 * 1024)
    new_mb = total_new_size / (1024 * 1024)
    saved_mb = orig_mb - new_mb
    pct = (saved_mb / orig_mb) * 100 if orig_mb > 0 else 0
    elapsed = time.time() - start_time

    print("\n================ SUMMARY ================")
    print(f"Completed processing: {completed} files")
    print(f"Skipped/Failed files: {skipped_files}")
    print(f"Time Elapsed:         {elapsed:.2f} seconds")
    print(f"Space Saved:          {saved_mb:.2f} MB ({pct:.1f}% reduction)")


if __name__ == "__main__":
    target_directory = "./pdfs"
    batch_compress_directory(target_directory)
