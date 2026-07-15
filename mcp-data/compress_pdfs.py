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


def compress_single_pdf(file_path, max_dim=1500, jpeg_quality=35):
    """More aggressive compression with progress print."""

    if not HAS_PYMUPDF:
        return 0, 0

    print(f"📄 Processing: {os.path.basename(file_path)}")  # ← Print current PDF

    try:
        orig_size = os.path.getsize(file_path)
        if orig_size == 0:
            return 0, 0

        temp_output = file_path + ".tmp"
        doc = fitz.open(file_path)

        # === MORE AGGRESSIVE IMAGE COMPRESSION ===
        doc.rewrite_images(
            dpi_threshold=120,  # Only touch higher-res images
            dpi_target=72,  # Downsample aggressively
            quality=jpeg_quality,  # Lower quality (was 50)
            lossy=True,
            lossless=False,  # Skip PNGs to avoid bloat/crashes
            set_to_gray=False,  # Optional: set to True for even smaller (grayscale)
        )

        # === SUPER AGGRESSIVE SAVE ===
        doc.save(
            temp_output,
            garbage=4,  # Maximum garbage collection (dedup + stream merge)
            deflate=True,
            deflate_images=True,  # Extra image stream compression
            deflate_fonts=True,  # Compress fonts
            use_objstms=True,
            clean=True,  # Rebuild structure for better compression
            pretty=False,  # No whitespace = smaller file
        )
        doc.close()

        new_size = os.path.getsize(temp_output)

        if new_size < orig_size * 0.98:  # Only replace if meaningful gain
            os.replace(temp_output, file_path)
            return orig_size, new_size
        else:
            os.remove(temp_output)
            return orig_size, orig_size

    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")
        if os.path.exists(file_path + ".tmp"):
            os.remove(file_path + ".tmp")
        return 0, 0


def _worker(full_path):
    """Worker wrapper."""
    orig, new = compress_single_pdf(full_path)
    return full_path, orig, new


def batch_compress_directory(root_folder):
    if not HAS_PYMUPDF:
        return

    pdf_files = []
    print(f"🔍 Scanning '{root_folder}' for PDFs...")
    for dirpath, _, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.lower().endswith(".pdf"):
                pdf_files.append(os.path.join(dirpath, filename))

    file_count = len(pdf_files)
    if file_count == 0:
        print("\nNo PDF files found.")
        return

    total_orig_size = 0
    total_new_size = 0
    start_time = time.time()

    num_processes = max(1, cpu_count() - 1)
    print(f"🚀 Starting parallel compression across {num_processes} CPU cores...")

    with Pool(processes=num_processes) as pool:
        results = pool.map(_worker, pdf_files)

    for full_path, orig, new in results:
        if orig > 0:
            total_orig_size += orig
            total_new_size += new
            rel_path = os.path.relpath(full_path, root_folder)
            pct = ((orig - new) / orig * 100) if orig > 0 else 0
            print(f"  ✓ {rel_path} : Reduced by {pct:.1f}%")

    orig_mb = total_orig_size / (1024 * 1024)
    new_mb = total_new_size / (1024 * 1024)
    saved_mb = orig_mb - new_mb
    pct = (saved_mb / orig_mb) * 100 if orig_mb > 0 else 0
    elapsed = time.time() - start_time

    print("\n================ SUMMARY ================")
    print(f"Files Processed:   {file_count}")
    print(f"Time Elapsed:      {elapsed:.2f} seconds")
    print(f"Original Volume:   {orig_mb:.2f} MB")
    print(f"Compressed Volume: {new_mb:.2f} MB")
    print(f"Total Space Saved: {saved_mb:.2f} MB ({pct:.1f}% reduction)")


if __name__ == "__main__":
    target_directory = "./pdfs"
    batch_compress_directory(target_directory)
