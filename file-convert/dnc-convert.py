import fitz  # PyMuPDF

# Path to your input PDF
dnc_file_name = "d&c-jspapers-2025-06-08 923am"
pdf_path = f"../texts/{dnc_file_name}.pdf"

# Open the PDF
doc = fitz.open(pdf_path)

# Collect text from all pages
all_text = ""
for page_number, page in enumerate(doc):
    text = page.get_text()
    all_text += f"\n\n--- Page {page_number + 1} ---\n{text}"

# Save to a plain text file
with open(f"../texts/{dnc_file_name}.txt", "w", encoding="utf-8") as f:
    f.write(all_text)

print(f"✅ Text extraction complete. Saved to '{dnc_file_name}.txt'")

