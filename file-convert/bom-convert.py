import fitz  # PyMuPDF

# Path to your input PDF
bom_file_name = "BOM-1830-2025-06-07 0737PM"
pdf_path = f"../texts/{bom_file_name}.pdf"

# Open the PDF
doc = fitz.open(pdf_path)

# Collect text from all pages
all_text = ""
for page_number, page in enumerate(doc):
    text = page.get_text()
    all_text += f"\n\n--- Page {page_number + 1} ---\n{text}"

# Save to a plain text file
with open(f"../texts/{bom_file_name}.txt", "w", encoding="utf-8") as f:
    f.write(all_text)

print(f"✅ Text extraction complete. Saved to '{bom_file_name}.txt'")

