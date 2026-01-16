import os
import markdown
from xhtml2pdf import pisa

def convert_md_to_pdf(source_dir: str):
    """
    Converts all .md files in source_dir to .pdf files.
    """
    print(f"Checking for Markdown files in: {source_dir}")
    
    if not os.path.exists(source_dir):
        print(f"Directory {source_dir} does not exist.")
        return

    md_files = [f for f in os.listdir(source_dir) if f.lower().endswith('.md')]
    
    if not md_files:
        print("No .md files found.")
        return

    print(f"Found {len(md_files)} Markdown files. Converting...")

    for md_file in md_files:
        md_path = os.path.join(source_dir, md_file)
        pdf_file = os.path.splitext(md_file)[0] + ".pdf"
        pdf_path = os.path.join(source_dir, pdf_file)

        print(f"Converting: {md_file} -> {pdf_file}")
        
        try:
            # 1. Read Markdown
            with open(md_path, "r", encoding="utf-8") as f:
                text = f.read()
            
            # 2. Convert to HTML
            html = markdown.markdown(text)
            
            # 3. Create PDF
            with open(pdf_path, "wb") as pdf_file:
                pisa_status = pisa.CreatePDF(html, dest=pdf_file)
            
            if pisa_status.err:
                print(f"❌ Error creating PDF for {md_file}")
            else:
                print(f"✅ Created {pdf_path}")
                # Optional: Remove original .md file
                os.remove(md_path)
                
        except Exception as e:
            print(f"❌ Exception converting {md_file}: {e}")

if __name__ == "__main__":
    convert_md_to_pdf("data")
