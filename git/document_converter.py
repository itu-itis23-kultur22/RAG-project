import time
from pathlib import Path
from pypdf import PdfReader

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.accelerator_options import AcceleratorOptions, AcceleratorDevice

from llama_index.core import Document
from llama_index.core.node_parser import HierarchicalNodeParser

print("All docl imports done")


def process_document_for_rag(file_path: str):
    # --- CACHING LOGIC START ---
    cache_dir = Path("cached_docs")
    cache_dir.mkdir(exist_ok=True)

    # Use a .md extension for the cache path
    md_cache_path = cache_dir / f"{Path(file_path).stem}.md"

    if md_cache_path.exists():
        print(f"Found cached Markdown at {md_cache_path}. Loading...")
        with open(md_cache_path, "r", encoding="utf-8") as f:
            md_text = f.read()
    else:
        # 1. Configure Pipeline
        pipeline_options = PdfPipelineOptions()

        pipeline_options.do_ocr = True

        # Configure Table Extraction
        pipeline_options.do_table_structure = True

        pipeline_options.do_formula_enrichment = False

        # GPU Acceleration
        pipeline_options.accelerator_options = AcceleratorOptions(
            device=AcceleratorDevice.CUDA
        )

        # 2. Initialize the converter
        doc_converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options
                )
            }
        )

        print(f"No cache found. Converting {file_path} on GPU...")
        start = time.time()

        # --- BATCH PROCESSING LOGIC ---
        reader = PdfReader(file_path)
        total_pages = len(reader.pages)
        batch_size = 20 # documents are processed in batches of 20 pages because of GPU memory constraints
        doc_parts = []

        for start_page in range(1, total_pages + 1, batch_size):
            end_page = min(start_page + batch_size - 1, total_pages)
            print(f"Processing pages {start_page} to {end_page} of {total_pages}...")

            conv_result = doc_converter.convert(file_path, page_range=(start_page, end_page))
            doc_parts.append(conv_result.document)

            print("processed until page", end_page, "in", time.time() - start, "seconds")

        # 3. Merge the sub-documents
        print("Merging document parts...")
        doc = type(doc_parts[0]).concatenate(doc_parts)

        # Export to Markdown directly
        md_text = doc.export_to_markdown()

        # --- SAVE MARKDOWN TO CACHE ---
        print(f"Saving Markdown to {md_cache_path} for future use...")
        with open(md_cache_path, "w", encoding="utf-8") as f:
            f.write(md_text)

    # 4. Apply Hierarchical Chunking
    print("Chunking document using LlamaIndex HierarchicalNodeParser...")

    llama_doc = Document(text=md_text, metadata={"source_file": file_path})

    # Use the 3 lean levels with 0 overlap
    chunker = HierarchicalNodeParser.from_defaults(
        chunk_sizes=[1024, 512, 256],
        chunk_overlap=0
    )

    chunks = chunker.get_nodes_from_documents([llama_doc])

    return chunks