from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.pipeline.simple_pipeline import SimplePipeline
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline
from docling.document_converter import (
    DocumentConverter,
    PdfFormatOption,
    WordFormatOption,
)
from docling.datamodel.base_models import ConversionStatus, InputFormat
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling_core.types.doc import PictureItem, TableItem, ImageRefMode
import time
import logging

logger = logging.getLogger(__name__)
IMAGE_RESOLUTION_SCALE = 2.0

def process_document(file_path, output_dir, global_client_id):
    """Process the document using Docling."""
    try:
        start_time = time.time()

        pipeline_options = PdfPipelineOptions()
        pipeline_options.images_scale = IMAGE_RESOLUTION_SCALE
        pipeline_options.generate_picture_images = True
        pipeline_options.generate_table_images = True

        doc_converter = DocumentConverter(
    allowed_formats=[
        InputFormat.PDF,
        InputFormat.IMAGE,
        InputFormat.DOCX,
        InputFormat.HTML,
        InputFormat.PPTX,
        InputFormat.ASCIIDOC,
        InputFormat.MD,
    ],  # Whitelist formats, non-matching files are ignored.
    format_options={
        InputFormat.PDF: PdfFormatOption(
            pipeline_cls=StandardPdfPipeline,
            backend=PyPdfiumDocumentBackend,
            pipeline_options=pipeline_options  # Include the pipeline options
        ),
        InputFormat.DOCX: WordFormatOption(
            pipeline_cls=SimplePipeline  # , backend=MsWordDocumentBackend
        ),
    }
)


        conv_result = doc_converter.convert(file_path)
        end_time = time.time()

        if conv_result.status != ConversionStatus.SUCCESS:
            logger.error(f"Failed to process {file_path}. Status: {conv_result.status}")
            return {"error": f"Failed to process {file_path}. Status: {conv_result.status}"}
        else:
            logger.info(f"Processing time: {end_time - start_time:.2f} seconds; It ended in a Success")
            
        # Save outputs (figures, tables, HTML)
        html_filename = output_dir / f"{global_client_id}-with-image-refs.html"
        logger.info(f"Saving HTML output to {html_filename}")
        conv_result.document.save_as_html(html_filename, image_mode=ImageRefMode.REFERENCED)

        return {"output_html": str(html_filename)}

    except Exception as e:
        logger.error(f"Error processing file {file_path}: {e}")
        return {"error": str(e)}
