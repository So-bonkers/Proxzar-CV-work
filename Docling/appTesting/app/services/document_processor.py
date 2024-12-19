from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import ConversionStatus, InputFormat
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
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
        )

        conv_result = doc_converter.convert(file_path)

        if conv_result.status != ConversionStatus.SUCCESS:
            logger.error(f"Failed to process {file_path}. Status: {conv_result.status}")
            return {"error": f"Failed to process {file_path}. Status: {conv_result.status}"}

        # Save outputs (figures, tables, HTML)
        html_filename = output_dir / f"{global_client_id}-with-image-refs.html"
        conv_result.document.save_as_html(html_filename, image_mode=ImageRefMode.REFERENCED)

        return {"output_html": str(html_filename)}

    except Exception as e:
        logger.error(f"Error processing file {file_path}: {e}")
        return {"error": str(e)}
