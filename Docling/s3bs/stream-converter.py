import boto3
from io import BytesIO
import json
import time
from docling.datamodel.base_models import DocumentStream
from docling.document_converter import DocumentConverter

CONFIG_FILE = "aws_access_config.json"
with open(CONFIG_FILE) as f:
    config = json.load(f)
    
access_key_id = config['Access-key']
secret_access_key = config['Secret-key']
# Replace with your temporary credentials
s3_client = boto3.client(
    's3',
    aws_access_key_id=access_key_id,
    aws_secret_access_key=secret_access_key,
    region_name='us-east-2'
)

# Specify the bucket name
bucket_name = 'shubankar'
file_key = '2412.13195v1.pdf'
print("No errors before the try block")

try:
    # Stream the file from S3
    print("retrieving the file using the get_object method")
    response = s3_client.get_object(Bucket=bucket_name, Key=file_key)
    print("Reading the file content into memory")
    binary_stream = response['Body'].read()  # Read the file content into memory

    # Use the binary stream with docling
    print("Creating a document stream")
    buf = BytesIO(binary_stream)
    source = DocumentStream(name="my_doc.pdf", stream=buf)
    print("Initializing the document converter")
    converter = DocumentConverter()

    # Convert the document
    start_time = time.time()
    print("Converting document now...")
    print("Current time: ", time.strftime('%H:%M:%S', time.localtime(start_time)))
    result = converter.convert(source)

    # Save or handle the result
    print("Conversion successful!")
    print("Time taken: ", time.time() - start_time)
    with open("conversion_result.txt", "w", encoding="utf-8") as f:
        f.write(str(result))  # Convert to string if necessary
    print("Result saved to conversion_result.txt")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
