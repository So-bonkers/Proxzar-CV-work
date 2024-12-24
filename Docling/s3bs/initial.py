import boto3
import json

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

try:
    # Get the object
    response = s3_client.get_object(Bucket=bucket_name, Key=file_key)
    
    # Stream the file content
    with response['Body'] as stream:
        for chunk in iter(lambda: stream.read(1024), b""):  # Read in 1 KB chunks
            print(chunk)  # Process the chunk (for binary files) or decode for text
except Exception as e:
    print(f"Error: {e}")
