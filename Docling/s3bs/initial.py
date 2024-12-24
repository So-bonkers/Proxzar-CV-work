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

try:
    # Initialize a paginator to handle buckets with many files
    paginator = s3_client.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=bucket_name)

    print("Files in bucket:")
    for page in pages:
        if 'Contents' in page:
            for obj in page['Contents']:
                print(obj['Key'])  # Print the file name (Key)
        else:
            print("No files found in the bucket.")
except Exception as e:
    print(f"Error: {e}")
