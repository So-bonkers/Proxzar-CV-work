from fastapi import APIRouter, HTTPException, Depends, UploadFile, File
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from datetime import datetime, timedelta
import jwt
import os
import shutil
from pathlib import Path
import logging
import requests
from app.utilsV2 import processStreamDocument, processDocument  # Adjust imports

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

router = APIRouter()  # Use APIRouter instead of FastAPI()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Base directory for storing files
BASE_STORAGE_DIR = Path("data/IngestedFiles")

# Secret key for JWT
SECRET_KEY = "your_secret_key"
ALGORITHM = "HS256"

# Dummy user database (For authentication)
fake_users_db = {
    "testuser": {"username": "testuser", "password": "password123"}
}

# Function to create JWT Token
def create_access_token(data: dict, expires_delta: timedelta):
    to_encode = data.copy()
    expire = datetime.utcnow() + expires_delta
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

# User Login API (Returns Token)
@router.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = fake_users_db.get(form_data.username)
    if not user or form_data.password != user["password"]:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    access_token = create_access_token(data={"sub": form_data.username}, expires_delta=timedelta(hours=1))
    return {"access_token": access_token, "token_type": "bearer"}

# Convert by URL (Download & Process)
@router.post("/convertLink")
async def convert_link(proxzarKeyID: str, file_link: str, token: str = Depends(oauth2_scheme)):
    """
    Download file from a link, process it, and return JSON output.
    """
    storage_dir = BASE_STORAGE_DIR / proxzarKeyID
    storage_dir.mkdir(parents=True, exist_ok=True)
    
    file_name = file_link.split("/")[-1]
    file_path = storage_dir / file_name
    
    try:
        response = requests.get(file_link)
        response.raise_for_status()
        with open(file_path, "wb") as f:
            f.write(response.content)
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"File download failed: {str(e)}")
    
    logger.info(f"Processing file {file_name} from link")
    result = processDocument(str(file_path), storage_dir, file_name)
    
    return {**result}

# Convert by File Upload
@router.post("/convertUpload")
async def convert_upload(proxzarKeyID: str, file: UploadFile = File(...), token: str = Depends(oauth2_scheme)):
    """
    Upload file, process it, and return JSON output.
    """
    storage_dir = BASE_STORAGE_DIR / proxzarKeyID
    file_name = file.filename
    file_dir = storage_dir / file_name
    file_dir.mkdir(parents=True, exist_ok=True)
    
    file_path = file_dir / file.filename
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    logger.info(f"Processing uploaded file {file_name}")
    result = processDocument(str(file_path), file_dir, file_name)
    
    return {**result}

# Convert by S3 (Fetch & Process)
@router.post("/convertS3")
async def convert_s3(proxzarKeyID: str, bucket_name: str, file_key: str, token: str = Depends(oauth2_scheme)):
    """
    Process document directly from an S3 bucket.
    """
    storage_dir = BASE_STORAGE_DIR / proxzarKeyID
    storage_dir.mkdir(parents=True, exist_ok=True)
    
    file_name = os.path.basename(file_key)
    logger.info(f"Processing S3 file {file_key} from bucket {bucket_name}")
    result = processStreamDocument(bucket_name, file_key, storage_dir, file_name)
    
    return {**result}
