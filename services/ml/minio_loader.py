import boto3
import numpy as np
import io
import os
from dotenv import load_dotenv

load_dotenv()

def load_npy_from_minio(bucket_name, file_key):
    endpoint = os.getenv("MINIO_ENDPOINT")
    access_key = os.getenv("MINIO_ACCESS_KEY")
    secret_key = os.getenv("MINIO_SECRET_KEY")
    
    s3_client = boto3.client(
        's3',
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key
    )
    
    try:
        response = s3_client.get_object(Bucket=bucket_name, Key=file_key)
        
        file_stream = response['Body'].read()
        
        image_array = np.load(io.BytesIO(file_stream))
        
        print(f"Görüntü başarıyla yüklendi! Boyutları: {image_array.shape}")
        return image_array
        
    except Exception as e:
        print(f"Hata oluştu: {e}")
        return None
