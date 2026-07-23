import boto3
import numpy as np
import io
import os
from dotenv import load_dotenv

load_dotenv()

def get_s3_client():
    return boto3.client(
        's3',
        endpoint_url=os.getenv("MINIO_ENDPOINT"),
        aws_access_key_id=os.getenv("MINIO_ACCESS_KEY"),
        aws_secret_access_key=os.getenv("MINIO_SECRET_KEY")
    )

def list_files_in_minio(bucket_name, prefix_path):
    s3_client = get_s3_client()
    file_keys = []
    
    try:
        response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix_path)
        
        if 'Contents' in response:
            for obj in response['Contents']:
                # Sadece .npy uzantılı dosyaları listeye ekle
                if obj['Key'].endswith('.npy'):
                    file_keys.append(obj['Key'])
                    
        return sorted(file_keys)
    except Exception as e:
        print(f"Hata - Dosyalar listelenemedi: {e}")
        return []

def load_npy_from_minio(bucket_name, file_key):
    s3_client = get_s3_client()
    try:
        response = s3_client.get_object(Bucket=bucket_name, Key=file_key)
        file_stream = response['Body'].read()
        image_array = np.load(io.BytesIO(file_stream))
        
        return image_array
    except Exception as e:
        print(f"Hata oluştu ({file_key}): {e}")
        return None