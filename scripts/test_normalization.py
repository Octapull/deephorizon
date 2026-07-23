import numpy as np
from services.ml.minio_loader import list_files_in_minio, load_npy_from_minio

def test_normalization(bucket_name="blackhole", prefix_path="datasets/training-512/v1/clean/", num_samples=5):
    print(f" '{bucket_name}' bucket'ındaki '{prefix_path}' dizini kontrol ediliyor...\n")
    
    files = list_files_in_minio(bucket_name, prefix_path)

    if not files:
        print(" Dosya bulunamadı! Lütfen MinIO bağlantınızı, bucket adını ve prefix_path'i kontrol edin.")
        return

    print(f" Toplam {len(files)} dosya bulundu. İlk {num_samples} dosya test ediliyor...\n")

    for i, file_key in enumerate(files[:num_samples]):
        data = load_npy_from_minio(bucket_name, file_key)
        
        if data is not None:
            min_val = np.min(data)
            max_val = np.max(data)

            is_normalized = (min_val >= 0.0) and (max_val <= 1.0)
            
            print(f"📄 Dosya {i+1}: {file_key}")
            print(f"   -> Veri Boyutu: {data.shape}")
            print(f"   -> Min Değer: {min_val:.6f}, Max Değer: {max_val:.6f}")
            
            if is_normalized:
                print(" SONUÇ: Başarılı! Dosya 0-1 aralığında normalize edilmiş.")
            else:
                print(" SONUÇ: Hatalı! Dosya 0-1 aralığında DEĞİL.")
        else:
            print(f" Dosya {i+1}: {file_key} yüklenirken hata oluştu.")
        
        print("-" * 60)

if __name__ == "__main__":
    test_normalization(bucket_name="blackhole", prefix_path="datasets/training-512/v1/clean/", num_samples=3)
    
    print("\n" + "="*60 + "\n")
    
    test_normalization(bucket_name="blackhole", prefix_path="datasets/training-512/v1/degraded/", num_samples=3)