# DVC — Veri Versiyonlama

Veri MinIO'da, Git'te yalnızca hangi sürüm olduğu durur. Depo düzeni ve DVC'siz
doğrudan erişim: [`DATA.md`](DATA.md).

## Depo bilgileri

| | |
|:---|:---|
| **Bucket** | `dvc-cache` — yalnızca `dvc push`/`pull` kullanır, elle dokunulmaz |
| **S3 endpoint** | `http://10.10.1.132:30900` |
| **Yazan** | kullanıcı `dvc` (`dvc-rw` policy — yalnız `dvc-cache`) |
| **Okuyan** | kullanıcı `ml-team` (`dvc-read` policy — listeleme + okuma) |

Parolalar DevOps'tan (parola kasası).

## Kurulum (bir kez)

> **Repo tarafı hazır** — `pyproject.toml`'a `[s3]` extra'sı eklendi ve `.gitignore`
> DVC pointer dosyalarının önünü açacak şekilde düzeltildi (bkz. [`DEVOPS.md`](DEVOPS.md) §11).

**1. Bağımlılıkları kur**

```bash
uv sync --extra data
```
> `pyproject.toml`'da `dvc[s3]>=3.59.0` yazıyor — `[s3]` boto3/s3fs getirir. Düz `dvc`
> ile S3'e bağlanılamaz (`dvc push` → `missing dependency: dvc-s3`).

**2. Remote'u tanımla**

```bash
dvc init

dvc remote add -d minio s3://dvc-cache
dvc remote modify minio endpointurl http://10.10.1.132:30900

dvc remote modify --local minio access_key_id dvc
dvc remote modify --local minio secret_access_key '<parola>'
```

> 🔴 **`--local` zorunlu.** Onsuz parola `.dvc/config`'e yazılır ve **Git'e commit edilir**.
> `--local` ise `.dvc/config.local`'e yazar — o dosya `.gitignore`'dadır.

**3. Doğrula ve commit et**

```bash
cat .dvc/config               # endpointurl VAR, parola YOK olmali
git add .dvc/config .dvc/.gitignore .dvcignore
git commit -m "chore(data): configure DVC remote (MinIO dvc-cache)"
```
`git status`'ta `.dvc/config.local` **görünmemeli**.

## Kullanım

**Veri ekleyen (Data squad):**
```bash
dvc add data/training
git add data/              # pointer (data/training.dvc); gercek veri zaten ignore'da
git commit -m "data: add training-512 v1"
dvc push && git push       # once veri, sonra pointer
```

> `dvc push`'ı `git push`'tan **önce** yap: pointer'ı gönderip veriyi göndermezsen
> takım arkadaşın `dvc pull`'da "hash bulunamadı" alır.
>
> `dvc add` normalde ignore kuralını `data/.gitignore`'a yazar; bu repoda üretim
> klasörleri kök `.gitignore`'da zaten kayıtlı olduğu için o dosyayı oluşturmayabilir —
> ikisi de normal.

**Veri çeken (ML ekibi):**
```bash
git pull && dvc pull
```

**Eski sürüme dönmek:**
```bash
git checkout <commit>
dvc pull
```

## Kritik notlar

- **Disk:** MinIO eğitim setiyle **aynı fiziksel diskte**. DVC cache veriyi çoğaltır —
  20 GiB'lık set DVC'ye girerse `dvc-cache` de o boyuta ulaşır. Yeni set eklemeden
  önce diski kontrol et.
- **`Access Denied` (pull, ML ekibi):** `ml-team`'de `dvc-read` yok demektir → DevOps'a söyle.
- Kurulum kaydı: [`DEVOPS.md`](DEVOPS.md) §11.
