import kfp
from kfp.dsl import pipeline, component
import os

# Define the component
@component(
    base_image="your-registry/your-image-name",  # Replace with your Docker image in your registry
    packages_to_install=["minio", "pydub"]
)
def data_processing_component(
    file_path: str,
    save_json_path: str,
    percent: int = 10,
    convert: bool = True,
    minio_endpoint: str = "",
    minio_access_key: str = "",
    minio_secret_key: str = "",
    minio_bucket: str = ""
):
    import json
    import random
    import csv
    from pydub import AudioSegment
    from minio import Minio
    from minio.error import S3Error
    import os

    try:
        # Initialize MinIO client
        minio_client = Minio(
            minio_endpoint,
            access_key=minio_access_key,
            secret_key=minio_secret_key,
            secure=False  # Change to True if using HTTPS
        )
        
        # Fetch the source file from MinIO
        local_file = '/tmp/source_file.tsv'
        minio_client.fget_object(minio_bucket, file_path, local_file)

        data = []
        directory = os.path.dirname(local_file)
        
        with open(local_file, newline='') as csvfile: 
            reader = csv.DictReader(csvfile, delimiter='\t')
            index = 1
            for row in reader:  
                file_name = row['path']
                filename = file_name.rpartition('.')[0] + ".wav"
                text = row['sentence']
                
                if convert:
                    data.append({
                        "key": f"{directory}/clips/{filename}",
                        "text": text
                    })
                    print(f"Converting file {index} to wav", end="\r")
                    src = f"{directory}/clips/{file_name}"
                    dst = f"{directory}/clips/{filename}"
                    sound = AudioSegment.from_mp3(src)
                    sound.export(dst, format="wav")
                    index += 1
                else:
                    data.append({
                        "key": f"{directory}/clips/{file_name}",
                        "text": text
                    })

        random.shuffle(data)
        
        # Creating JSONs for train and test sets
        train_json_path = os.path.join(save_json_path, "train.json")
        test_json_path = os.path.join(save_json_path, "test.json")

        with open(train_json_path, 'w') as f:
            for i in range(len(data) - (len(data) // percent)):
                f.write(json.dumps(data[i]) + "\n")

        with open(test_json_path, 'w') as f:
            for i in range(len(data) - (len(data) // percent), len(data)):
                f.write(json.dumps(data[i]) + "\n")

        # Upload JSON files back to MinIO
        minio_client.fput_object(minio_bucket, "train/train.json", train_json_path)
        minio_client.fput_object(minio_bucket, "test/test.json", test_json_path)

        print("Done with processing and uploading JSON files!")

    except S3Error as e:
        print(f"MinIO error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

# Define the pipeline
@pipeline(name="Data Processing Pipeline", description="Pipeline to process data and create JSON files.")
def data_processing_pipeline(file_path: str, save_json_path: str, minio_endpoint: str, minio_access_key: str, minio_secret_key: str, minio_bucket: str):
    # Create a task by calling the component
    data_processing_task = data_processing_component(
        file_path=file_path,
        save_json_path=save_json_path,
        percent=10,
        convert=True,
        minio_endpoint=minio_endpoint,
        minio_access_key=minio_access_key,
        minio_secret_key=minio_secret_key,
        minio_bucket=minio_bucket
    )

# Main block to run the pipeline
if __name__ == "__main__":
    from kfp import Client
    client = Client()
    client.create_run_from_pipeline_func(data_processing_pipeline, arguments={
        "file_path": "/path/in/minio/file.tsv",  # Replace with your actual file path
        "save_json_path": "/path/in/minio/save",  # Replace with your actual save path
        "minio_endpoint": "your-minio-endpoint",  # Replace with your Minio endpoint
        "minio_access_key": "your-minio-access-key",  # Replace with your Minio access key
        "minio_secret_key": "your-minio-secret-key",  # Replace with your Minio secret key
        "minio_bucket": "your-minio-bucket"  # Replace with your Minio bucket name
    })
