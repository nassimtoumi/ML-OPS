import kfp
from kfp.dsl import pipeline, component
import os

# Define the MinIO connection component
@component(
    base_image="nassimtoumi98/data_processing:1.0.3",  # Replace with your Docker image in your registry
    packages_to_install=["minio"]
)
def minio_connection_component(
    minio_endpoint: str,
    minio_access_key: str,
    minio_secret_key: str,
    minio_bucket: str
):
    from minio import Minio
    from minio.error import S3Error

    try:
        # Initialize MinIO client
        minio_client = Minio(
            minio_endpoint,
            access_key=minio_access_key,
            secret_key=minio_secret_key,
            secure=False  # Change to True if using HTTPS
        )

        # Check if bucket exists
        if not minio_client.bucket_exists(minio_bucket):
            minio_client.make_bucket(minio_bucket)
        print("Connected to MinIO and bucket is ready!")

    except S3Error as e:
        print(f"MinIO error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

# Define the data processing component (already provided)
@component(
    base_image="nassimtoumi98/data_processing:latest",  # Replace with your Docker image in your registry
    packages_to_install=["minio", "pydub" , "ffmpeg" , "ffprobe"]
)
def data_processing_component(
    file_path: str,               # Relative path in MinIO to the .tsv file
    save_json_path: str,           # Path in MinIO to save JSON files
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

    # Initialize MinIO client
    minio_client = Minio(
        minio_endpoint,
        access_key=minio_access_key,
        secret_key=minio_secret_key,
        secure=False  # Change to True if using HTTPS
    )

    # Download the source .tsv file from MinIO
    local_file_path = "/tmp/" + os.path.basename(file_path)
    minio_client.fget_object(minio_bucket, file_path, local_file_path)
    print(f"Downloaded {file_path} from MinIO.")

    # Process the .tsv file
    data = []
    directory = os.path.dirname(local_file_path)

    with open(local_file_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t')
        total_lines = sum(1 for _ in open(local_file_path))
        index = 1
        for row in reader:
            file_name = row['path']
            text = row['sentence']
            wav_filename = file_name.rpartition('.')[0] + ".wav" if convert else file_name
            audio_path = os.path.join("/tmp", "clips", wav_filename)
            
            # Convert to WAV if needed
            if convert:
                print(f"Converting file {index}/{total_lines} to wav")
                mp3_path = os.path.join("/tmp", "clips", file_name)
                # Download mp3 file from MinIO
                minio_client.fget_object(minio_bucket, f"clips/{file_name}", mp3_path)
                sound = AudioSegment.from_mp3(mp3_path)
                sound.export(audio_path, format="wav")
            
            data.append({"key": audio_path, "text": text})
            index += 1

    # Shuffle and split data for train/test
    random.shuffle(data)
    train_data = data[: int(len(data) * (1 - percent / 100))]
    test_data = data[int(len(data) * (1 - percent / 100)) :]

    # Save JSON files for training and testing
    train_json_path = "/tmp/train.json"
    test_json_path = "/tmp/test.json"

    with open(train_json_path, "w") as f:
        for entry in train_data:
            f.write(json.dumps(entry) + "\n")
    with open(test_json_path, "w") as f:
        for entry in test_data:
            f.write(json.dumps(entry) + "\n")
            
    print("Data processing complete. Training and testing JSON files created.")

    # Upload JSON files back to MinIO
    minio_client.fput_object(minio_bucket, os.path.join(save_json_path, "train.json"), train_json_path)
    minio_client.fput_object(minio_bucket, os.path.join(save_json_path, "test.json"), test_json_path)
    print(f"JSON files uploaded to {save_json_path} in MinIO.")


# Define the training component (empty for now)
@component(
    base_image="nassimtoumi98/data_processing:latest",  # Replace with your Docker image in your registry
    packages_to_install=[]
)
def training_component(
    training_data_path: str,
    model_save_path: str
):
    print("Training the model with data from:", training_data_path)
    print("Saving model to:", model_save_path)
    # Training logic goes here

# Define the evaluation component (empty for now)
@component(
    base_image="nassimtoumi98/data_processing:latest",  # Replace with your Docker image in your registry
    packages_to_install=[]
)
def evaluation_component(
    model_path: str,
    evaluation_data_path: str
):
    print("Evaluating the model located at:", model_path)
    print("Using evaluation data from:", evaluation_data_path)
    # Evaluation logic goes here

# Define the pipeline
@pipeline(name="Enhanced Data Processing Pipeline", description="Pipeline to process data, train and evaluate model.")
def enhanced_data_processing_pipeline(
    file_path: str,
    save_json_path: str,
    minio_endpoint: str,
    minio_access_key: str,
    minio_secret_key: str,
    minio_bucket: str,
    model_save_path: str
):
    # Step 1: Establish MinIO Connection
    minio_conn_task = minio_connection_component(
        minio_endpoint=minio_endpoint,
        minio_access_key=minio_access_key,
        minio_secret_key=minio_secret_key,
        minio_bucket=minio_bucket
    )
    
    # Step 2: Process Data
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
    data_processing_task.after(minio_conn_task)
    
    # Step 3: Train Model
    training_task = training_component(
        training_data_path=str(save_json_path) + "/train.json",
        model_save_path=model_save_path
    )
    training_task.after(data_processing_task)
    
    # Step 4: Evaluate Model
    evaluation_task = evaluation_component(
        model_path=model_save_path,
        evaluation_data_path=str(save_json_path) + "/test.json"
    )
    evaluation_task.after(training_task)

# Main block to run the pipeline
if __name__ == "__main__":
    from kfp import Client
    client = Client()
    client.create_run_from_pipeline_func(enhanced_data_processing_pipeline, arguments={
        "file_path": "/path/in/minio/file.tsv",  # Replace with your actual file path
        "save_json_path": "/path/in/minio/save",  # Replace with your actual save path
        "minio_endpoint": "your-minio-endpoint",  # Replace with your Minio endpoint
        "minio_access_key": "your-minio-access-key",  # Replace with your Minio access key
        "minio_secret_key": "your-minio-secret-key",  # Replace with your Minio secret key
        "minio_bucket": "your-minio-bucket",  # Replace with your Minio bucket name
        "model_save_path": "/path/to/save/model"  # Replace with the path to save the model
    })
