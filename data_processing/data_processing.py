import argparse
import json
import random
import csv
import os
from minio import Minio
from minio.error import S3Error
from pydub import AudioSegment


def main(args):
    # Initialize MinIO client with command-line arguments
    minio_client = Minio(
        args.minio_endpoint,
        access_key=args.minio_access_key,
        secret_key=args.minio_secret_key,
        secure=False  # Change to True if using HTTPS
    )
    
    # Fetch the source file from MinIO
    def download_folder(minio_bucket, folder_name, local_folder):
        # Create local directory if it does not exist
        os.makedirs(local_folder, exist_ok=True)

        # List all files in the folder on MinIO
        objects = minio_client.list_objects(minio_bucket, prefix=folder_name, recursive=True)
        for obj in objects:
            # Set local path
            local_file_path = os.path.join(local_folder, os.path.relpath(obj.object_name, folder_name))
            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
            
            # Download each file
            minio_client.fget_object(minio_bucket, obj.object_name, local_file_path)
            print(f"Downloaded {obj.object_name} to {local_file_path}")

    # Usage
    file_path = args.file_path
    local_file = '/tmp/source_file.tsv'
    minio_bucket = args.minio_bucket

    # Download single file
    minio_client.fget_object(minio_bucket, file_path, local_file)

    # Download the clips folder
    download_folder(minio_bucket, 'clips', '/tmp/clips')    
    data = []
    directory = os.path.dirname(local_file)
    percent = args.percent
    
    with open(local_file, newline='') as csvfile: 
        reader = csv.DictReader(csvfile, delimiter='\t')
        index = 1
        if(args.convert):
            print(str(len(data)) + " files found")
        for row in reader:  
            file_name = row['path']
            filename = file_name.rpartition('.')[0] + ".wav"
            text = row['sentence']
            if(args.convert):
                data.append({
                    "key": f"{directory}/clips/{filename}",
                    "text": text
                })
                print(f"Converting file {index} to wav", end="\r")
                src = f"{directory}/clips/{file_name}"
                dst = f"{directory}/clips/{filename}"
                sound = AudioSegment.from_mp3(src)
                sound.export(dst, format="wav")
                minio_client.fput_object("waves", filename, f"{directory}/clips/{filename}")
                index += 1
            else:
                data.append({
                    "key": f"{directory}/clips/{file_name}",
                    "text": text
                })

    random.shuffle(data)
    print("Creating JSON's")
    train_json_path = f"{directory}/train.json"
    test_json_path = f"{directory}/test.json"

    # Write train.json
    with open(train_json_path, 'w') as f:
        for i in range(len(data) - (len(data) // percent)):
            f.write(json.dumps(data[i]) + "\n")

    # Write test.json
    with open(test_json_path, 'w') as f:
        for i in range(len(data) - (len(data) // percent), len(data)):
            f.write(json.dumps(data[i]) + "\n")

    # Upload the JSON files back to MinIO
    minio_client.fput_object(args.minio_bucket, "train.json", train_json_path)
    minio_client.fput_object(args.minio_bucket, "test.json", test_json_path)


    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="""
    Utility script to convert commonvoice into wav and create the training and test json files for speech recognition."""
    )
    parser.add_argument('--file_path', type=str, required=True,
                        help='Path to one of the .tsv files found in cv-corpus')
    parser.add_argument('--save_json_path', type=str, required=True,
                        help='Path to the directory where the JSON files will be saved')
    parser.add_argument('--percent', type=int, default=10, required=False,
                        help='Percent of clips put into test.json instead of train.json')
    parser.add_argument('--convert', default=True, action='store_true',
                        help='Convert mp3 to wav')
    parser.add_argument('--not-convert', dest='convert', action='store_false',
                        help='Do not convert mp3 to wav')

    # Add MinIO arguments
    parser.add_argument('--minio_endpoint', type=str, required=True,
                        help='MinIO server endpoint')
    parser.add_argument('--minio_access_key', type=str, required=True,
                        help='MinIO access key')
    parser.add_argument('--minio_secret_key', type=str, required=True,
                        help='MinIO secret key')
    parser.add_argument('--minio_bucket', type=str, required=True,
                        help='MinIO bucket name')

    args = parser.parse_args()
    main(args)
