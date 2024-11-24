from kfp import dsl
from kfp import compiler

# Data Processing Component
@dsl.container_component
def data_processing(
    file_path: str,
    save_json_path: str,
    percent: int = 10,
    convert: bool = True,
    minio_endpoint: str = '',
    minio_access_key: str = '',
    minio_secret_key: str = '',
    minio_bucket: str = ''
):
    return dsl.ContainerSpec(
        image='alpine',
        command=["/app/venv/bin/python", "data_processing.py"],
        args=[
            '--file_path', file_path,
            '--save_json_path', save_json_path,
            '--percent', str(percent),
            '--convert' if convert else '--not-convert',
            '--minio_endpoint', minio_endpoint,
            '--minio_access_key', minio_access_key,
            '--minio_secret_key', minio_secret_key,
            '--minio_bucket', minio_bucket
        ]
    )

# Model Training Component
@dsl.container_component
def model_training(
    minio_endpoint: str,
    minio_access_key: str,
    minio_secret_key: str,
    minio_bucket: str,
    train_file: str,
    valid_file: str,
    save_model_path: str,
    logdir: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
):
    return dsl.ContainerSpec(
        image='tensorflow:2.9.1',
        command=["python", "model_training.py"],
        args=[
            '--minio_endpoint', minio_endpoint,
            '--minio_access_key', minio_access_key,
            '--minio_secret_key', minio_secret_key,
            '--minio_bucket', minio_bucket,
            '--train_file', train_file,
            '--valid_file', valid_file,
            '--save_model_path', save_model_path,
            '--logdir', logdir,
            '--epochs', str(epochs),
            '--batch_size', str(batch_size),
            '--learning_rate', str(learning_rate)
        ]
    )

# Model Evaluation Component
@dsl.container_component
def model_evaluation():
    return dsl.ContainerSpec(
        image='python:3.9',
        command=["python", "model_evaluation.py"]
    )

# Pipeline Definition
@dsl.pipeline(name="data-processing-and-model-training-pipeline")
def data_processing_and_model_training_pipeline(
    file_path: str,
    save_json_path: str,
    minio_endpoint: str,
    minio_access_key: str,
    minio_secret_key: str,
    minio_bucket: str,
    percent: int = 10,
    convert: bool = True,
    logdir: str = 'tb_logs',
    epochs: int = 10,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    save_model_path: str = '/tmp/model.ckpt',
):
    # Data Processing
    data_processing_step = data_processing(
        file_path=file_path,
        save_json_path=save_json_path,
        percent=percent,
        convert=convert,
        minio_endpoint=minio_endpoint,
        minio_access_key=minio_access_key,
        minio_secret_key=minio_secret_key,
        minio_bucket=minio_bucket,
    )

    # Model Training
    model_training_step = model_training(
        minio_endpoint=minio_endpoint,
        minio_access_key=minio_access_key,
        minio_secret_key=minio_secret_key,
        minio_bucket=minio_bucket,
        train_file=f"{save_json_path}/train.json",
        valid_file=f"{save_json_path}/test.json",
        save_model_path=save_model_path,
        logdir=logdir,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )

    # Model Evaluation
    model_evaluation_step = model_evaluation(
                file_path=file_path,
        save_json_path=save_json_path,
        percent=percent,
        convert=convert,
        minio_endpoint=minio_endpoint,
        minio_access_key=minio_access_key,
        minio_secret_key=minio_secret_key,
        minio_bucket=minio_bucket,
    )

# Compile Pipeline
if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=data_processing_and_model_training_pipeline,
        package_path="data_processing_and_model_training_pipeline.yaml"
    )
