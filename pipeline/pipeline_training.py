from kfp import dsl
from kfp import compiler

# Composant de prétraitement des données (fourni par l'utilisateur)
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
        image='nassimtoumi98/data_processing:2.0.1',
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

# Composant d'entraînement du modèle
@dsl.container_component
def model_training(
    minio_endpoint: str,
    minio_access_key: str,
    minio_secret_key: str,
    minio_bucket: str,
    train_file: str,
    valid_file: str,
    save_model_path: str,
    logdir: str = 'tb_logs',
    epochs: int = 10,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    load_model_from: str = None,  # Set default to None
):
    args = [
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
        '--learning_rate', str(learning_rate),
    ]
    
    # Only include load_model_from if it’s not None
    if load_model_from:
        args.extend(['--load_model_from', load_model_from])

    return dsl.ContainerSpec(
        image='nassimtoumi98/model_training:3.0.6',  # Remplacez par votre image Docker
        command=["python", "train.py"],
        args=args
    )

# Pipeline de traitement des données et d'entraînement
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
    # Étape de prétraitement des données
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

    # Étape d'entraînement du modèle
    model_training(
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


# Compilation du pipeline
if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=data_processing_and_model_training_pipeline,
        package_path="data_processing_and_model_training_pipeline.yaml"
    )
