from kfp import dsl
from kfp import compiler

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
        command=["/app/venv/bin/python", "data_processing.py"],  # Adjust if the script entry point differs
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

@dsl.pipeline(name="data-processing-pipeline")
def data_processing_pipeline(
    file_path: str,
    save_json_path: str,
    percent: int = 10,
    convert: bool = True,
    minio_endpoint: str = '',
    minio_access_key: str = '',
    minio_secret_key: str = '',
    minio_bucket: str = ''
):
    data_processing(
        file_path=file_path,
        save_json_path=save_json_path,
        percent=percent,
        convert=convert,
        minio_endpoint=minio_endpoint,
        minio_access_key=minio_access_key,
        minio_secret_key=minio_secret_key,
        minio_bucket=minio_bucket
    )

# Compile the pipeline
if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=data_processing_pipeline,
        package_path="data_processing_pipeline.yaml"
    )
