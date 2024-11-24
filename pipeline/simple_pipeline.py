from kfp import dsl
from kfp import compiler

@dsl.container_component
def say_hello1():
    return dsl.ContainerSpec(
        image='alpine',
        command=['echo'],
        args=['Hellhfhfo']
    )

@dsl.pipeline(name="hello-world-pipeline")
def hello_pipeline():
    say_hello1()

# Compile the pipeline
if __name__ == "__main__":
    compiler.Compiler().compile(pipeline_func=hello_pipeline, package_path="hello_pipeline.yaml")
