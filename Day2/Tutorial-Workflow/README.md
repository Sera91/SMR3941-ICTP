# Common Workflow Language Tutorial

For this tutorial, the students are asked to create a simple CWL orkflow that executes these two commands in sequence:

```bash
tar -xf ${archive} ${name_of_file_to_extract} -> ${name_of_file_to_extract}
javac -d . ${java_file} -> ${class_file}
```

These commands extract a `*.java` file from a `*.tar` archive and compile it into a `*.class` file.

The skeleton of the workflow is already provided:

 - The `main.cwl` file should contain the workflow description
 - The `tar-param.cwl` and `arguments.cwl` files should execute the two commands stated above
 - The `config.yml` should describe the workflow input parameters

The input file of the workflow is called `data/hello.tar`, while the name of the file that should be extracted is `Hello.java`.

Since the execution environment does not provide the `javac` executable, the second command should be executed inside a Docker image. You can use the `openjdk:9.0.1-11-slim` image.

The workflow can then be executed using `cwltool`, the CWL reference implementation. To execute the workflow, run the following commands:

```bash
python -m venv cwl-venv
source cwl-venv/bin/activate
pip install -r requirements.txt

cwl-runner main.cwl config.yml
```