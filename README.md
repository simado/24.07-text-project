## Start

1. Install poetry if not installed by following [official instructions](https://python-poetry.org/docs/)
2. Run `poetry install --with dev,inference` and poetry will take care of everything for you.
3. When running notebooks, select the poetry's kernel

## Known Issue

| Dependency | Link |
| --- | --- |
| `numpy<2` | [link](https://stackoverflow.com/questions/78650222/valueerror-numpy-dtype-size-changed-may-indicate-binary-incompatibility-expec) |

## Poetry

Poetry is the main package manager for reproducability of ML project. Main commands are:

| Command | Description |
| --- | --- | 
| `poetry init` | creates the pyproject.toml file |
| `poetry add seaborn@* --group dev` | adds seaborn any version to the dev group |
| `poetry add pydantic^2` | adds pydantic with major version **2** to the main dependencies |
| `poetry install --with dev,inference` | Install main dependencies and additional groups |
| `poetry show` | Show the list of libraries added and their versions |

Read more at the documentation page [here](https://python-poetry.org/docs/cli/)


## Docker

Building the image

---

```bash
docker build --pull --rm -f "Dockerfile" -t project:latest "." 
```


Running the image/Serving the fastapi app

---

```bash
docker run -d -p 80:80 project:latest
```