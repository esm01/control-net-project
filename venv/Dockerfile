FROM python:3.10

WORKDIR /code

RUN python -m venv /code/venv

RUN . /code/venv/bin/activate

COPY ./requirements.txt /code/venv/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/venv/requirements.txt

RUN apt-get update && apt-get install -y libgl1-mesa-glx

COPY ./models /code/venv/models

COPY ./src /code/venv/src

EXPOSE 80

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "80", "--app-dir", "venv"]