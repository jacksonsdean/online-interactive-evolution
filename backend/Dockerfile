FROM python:3.9
RUN apt-get update \
	&& apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
	&& rm -rf /var/lib/apt/lists/*
WORKDIR /backend
RUN PYTHONPATH=/backend
COPY . .
RUN python -m pip install -r requirements.full.txt
ENV FLASK_APP=local_server.py
CMD [ "python", "-u", "-m" ,"flask", "run", "--host=0.0.0.0"]
