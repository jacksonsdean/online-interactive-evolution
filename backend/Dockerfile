FROM python:3.9
WORKDIR /backend
COPY . .
RUN python -m pip install -r requirements.txt
RUN FLASK_APP=local_test.py flask run
EXPOSE 5000