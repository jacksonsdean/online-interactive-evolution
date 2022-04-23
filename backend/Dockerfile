FROM python:3.9
WORKDIR /backend
RUN PYTHONPATH=/backend
COPY . .
RUN python -m pip install -r requirements.txt
ENV FLASK_APP=local_test.py
CMD [ "python", "-u", "-m" ,"flask", "run", "--host=0.0.0.0"]
