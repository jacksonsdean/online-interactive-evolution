FROM python:3.9
WORKDIR /backend
RUN PYTHONPATH=/backend
COPY . .
ENV REACT_APP_NODE_ENV="local"
RUN python -m pip install -r requirements.txt
ENV FLASK_APP=local_test.py
CMD [ "python", "-m" , "flask", "run", "--host=0.0.0.0"]
