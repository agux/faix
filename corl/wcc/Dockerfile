FROM tensorflow/tensorflow:2.4.1-gpu

COPY requirements.txt .

RUN python3 -m pip install --upgrade pip
RUN pip install -r requirements.txt;
RUN rm -rf requirements.txt