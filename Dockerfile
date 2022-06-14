FROM python:3.8

WORKDIR /app


COPY ./ui /app/ui
COPY ./exdpn /app/exdpn
COPY ./setup.py /app/setup.py
COPY ./requirements.txt /app/requirements.txt

RUN pip install -r requirements.txt

RUN python setup.py bdist_wheel

RUN pip install dist/*.whl

WORKDIR /app/ui

ENTRYPOINT ["python"]

CMD ["app.py"]