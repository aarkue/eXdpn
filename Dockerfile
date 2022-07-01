FROM python:3.8

WORKDIR /app


COPY ./ui /app/ui
COPY ./exdpn /app/exdpn
COPY ./setup.cfg /app/setup.cfg
COPY ./pyproject.toml /app/pyproject.toml
COPY ./requirements.txt /app/requirements.txt


# RUN python3 -m venv venv
# ENV VIRTUAL_ENV=/app/venv
# ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN pip install -r requirements.txt

RUN python -m build

RUN pip install dist/*.whl

WORKDIR /app/ui

ENTRYPOINT ["python"]

CMD ["app.py"]