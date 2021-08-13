FROM python:3.8-slim-buster

# RUN /usr/local/bin/python -m pip install --upgrade pip
RUN pip install torch==1.8.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install torchvision==0.9.1+cpu -f https://download.pytorch.org/whl/torch_stable.html

COPY requirements.txt .
RUN pip install -r requirements.txt

# cellpose requires large models, load these from cache with:
RUN python -c 'from cellpose import models'

EXPOSE 8501
ENTRYPOINT ["streamlit","run"]
CMD ["root/app.py"]