FROM pytorch/pytorch:1.13.0-cuda11.6-cudnn8-devel

RUN pip install langchain faiss-gpu transformers pandas sentence-transformers streamlit accelerate

EXPOSE 8501

COPY . .

# Set the entry point to the script
ENTRYPOINT ["streamlit", "run"]
CMD ["app.py"]