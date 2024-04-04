import re

import requests
import streamlit as st
import torch
import transformers
from bs4 import BeautifulSoup
import subprocess
from transformers import AutoTokenizer

from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter


# Log-in HuggingFace
def huggingface_login(token):
    try:
        subprocess.run(["huggingface-cli", "login", "--token", token], check=True)
        print("Logged in successfully to Hugging Face with the provided token.")
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")


# Extraction
def basic_cleaning(text):
    """
    A regex based function to remove html tags, non-UTF-8 characters, unicode and special characters from documents.
    """
    text = BeautifulSoup(text, "html.parser").get_text()

    # Use a regular expression to remove non-UTF-8 characters
    text = re.sub(r"[^\x00-\x7F]", "", text)

    # Remove Unicode escape sequences
    text = re.sub(r"u([0-9a-fA-F]{4})", lambda m: chr(int(m.group(1), 16)), text)

    # Remove special characters except dots
    text = re.sub(r"[^a-zA-Z0-9\s.]", "", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text)

    # Strip leading and trailing spaces
    text = text.strip()

    return text  # .encode('utf-8', errors='ignore')


def load_html(url):
    """
    Ad-hoc method using bs4 to extract relevant text from HTML pages. Tags stored in irrelevant_tags are not useful
    for intended application so those are being excluded.
    """
    concatenated_text = ""
    try:
        # Send a GET request to the URL
        response = requests.get(url)
        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # Get the HTML content from the response
            html_content = response.content

            soup = BeautifulSoup(html_content, "html.parser")
            # Find and remove cookie banners or pop-ups
            cookie_elements = soup.select(".cookie-banner, .popup")
            for element in cookie_elements:
                element.extract()

            # Remove irrelevant tags
            irrelevant_tags = [
                "img",
                "header",
                "footer",
                "script",
                "style",
                "head",
                "nav",
                "aside",
                "figure",
                "figcaption",
                "select",
                "option",
                "input",
                "button",
                "audio",
                "video",
                "source",
            ]
            irrelevant_variations = ["." + tag for tag in irrelevant_tags] + [
                "#" + tag for tag in irrelevant_tags
            ]
            for tag in irrelevant_tags + irrelevant_variations:
                elements = soup.select(tag)
                for element in elements:
                    element.extract()

            # Extract the text from each div separately
            for element in soup.find_all(
                True, recursive=False
            ):  # Find all elements, not just divs
                if element.name not in ["script", "style"]:
                    text = basic_cleaning(element.get_text(separator="\n"))
                    if text.strip():
                        concatenated_text += text.strip() + "\n"
        return concatenated_text

    except (requests.exceptions.RequestException, requests.exceptions.HTTPError) as e:
        error_message = f"Error occurred while requesting URL: {url}. Error: {str(e)}"
        print(error_message)
        return {"error": error_message}

    except Exception as e:
        error_message = f"Error: {str(e)}"
        print(error_message)
        return {"error": error_message}


def load_pdf(url):
    """
    Using PyPDFLoader method from LangChain which uses PyPDF underneath to extract text from PDF files.
    """
    try:
        results = basic_cleaning(
            "\n".join([page.page_content for page in PyPDFLoader(url).load()])
        )
        print(type(results))
    except Exception as e:
        error_message = f"Error: {str(e)}"
        print(error_message)
        return {"error": error_message}
    return results


def load_text_from_url(url):
    """
    Ad-hoc method using bs4 to extract relevant text from HTML pages. Tags stored in irrelevant_tags are not useful
    for intended application so those are being excluded.
    """
    try:
        if "pdf" in url:
            return load_pdf(url)
        else:
            return load_html(url)
    except Exception as e:
        error_message = f"Error while getting data from file: {str(e)}"
        return {"error": error_message}


# Chain
@st.cache_resource
def define_embeddings_llm():
    """
    A method to define embeddings and the LLM to use. LLM must be a chat version to work with RAG later on.
    """
    model_name = "climatebert/distilroberta-base-climate-f"
    model_kwargs = {"device": "cpu"}  # Change to 'cpu'
    encode_kwargs = {"normalize_embeddings": False}

    embeddings = HuggingFaceEmbeddings(
        model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
    )

    model = "meta-llama/Llama-2-7b-chat-hf"
    tokenizer = AutoTokenizer.from_pretrained(model)
    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        #torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="cpu",  # Set to 'cpu' for running on CPU, 'auto' is also valid
        max_length=4096,  # max_model capacity
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
    )

    llm = HuggingFacePipeline(pipeline=pipeline, model_kwargs={"temperature": 0})

    return embeddings, llm


def split_text(document, chunk_size, chunk_overlap):
    """
    Method to split documents in smaller chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len
    )
    return text_splitter.create_documents([document])
