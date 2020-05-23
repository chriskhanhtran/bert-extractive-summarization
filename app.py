import streamlit as st
import os
import torch
import urllib.request
from models.model_builder import ExtSummarizer
from newspaper import Article
from ext_sum import summarize


@st.cache(suppress_st_warning=True)
def download_model():
    url = 'https://www.googleapis.com/drive/v3/files/1tXdugYx8NU73_G4FK7XX08aGip_D1SiJ?alt=media&key=AIzaSyCmo6sAQ37OK8DK4wnT94PoLx5lx-7VTDE'
    urllib.request.urlretrieve(url, 'checkpoint/cnndm_ext.pt')

@st.cache(suppress_st_warning=True)
def load_model(path):
    checkpoint = torch.load(path)
    model = ExtSummarizer(device="cpu", checkpoint=checkpoint, max_pos=512)
    return model

def crawl_url(url):
    article = Article(url)
    article.download()
    article.parse()
    return article.text

def main():
    st.markdown("<h1 style='text-align: center;'>Extractive Summary✏️</h1>", unsafe_allow_html=True)

    # Download model
    if not os.path.exists('checkpoint'):
        os.makedirs('checkpoint', exist_ok=True)
        download_model()

    # Load model
    model = load_model('checkpoint/cnndm_ext.pt')

    # Input
    input_type = st.radio("Input Type: ", ["URL", "Raw Text"])
    st.markdown("<h3 style='text-align: center;'>Input</h3>", unsafe_allow_html=True)
    if input_type == "Raw Text":
        with open("raw_data/cnn.txt") as f:
            sample_text = f.read()
        text = st.text_area("", sample_text, 200)
    else:
        url = st.text_input("", "https://www.cnn.com/2020/05/23/us/cicadas-emerge-17-years-underground-scn-trnd/index.html")
        text = crawl_url(url)

    input_fp = "raw_data/input.txt"
    with open(input_fp, 'w') as file:
        file.write(text)

    # Summarize
    sum_level = st.radio("Output Length: ", ["Short", "Medium"])
    max_length = 3 if sum_level == "Short" else 6
    result_fp = 'results/summary.txt'
    summary = summarize(input_fp, result_fp, model, max_length=max_length)
    st.markdown("<h3 style='text-align: center;'>Summary</h3>", unsafe_allow_html=True)
    st.markdown(f"<p align='justify'>{summary}</p>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()

    





