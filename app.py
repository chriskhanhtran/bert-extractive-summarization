import streamlit as st
import os
import torch
import nltk
import urllib.request
from models.model_builder import ExtSummarizer
from newspaper import Article
from ext_sum import summarize


def main():
    st.markdown("<h1 style='text-align: center;'>Extractive Summary✏️</h1>", unsafe_allow_html=True)

    # Download model
    if not os.path.exists('checkpoints/mobilebert_ext.pt'):
        download_model()

    # Load model
    model = load_model('mobilebert')

    # Input
    input_type = st.radio("Input Type: ", ["URL", "Raw Text"])
    st.markdown("<h3 style='text-align: center;'>Input</h3>", unsafe_allow_html=True)

    if input_type == "Raw Text":
        with open("raw_data/input.txt") as f:
            sample_text = f.read()
        text = st.text_area("", sample_text, 200)
    else:
        url = st.text_input("", "https://www.cnn.com/2020/05/29/tech/facebook-violence-trump/index.html")
        st.markdown(f"[*Read Original News*]({url})")
        text = crawl_url(url)

    input_fp = "raw_data/input.txt"
    with open(input_fp, 'w') as file:
        file.write(text)

    # Summarize
    sum_level = st.radio("Output Length: ", ["Short", "Medium"])
    max_length = 3 if sum_level == "Short" else 5
    result_fp = 'results/summary.txt'
    summary = summarize(input_fp, result_fp, model, max_length=max_length)
    st.markdown("<h3 style='text-align: center;'>Summary</h3>", unsafe_allow_html=True)
    st.markdown(f"<p align='justify'>{summary}</p>", unsafe_allow_html=True)


def download_model():
    nltk.download('popular')
    url = 'https://www.googleapis.com/drive/v3/files/1umMOXoueo38zID_AKFSIOGxG9XjS5hDC?alt=media&key=AIzaSyCmo6sAQ37OK8DK4wnT94PoLx5lx-7VTDE'

    # These are handles to two visual elements to animate.
    weights_warning, progress_bar = None, None
    try:
        weights_warning = st.warning("Downloading checkpoint...")
        progress_bar = st.progress(0)
        with open('checkpoints/mobilebert_ext.pt', 'wb') as output_file:
            with urllib.request.urlopen(url) as response:
                length = int(response.info()["Content-Length"])
                counter = 0.0
                MEGABYTES = 2.0 ** 20.0
                while True:
                    data = response.read(8192)
                    if not data:
                        break
                    counter += len(data)
                    output_file.write(data)

                    # We perform animation by overwriting the elements.
                    weights_warning.warning("Downloading checkpoint... (%6.2f/%6.2f MB)" %
                        (counter / MEGABYTES, length / MEGABYTES))
                    progress_bar.progress(min(counter / length, 1.0))

    # Finally, we remove these visual elements by calling .empty().
    finally:
        if weights_warning is not None:
            weights_warning.empty()
        if progress_bar is not None:
            progress_bar.empty()


@st.cache(suppress_st_warning=True)
def load_model(model_type):
    checkpoint = torch.load(f'checkpoints/{model_type}_ext.pt', map_location='cpu')
    model = ExtSummarizer(device="cpu", checkpoint=checkpoint, bert_type=model_type)
    return model


def crawl_url(url):
    article = Article(url)
    article.download()
    article.parse()
    return article.text


if __name__ == "__main__":
    main()
