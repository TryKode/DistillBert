import streamlit as st

from bs4 import BeautifulSoup as bs
import requests
import re

from transformers import DistilBertForQuestionAnswering
from transformers import DistilBertTokenizer
from textblob import TextBlob
import torch
import textwrap

from PIL import Image
image = Image.open('logo.jpg')
st.image(image, use_column_width =True)

@st.cache()
def load_model( ):
    return DistilBertForQuestionAnswering.from_pretrained('distilbert-base-uncased-distilled-squad')

@st.cache()
def load_tokenizer( ):
    return DistilBertTokenizer.from_pretrained('distilbert-base-uncased',return_token_type_ids = True)
    
def scrape_data(product_url):
    # scrape data
    productURL = product_url
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/78.0.3904.108 Safari/537.36", "Accept-Encoding":"gzip, deflate", "Accept":"text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8", "DNT":"1","Connection":"close", "Upgrade-Insecure-Requests":"1"}
    productPage = requests.get(productURL, headers=headers)
    productSoup = bs(productPage.content,'html.parser')

    productNames = productSoup.find_all('span', id='productTitle')
    productNames = productNames[0].get_text().strip()
    
    ids = ['priceblock_dealprice', 'priceblock_ourprice', 'tp_price_block_total_price_ww']
    for ID in ids:
        productDiscountPrice = productSoup.find_all('span', id=ID)
        if len(productDiscountPrice) > 0 :
            break
    productDiscountPrice = productDiscountPrice[0].get_text().strip()
    productDiscountPrice = 'Product Price after Discount '+productDiscountPrice

    classes = ['priceBlockStrikePriceString', 'a-text-price']
    for CLASS in classes:
        productActualPrice = productSoup.find_all('span', class_=CLASS)
        if productActualPrice != [] :
            break
    productActualPrice = productActualPrice[0].get_text().strip()
    productActualPrice = 'Product Actual Price '+productActualPrice

    productFeatures = productSoup.find_all('div', id='feature-bullets')
    productFeatures = productFeatures[0].get_text().strip()
    productFeatures = re.split('\n|  ',productFeatures)
    temp = []
    for i in range(len(productFeatures)):
        if productFeatures[i]!='' and productFeatures[i]!=' ' :
            temp.append( productFeatures[i].strip() )
    productFeatures = temp
    
    productSpecs = productSoup.find_all('table', id='productDetails_techSpec_section_1')
    productSpecs = productSpecs[0].get_text().strip()
    productSpecs = re.split('\n|\u200e|  ',productSpecs) 
    temp = []
    for i in range(len(productSpecs)):
        if productSpecs[i]!='' and productSpecs[i]!=' ' :
            temp.append( productSpecs[i].strip() )
    productSpecs = temp

    productDetails = productSoup.find_all('div', id='productDetails_db_sections')
    productDetails = productDetails[0].get_text()
    productDetails = re.split('\n|  ',productDetails) 
    temp = []
    for i in range(len(productDetails)):
        if productDetails[i]!='' and productDetails[i]!=' ' :
            temp.append( productDetails[i].strip() )
    productDetails = temp
    
    context = productNames + '\n' + productDiscountPrice + '. ' + productActualPrice + '.\n'
    i = 0
    while i<len(productFeatures):
        context = context + productFeatures[i]+', '
        i = i+1

    i = 0
    while i<len(productSpecs):
        context = context + productSpecs[i]+' '+productSpecs[i+1]+', '
        i = i+2
    context = context[:len(context)-2] + '.\n'

    # print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> | ', productNames, productDiscountPrice, productActualPrice, productFeatures, productSpecs, productDetails, context, sep="_-_-_-_-_")
    details = {
        'product_data' : {
            'productNames' : productNames,
            'productDiscountPrice' : productDiscountPrice,
            'productActualPrice' : productActualPrice,
            'productFeatures' : productFeatures,
            'productSpecs' : productSpecs,
            'productDetails' : productDetails,
            'context' : context
        }
    }

    return details

def qna_bert(context, question):
    model = load_model()
    tokenizer = load_tokenizer()
        
    def check_spelling(question):
        question = re.sub(r'[^\w\s]', '', question)
        question = question.lower()
        question_list = question.split()

        for i in range(len(question_list)):
            question_list[i] = str( TextBlob(question_list[i]).correct() )
        
        question = " ".join(question_list)
        return (question + " ?")

    def answer_question(question, answer_text):
        encoding = tokenizer.encode_plus(question, answer_text)
        input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]

        outputs = model(torch.tensor([input_ids]), attention_mask=torch.tensor([attention_mask]))
        start_scores = outputs.start_logits
        end_scores = outputs.end_logits

        ans_tokens = input_ids[torch.argmax(start_scores) : torch.argmax(end_scores)+1]
        answer_tokens = tokenizer.convert_ids_to_tokens(ans_tokens , skip_special_tokens=True)

        print ("\nQuestion ",question)
        print ("\nAnswer Tokens: ")
        print (answer_tokens)

        answer_tokens_to_string = tokenizer.convert_tokens_to_string(answer_tokens)

        print ("\nAnswer : ",answer_tokens_to_string)
        return answer_tokens_to_string

    context = context
    question = check_spelling(question)
    answer = answer_question(question, context)

    return {'context': context, 'question' : question, 'answer' : answer}

data = None
st.title('Product Question-Answering System')

product_url_checkbox = st.checkbox('Product URL')
if product_url_checkbox:
    product_url = st.text_input('', placeholder='URL of Product')
    if product_url != '':
        data = scrape_data(product_url)

if product_url_checkbox and data:
    question = st.text_input('', placeholder='Ask any Question')
    answer = qna_bert(data['product_data']['context'], question)

    if '[CLS]' in answer['answer'] or '[SEP]' in answer['answer'] and question!='' :
        st.warning('Please Try Changing the Keyword !!!')
        st.warning(answer['answer'])
    elif question!='':
        st.success(answer['answer'])