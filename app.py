
#core package
import streamlit as st
st.set_page_config(page_title="NLP Web App", page_icon=None, layout="centered", initial_sidebar_state="auto")


#NLP Packages
from textblob import TextBlob
import spacy
import neattext as nt

#Visualization Package
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")
from wordcloud import WordCloud


from collections import Counter
import re
import requests

#Translator Library
from deep_translator import GoogleTranslator


def generate_summary(text):
    # Set the API endpoint and parameters
    endpoint = 'https://api.openai.com/v1/completions'
    params = {
        'model': 'llama',
        'prompt': text,
        'max_tokens': 100}
    
    # Send a POST request to the API
    response = requests.post(endpoint, json=params)
    
    # Extract the summary from the response
    summary = response.json()['choices'][0]['text']
    return summary

def summarize_text(text, num_sentences=5):
	#Remove special character and convert text to lowercase
	clean_text = re.sub('[^a-zA-Z]', ' ', text).lower()

	#Split the text into words
	words = clean_text.split()

	#Calculate the frequency of each word
	word_freq = Counter(words)

	#Sort the words based on their frequency in decending order
	sorted_words = sorted(word_freq, key=word_freq.get, reverse=True)

	#Extract the top 'Num Sentences' most frequent words
	top_words = sorted_words[:num_sentences]

	#Create the summary by joining the top words
	summary = ' '.join(top_words)

	return summary


@st.cache_data
#Lemma and Tokens Function
def text_analyzer(text):
    #import english library
    nlp = spacy.load('en_core_web_sm')
    #create an nlp object
    doc = nlp(text)
    #Extract Token & Lemma
    alldata = [('"Token":{},\n"Lemma":{}'.format(token.text, token.lemma_)) for token in doc]
    return alldata





def main():
    """NLP web app with Streamlit"""

    title_template = """
    <div style="background-color:crimson; padding:8px;">
    <h1 style="color:white">NLP Web Application</h1>
    </div>
    """

    st.markdown(title_template, unsafe_allow_html=True)

    subheader_template = """
    <div style="background-color:papayawhip; padding:8px;">
    <h3 style="color:blue"><i><b>Created By -AWD- Powered by Streamlit</b></i> </h1>
    """

    st.markdown(subheader_template, unsafe_allow_html=True)

    st.sidebar.image("nlp.jpg", width=300)
    
    activity = ["Text Analysis", "Translation", "Sentiment Analysis", "About"]
    choice = st.sidebar.selectbox("Menu", activity)

    if choice == "Text Analysis":   
      st.subheader("Text Analysis")
      st.write("")

      raw_text = st.text_area("Write something", "Enter a text in english...", height=300)
      
      if st.button("Analyze"):
          if len(raw_text) == 0:
              st.warning("Pleae enter a text...")
          else:
              #blob = TextBlob(raw_text)
              st.info("Basic Function")
              
              col1, col2 = st.columns(2)
              
              with col1:
                  with st.expander("Basic Info"):
                      st.write("Text Stats")
                      word_desc = nt.TextFrame(raw_text).word_stats()
                      result_desc = {"Length of Text":word_desc['Length of Text'],
                                     "Num of Vowels":word_desc['Num of Vowels'],
                                     "Num of Consonants":word_desc['Num of Consonants'],
                                     "Num of Stopwords":word_desc['Num of Stopwords']}
                      st.write(result_desc)

                  with st.expander("Stopwords"):
                      st.success("Stop Words List")
                      stop_w = nt.TextExtractor(raw_text).extract_stopwords()
                      st.error(stop_w)
                
              with col2:
                  with st.expander("Procesed Text"):
                      st.success("Stopwords Excluded Text")
                      processed_text = str(nt.TextFrame(raw_text).remove_stopwords())
                      st.write(processed_text)

                  with st.expander("Plot Wordcloud"):
                      wordcloud = WordCloud().generate(processed_text)
                      fig = plt.figure(1, figsize=(20,20))
                      plt.imshow(wordcloud, interpolation = 'bilinear')
                      plt.axis('off')
                      st.pyplot(fig)

              st.write("")
              st.write("")
              st.info("Advance Features")

              col3, col4 = st.columns(2)

              with col3:
                  with st.expander("Tokens & Lemmas"):
                      st.write("T&K")
                      processed_text_mid = str(nt.TextFrame(raw_text).remove_stopwords())
                      processed_text_mid = str(nt.TextFrame(processed_text_mid).remove_puncts())
                      processed_text_fin = str(nt.TextFrame(processed_text_mid).remove_special_characters())
                      tandl = text_analyzer(processed_text_fin)
                      st.json(tandl)
			  
              with col4:
                  with st.expander("Summarize"):
                      st.success("Summarize")
                      summary_text = str(nt.TextFrame(raw_text).remove_stopwords())
                      summary_text = str(nt.TextFrame(summary_text).remove_puncts())
                      summary_text_fin = str(nt.TextFrame(summary_text).remove_special_characters())
                      summary = summarize_text(summary_text_fin)
                      st.success(summary)
                  
    if choice == "Translation":
      st.subheader("Translation")
      st.write("")
      st.write("")
      raw_text = st.text_area("Original Text", "Write Something to be translated...", height=300)
      if len(raw_text) < 3:
          st.warning("Please provide a text with at least 3 Characters. . .")
      else:
          target_lang = st.selectbox("Target Language", ["German","Spanish","French","Italian","Bahasa"])
          if target_lang == "German":
              target_lang = "de"
          elif target_lang == "Spanish":
              target_lang = "es"
          elif target_lang == "French":
              target_lang = "fr"
          elif target_lang == "Italian":
              target_lang = "it"
          else:
              target_lang = "id"

          if st.button("Translate"):
              translator = GoogleTranslator(source='auto', target=target_lang) #Set Source and target language
              translated_text = translator.translate(raw_text)
              st.write(translated_text)

              

    if choice == "Sentiment Analysis":
      st.subheader("Sentiment Analysis")
      st.write("")
      raw_text = st.text_area("Text to analyse", "Enter a text here. . .", height=300)
      if st.button("Evaluate"):
          if len(raw_text) == 0:
              st.warning("Enter a text. . .")
          else:
              blob = TextBlob(raw_text)
              st.info("Sentiment Analysis")
              st.write(blob.sentiment)
              st.write("")
          

    if choice == "About":
      st.subheader("About")
      st.write("")
       
      st.markdown("""
      ### NLP Web App Made With Streamlit
      
      For Info:
      - [streamlit](https://streamlit.io)
      """)

    
    
if __name__ == "__main__":
      main()
    



    
    

