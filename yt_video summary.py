import validators,streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader,UnstructuredURLLoader
import os

## sstreamlit APP
st.set_page_config(page_title="Youtube Video summary")
st.title("Youtube Video summary")
st.write("AI tries to summarise video based on transcripts obtainedf rom youtube.")
st.write("More context more training time and more reasonable summary")
st.subheader('Youtube URL')



## Get the Groq API Key and url(YT or website)to be summarized
groq_api_key=st.secrets["GROQ_API_KEY"]
llm=ChatGroq(groq_api_key=groq_api_key,model="Gemma-7b-It")
generic_url=st.text_input("URL",label_visibility="collapsed")

## Gemma Model USsing Groq API
llm =ChatGroq(model="Gemma-7b-It", groq_api_key=groq_api_key)

prompt_template="""
Provide a summary of the following content in 300 words:
Content:{text}

"""
prompt=PromptTemplate(template=prompt_template,input_variables=["text"])

if st.button("Summarize the Content for youtube video"):
    ## Validate all the inputs
    if not groq_api_key.strip() or not generic_url.strip():
        st.error("Please provide the information to get started")
    elif not validators.url(generic_url):
        st.error("Please enter a valid youtube Url.")

    else:
        try:
            with st.spinner("Waiting..."):
                ## loading the website or yt video data
                
                loader=YoutubeLoader.from_youtube_url(generic_url,add_video_info=True)
                docs=loader.load()

                ## Chain For Summarization
                chain=load_summarize_chain(llm,chain_type="stuff",prompt=prompt)
                output_summary=chain.run(docs)

                st.success(output_summary)
        except Exception as e:
            st.exception(f"Exception:{e}")
                    