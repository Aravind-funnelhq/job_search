import streamlit as st
import os
from langchain.chains import ConversationalRetrievalChain
import pathlib
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain import Cohere, ConversationChain, LLMChain, PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import textwrap
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage
from IPython.display import display
from IPython.display import Markdown
import google.generativeai as genai
from langchain.prompts import PromptTemplate
from IPython.display import Markdown
import base64
from csv import writer
from langchain_community.embeddings import CohereEmbeddings
from langchain_text_splitters import CharacterTextSplitter


def to_markdown(text):
  text = text.replace('•', '  *')
  return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))

def main():
    st.set_page_config(layout="wide", page_title="Job Search")
    st.sidebar.title("Features")
    page = st.sidebar.radio("Go to", ["Job Finder","Recruiter"])
    super = ChatGoogleGenerativeAI(model='gemini-pro-vision',google_api_key='AIzaSyBP0pIkZbUiJ-UWCxYKk75CWN78usAA8Sk')
    embeddings = CohereEmbeddings(model="embed-english-light-v3.0",user_agent="my-langchain-app",cohere_api_key='xPQlz92wRppOVGZCHNETnpZhtN9o2ZfO2QEVWh22')
    new_db = FAISS.load_local("job_descriptions", embeddings,allow_dangerous_deserialization=True)

    if page == "Job Finder":
        st.title("Job Finder")
        uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            content = uploaded_file.read()
            image = base64.b64encode(content).decode("utf-8")
            input2 = [
       HumanMessage(
                content=[
                    {f"type": "text", "text": """You are Here to Extract all the required fileds given below from the uploaded resume.with all the below listed featured
                     write the overview in a single paragraph of the resume in not more than 1000 characters with all the keywords required.

                     fields to capture:
                     1.Skill set
                     2.specialization or study
                     3.Technologies worked with
                     4.work experience
                     """},
                    {"type": "image_url", "image_url": f"data:image/png;base64,{image}"},
                ]
            )
        ]
            res = super.invoke(input2)
            st.header("Your Resume Summary")
            st.write(res.content)
            context = new_db.similarity_search(res.content)
            st.header("Job Similarities")
            company = context[0].page_content
            st.write(company)
            if st.button('Apply'):
                data = {"job_description":[res.content],"applicants":[company]}
                df = pd.DataFrame(data)
                df.to_csv('job_application.csv',index=False)
                st.write("Applied")

    if page == "Recruiter":
        st.title("Recruiters Page")
        description = st.text_area("Please Enter the Job Description")
        if st.button('Post Job'):
            db1 = FAISS.from_texts([description], embeddings)
            new_db.merge_from(db1)
            new_db.save_local("job_descriptions")
            st.write('Job posted')


if __name__ == "__main__":
    main()

