import streamlit as st
import os

#LLM
from langchain.llms import OpenAI

#Langchain Features
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain

#PageConfig
st.set_page_config(page_title="Understood.AI")

#Import API keys
#Set API keys
os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']

#Set OpenAI as llm
llm = OpenAI(temperature = 0.9)


#Prompt Engine
#Prompt Templates
email_summary_template = PromptTemplate(
    input_variables = ['email'],
    template ='Give a summary the of following email in less than 150 words, using simple language. Start with the sender\'s name, and then explain the contents of the email. This is the email: {email}'
)

email_tone_template = PromptTemplate(
    input_variables = ['email'],
    template = 'Describe the tone of the language used in the following email: {email}.'
)

email_details_template = PromptTemplate(
    input_variables = ['email'],
    template = """
    Give a list of the people, organisations, events and places, other than the recipient and the sender, that are mentioned in the following email. Include a short summary of why they were mentioned or when the event is, based on the contents of the email. For Example: 'Arts Bar Baltic - Potential venue for a concert' or 'Start Up Conference - Takes place on the 15th of July'. Use markdown formatting for the list. Here is the email: {email}."""
)

email_requests_template = PromptTemplate(
    input_variables = ['email'],
    template = """Give a list of the requests being made to the recipient of the following email. Use markdown formating for the list. Here is the email: {email}."""
)

email_follow_up_template = PromptTemplate(
    input_variables = ['email'],
    template = """Give a list of up to 5 recomended follow up actions that the recipient of the following email should take. Use markdown formatting. Here is the email: {email}."""
)

draft_response_template = PromptTemplate(
    input_variables = ['email'],
    template = """Draft an initial response to the following email in a positive and professional tone using clear and simple language. Use markdown formatting. Here is the email: {email}. """
)


#LLM Chain
summary_chain = LLMChain(llm=llm, prompt = email_summary_template, verbose = True, output_key='summary')
tone_chain = LLMChain(llm=llm, prompt = email_tone_template, verbose = True, output_key='tone')
details_chain = LLMChain(llm=llm, prompt = email_details_template, verbose = True, output_key='details')
requests_chain = LLMChain(llm=llm, prompt = email_requests_template, verbose = True, output_key='email_requests')
follow_up_chain = LLMChain(llm=llm, prompt = email_follow_up_template, verbose = True, output_key='follow_up_actions')
draft_response_chain = LLMChain(llm=llm, prompt = draft_response_template, verbose = True, output_key='draft_response')


sequential_chain = SequentialChain(chains=[summary_chain, tone_chain, details_chain, requests_chain, follow_up_chain, draft_response_chain], input_variables=['email'], output_variables=['summary','tone','details', 'email_requests', 'follow_up_actions', 'draft_response'], verbose=True)

#App framework
st.title('✉️ Understood AI')
st.subheader('Understood AI is a Generative MindsAI tool designed to help neurodiverse people to easily understand the content, context and implicit meaning of the emails they receive') 
with st.form('my_form'):
    email = st.text_area('Copy in an email you have received and Understood AI will help you to understand what it means')
    submitted = st.form_submit_button('Submit')
    # if not openai_api_key.startswith('sk-') :
    #     st.warning('Please enter your OpenAI API key!', icon='⚠')
    if submitted:
        # and openai_api_key.startswith('sk-'):
        understood = sequential_chain(email)
        st.subheader("Here's a brief summary of the email:")
        st.write(understood['summary'])

        st.subheader('The email tone is:')
        st.write(understood['tone'])

        st.divider()

        st.subheader('The person who sent the email has asked you to:')
        st.write(understood['email_requests'])

        st.subheader('Recommended follow up actions:')
        st.write(understood['follow_up_actions'])
        
        st.subheader('Here\'s a draft of a response:')
        st.write(understood['draft_response'])

        st.divider()

        st.subheader("Key People, Places, Organisations and Events:")
        st.write(understood['details'])
        



        # email_follow_ups = follow_up_chain(email)
        # col1, col2, col3 = st.columns(3)
        # col1.write("Some of the people mentioned in this email:")
        # for person in follow_ups['people']:
        #     col1.write(person)