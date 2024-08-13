#import libraries

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from crewai import Crew, Agent, Task, Process
from langchain_openai import ChatOpenAI
import json
import os
import requests
from langchain.tools import tool
from langchain.agents import load_tools
from langchain_openai import ChatOpenAI
from crewai_tools import tool
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tempfile
from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain.agents import load_tools
import streamlit as st

# Set the title of the app

st.title("Deeptech Innovation Builder")

#insert picture in strealit app

st.image('Innovation builder.png', caption='Deeptech Innovation Builder', use_column_width=True)

# Set the description of the app

st.write("This app is designed to help you generate innovative product ideas based on a set of patents. The app uses a set of AI agents to help you identify the most common topics in the patents, explore novel product ideas, and optimize the product ideas based on technical feasibility and market analysis. This app is developed by Dries Faems. For more information, please visit: https://www.linkedin.com/in/dries-faems-0371569/.")

# ask for the API key in password form

api_key = st.text_input("Please provide your OpenAI API Key. If you do not have an OpenAI API Key, you can create it here: https://platform.openai.com/api-keys. This information will not be stored", type="password")

# set the API key as an environment variable

os.environ["OPENAI_API_KEY"] = api_key

#we ask the user to upload PDF files that can be used as knowledge base for the agents. 
uploaded_file = st.file_uploader("Please upload a PDF file that contains the patents and their descripions.", type="pdf")

# Save the uploaded PDF to a temporary file
if st.button('Confirm') and api_key:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name

    # Load PDF into docs
    loaders = [
        PyPDFLoader(temp_file_path),
    ]
    docs = []
    for loader in loaders:
        docs.extend(loader.load())

    # Split docs into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=150
    )
    splits = text_splitter.split_documents(docs)

    # Create embeddings for all the chunks
    from langchain_openai import OpenAIEmbeddings
    embedding = OpenAIEmbeddings(api_key=api_key)

    # Store embeddings in vector database
    vectorstore = FAISS.from_documents(splits, embedding)

    # Setting up the retrieval function
    from langchain.chains import ConversationalRetrievalChain
    chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(temperature=0.0, model_name='gpt-4o', api_key=api_key),
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
        verbose=True
    )
    #creation of tool for searching in patent database
    @tool("knowledge_search")
    def knowledge_search(question: str) -> str:
        """This tool can search in the provided knowledge by exclusively screening the context provided."""
        return chain.invoke({"question": question, "chat_history": []})

    # Define agents for first crew that needs to engage in product ideation

    topic_identification_specialist = Agent(
        role='Topic Identification Specialist',
        goal='Summarize the three most common topics in the provided patent database',
        backstory="""You are a great expert in summarizing topics from the provided patent database.""",
        verbose=True,
        allow_delegation=False,
        max_iter=5,
        memory=True,
        tools=[knowledge_search]
    )

    product_development_specialist = Agent(
        role='Product Development Specialist',
        goal='Use the identified topics from the topic_identification_specialist and the information of the patent_database_search tool to explore a novel product that can leverage the topics and patents.',
        backstory="""You are a very experienced product developer that can come up with creative suggestions for new products based on recombining available patents.""",
        verbose=True,
        allow_delegation=False,
        max_iter=5,
        memory=True,
        tools=[knowledge_search]
    )

    # Create tasks for the agents
    get_topics = Task(
        description="""Summarize the three most common topics in the provided patent database.""",
        expected_output="""As output, you provide a list of three topics that are present in multiple patents in the database and that have potential for developing a novel product.""",
        agent=topic_identification_specialist
    )

    get_product_idea = Task(
        description=f"""Use the patent_database_search tool and the topics identified by the topic_identification_specialist to ideate on novel product. You should ask the patent_database_search tool questions such as: Question 1: What are potential painpoints that can be solved with the patents in the database?; Question 2: What could be an interesting proudct that can be developed by recombining the patents in the database? The product should be very specific and leverage specific aspects of the patents. You are allowed to engage in multiple iterations to fine-tune and optimize the product idea.""",
        expected_output='As output, provide the following two elements: (i) a clear and compelling description of the product and (ii) Provide a list of the patents that are used in the product idea.',
        agent=product_development_specialist
    )

    # Instantiate the first crew with a sequential process
    crew = Crew(
        agents=[topic_identification_specialist, product_development_specialist],
        tasks=[get_topics, get_product_idea],
        verbose=2,
        process=Process.sequential,
        full_output=True,
        share_crew=False,
    )

    # Kick off the crew's work
    results = crew.kickoff()

    initial_idea = get_product_idea.output.raw_output
    st.markdown('Product Idea Generated by the Ideation Crew')
    st.write(initial_idea)

    #intiate agents for second crew that needs to optimize the product idea

    technical_expert = Agent(
        role='Technical Expert',
        goal='You can optimize a product idea by analyzing its technological feasibility and innovativeness.',
        backstory="""You are a technical expert that can optimize a product idea based on the technical feasibility of the product idea.""",
        verbose=True,
        allow_delegation=False,
        max_iter=5,
        memory=True,
    )

    market_expert = Agent(
        role='Competition Expert',
        goal='You can optimize a product by identifying specific customers and target markets for the product idea.',
        backstory="""You are an expert in identifying optmal customer profiles for the product idea.""",
        verbose=True,
        allow_delegation=False,
        max_iter=5,
        memory=True,
        tools=[knowledge_search]
    )

    finalization_expert = Agent(
        role='Finalization Expert',
        goal='You can finalize the product idea by providing a final product description based on the optimizations of the technical_expert and competition_expert.',
        backstory="""You are an expert in finalizing product ideas based on the technical feasibility and competition analysis.""",
        verbose=True,
        allow_delegation=False,
        max_iter=3,
        memory=True,
    )
    # Create tasks for the agents

    technical_optimization = Task(
        description=f"""Engage in a technical optimization of the following product idea: {initial_idea}.""",
        expected_output='As output, provide an optimized description of the product idea.',
        agent=technical_expert,
    )

    competition_optimization = Task(
        description=f"""Engage in a market optimization where you identify the optimal customer segment for the following idea and optimize it accordingly: {initial_idea}.""",
        expected_output='As output, provide an optimized description of the product idea.',
        agent=market_expert,
    )

    finalization = Task(
        description=f"""Finalize the product idea by providing a final product description based on the optimizations of the technical_expert and competition_expert.""",
        expected_output='As output, provide a final description of the product idea.',
        agent=finalization_expert,
    )
    # Instantiate the second crew with a sequental process

    second_crew = Crew(
        agents=[technical_expert, market_expert, finalization_expert],
        tasks=[technical_optimization, competition_optimization, finalization],
        verbose=2,
        process=Process.sequential,
        full_output=True,
        share_crew=False,
    )

    # Kick off the crew's work
    second_results = second_crew.kickoff()


    st.markdown("**Product Optimization by the Optimization Crew**")
    st.write(f"""{finalization.output.raw_output}""")
else:
    st.write("Please provide the API key your patents and click Confirm to proceed.")