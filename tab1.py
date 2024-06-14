import os
import streamlit as st
import psycopg2
import pandas as pd
import plotly.express as px
import logging
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain
import streamlit as st
import pandas as pd
import re
import openai
import matplotlib.pyplot as plt
from openai import OpenAI

# Load environment variables
# load_dotenv()

# Function to get database schema and table names
# # Initialize OpenAI API
# openai.api_key = "sk-proj-g9QRD3upTyLq9irN2VvdT3BlbkFJ4CaSTW6DNs7lQOXq7jaE"
os.environ["OPENAI_API_KEY"] = "sk-proj-g9QRD3upTyLq9irN2VvdT3BlbkFJ4CaSTW6DNs7lQOXq7jaE"
client = OpenAI()
def get_db_info():
    db = SQLDatabase.from_uri('postgresql+psycopg2://ajuservpostgresql:pgsql%402025@aj-flexible-server-postgre.postgres.database.azure.com:5432/devproductsdb')
    tables = db.get_usable_table_names()
    return db.dialect, tables, db

def generate_plotly_code(df):
    column_names = df.columns.tolist()
    data_types = df.dtypes.tolist()
    columns_info = "\n".join([f"{name}: {dtype}" for name, dtype in zip(column_names, data_types)])
    
    prompt = f"""
    Given the following DataFrame columns and their data types, generate a Plotly code to visualize the data:

    Columns and data types:
    {columns_info}

    Use a bar chart for visualization.
    """

    try:
        llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            api_key=os.getenv("OPENAI_API_KEY"),
            max_tokens=2048
        )

        response = llm(messages=[{"role": "system", "content": prompt}])
        plotly_code = response.choices[0].message["content"].strip()

        return plotly_code

    except Exception as e:
        st.error(f"Error generating Plotly code: {e}")
        return None


def ask(question, visualize=True, allow_llm_to_see_data=False):
    """
    Function to generate SQL query, run it and visualize results.

    Args:
        question (str): The question to ask.
        visualize (bool): Whether to generate Plotly visualization.
        allow_llm_to_see_data (bool): Whether to allow the LLM to see data.

    Returns:
        Tuple[str, pd.DataFrame, plotly.graph_objs.Figure]: The SQL query, the results of the SQL query, and the plotly figure.
    """
    dialect, tables,db = get_db_info()
    schema = f"Dialect: {dialect}\nTables: {tables}"

    prompt = f"""
    Given the following schema: {schema}, generate an SQL query to answer: {question}
    """

    try:
        # Initialize ChatOpenAI with appropriate settings
        llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            api_key=os.getenv("OPENAI_API_KEY"),
            max_tokens=2048
        )

        # Create chain linking LLM to SQL Database
        chain = create_sql_query_chain(llm, db)

        # Invoke the chain with the user question
        response = chain.invoke({"question": question})

        # Extract the SQL query from the response
        sql = response
        # sql = response.choices[0].text.strip().replace("```", "").replace("sql", "")

    except Exception as e:
        st.error(f"Error generating SQL query: {e}")
        return None, None, None

    st.text(f"Generated SQL Query:\n{sql}")

    try:
        # Run SQL query
        conn = psycopg2.connect(
            dbname="devproductsdb",
            user="ajuservpostgresql",
            password="pgsql@2025",
            host="aj-flexible-server-postgre.postgres.database.azure.com",
            port="5432"
            )
        df = pd.read_sql_query(sql, conn)
        conn.close()

        if df is not None:
            with st.expander("Show data"):
                st.write(df)

            column_names = ", ".join(df.columns)
            st.text(f"Column names: {column_names}")

            prompt_content = f"""
            The dataset is ALREADY loaded into a DataFrame named 'df'. DO NOT load the data again.
            
            The DataFrame has the following columns: {column_names}
            
            Use package Pandas and Matplotlib ONLY.
            Provide SINGLE CODE BLOCK with a solution using Pandas and Matplotlib plots in a single figure to address the following query:
            
            {question}

            - USE SINGLE CODE BLOCK with a solution. 
            - Do NOT EXPLAIN the code 
            - DO NOT COMMENT the code. 
            - ALWAYS WRAP UP THE CODE IN A SINGLE CODE BLOCK.
            """

            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful Data Visualization assistant who gives a single block without explaining or commenting the code to plot. IF ANYTHING NOT ABOUT THE DATA, JUST politely respond that you don't know.",
                },
                {"role": "user", "content": prompt_content},
            ]

            with st.spinner("📟 *Prompting is the new programming*..."):
                response = client.chat.completions.create(model="gpt-3.5-turbo",
        messages=messages
    )
                # result = response['choices'][0]['message']['content'].strip()
                result = response.choices[0].message.content
                print(result )
                # st.code(result)
                

                code_blocks = re.findall(r"```(python)?(.*?)```", result, re.DOTALL)
                code = "\n".join([block[1].strip() for block in code_blocks])

                if code:
                    try:
                        fig, ax = plt.subplots()
                        exec(code, {'df': df, 'plt': plt, 'fig': fig, 'ax': ax})
                        st.pyplot(fig)
                        # st.plotly_chart(fig)
                    except Exception as e:
                        st.error(f"📟 Apologies, failed to execute the code due to the error: {str(e)}")
                        st.warning(
                            """
                            📟 Check the error message and the code executed above to investigate further.
                            """
                        )
                        # fig = px.bar(df, x=x_col, y=y_col, title=title)
                        # st.plotly_chart(fig)
                        # st.code(result)


            # if not df.empty:
            #     if visualize:
            #         plotly_code = generate_plotly_code(df)
            #         if plotly_code:
            #             exec(plotly_code)
            # else:
            #     st.warning("The given data is empty.")
        else:
            st.warning("No data was returned by the query.")

    except Exception as e:
        st.error(f"Error running SQL query: {e}")
        return None, None, None
    

def extract_code_from_markdown(md_text):
    code_blocks = re.findall(r"```(python)?(.*?)```", md_text, re.DOTALL)
    code = "\n".join([block[1].strip() for block in code_blocks])
    return code




# Streamlit application interface
st.title("SQL Query Generator with OpenAI and PostgreSQL")

if "user_query" not in st.session_state:
    st.session_state.user_query = ""

user_query = st.text_input("Enter your question:", key="user_query")

if st.button("Submit"):
    ask(user_query)
