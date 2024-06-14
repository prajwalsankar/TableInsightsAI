from vanna.openai import OpenAI_Chat
from vanna.vannadb import VannaDB_VectorStore
from langchain_community.utilities import SQLDatabase
 
class MyVanna(VannaDB_VectorStore, OpenAI_Chat):
    def __init__(self, config=None):
        MY_VANNA_MODEL = "tableinsightai"
        VannaDB_VectorStore.__init__(self, vanna_model=MY_VANNA_MODEL, vanna_api_key="104208042c64490bb7ab9f149da970b8", config=config)
        OpenAI_Chat.__init__(self, config=config)
 
 
vn = MyVanna(config={'api_key': 'sk-proj-g9QRD3upTyLq9irN2VvdT3BlbkFJ4CaSTW6DNs7lQOXq7jaE', 'model': 'gpt-4'})
connectionString = 'postgresql+psycopg2://ajuservpostgresql:pgsql%402025@aj-flexible-server-postgre.postgres.database.azure.com:5432/devproductsdb'
# db = SQLDatabase.from_uri(connectionString)
# print(db.dialect)
# print(db.get_usable_table_names())
# db.run("SELECT * FROM Artist LIMIT 10;")
 
vn.connect_to_postgres(
    host='aj-flexible-server-postgre.postgres.database.azure.com',
    dbname='devproductsdb',
    user='ajuservpostgresql',
    password='pgsql@2025',
    port='5432'
)

# Run an initial query to get the information schema
# df_information_schema = db.run("SELECT * FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = 'products'")
df_information_schema = vn.run_sql("SELECT * FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = 'Products'")
 
 
plan = vn.get_training_plan_generic(df_information_schema)
print(plan)  # Inspect the plan
 
# Uncomment the next line to train with the plan if you are satisfied
# vn.train(plan=plan)
 
# Add documentation about your business terminology or definitions
# vn.train(documentation="Our business defines OTIF score as the percentage of orders that are delivered on time and in full")
 
 
# vn.train(sql="SELECT * FROM Products WHERE productname = 'Laptop'")
 
 
# training_data = vn.get_training_data()
# # print(training_data)
 
# ```## Asking the AI
# Whenever you ask a new question, it will find the 10 most relevant pieces of training data and use it as part of the LLM prompt to generate the SQL.
# ```python
# vn.ask(question=...)
response = vn.ask(question="How many unitsinstock of 'Laptop' are currently in stock?")
print(response)
# try:
#     df = vn.run_sql("SELECT AVG(unitprice) FROM Products")
#     print(df)
# except Exception as e:
#     print(f"Couldn't run SQL query: {e}")

from vanna.flask import VannaFlaskApp
app = VannaFlaskApp(vn)
app.run()

 