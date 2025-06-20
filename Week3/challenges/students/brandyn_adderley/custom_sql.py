######## Imports ##########

import yaml
from typing import Dict
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase

import pandas as pd
import sqlalchemy as sql
import os
import yaml
import re
from pprint import pprint
from IPython.display import Markdown



######### Create Functions ##############

class SchemaContext:
    def __init__(self, yaml_path):
        with open(yaml_path, 'r') as file:
            self.schema = yaml.safe_load(file)

    def get_dialect(self) -> str:
        """
        Return the SQL dialect specified in the YAML. Defaults to 'postgresql' if not specified.
        """
        return self.schema.get('dialect', 'postgresql')

    def format_for_prompt(self) -> str:
        formatted = "Tables:\n"
        for model in self.schema.get('models', []):
            table_name = model.get('name')
            table_description = model.get('description', 'No description provided')
            formatted += f"- {table_name}: {table_description}\n"
            formatted += "  Columns:\n"
            for col in model.get('columns', []):
                col_desc = f"    - {col['name']}: {col.get('description', 'No description provided')}"
                if col.get('sensitive', False):
                    col_desc += " [sensitive: true]"
                if col.get('aggregations'):
                    col_desc += f" [aggregations allowed: {', '.join(col['aggregations'])}]"
                if col.get('meanings'):
                    col_desc += f" [meanings: {', '.join(col['meanings'])}]"
                formatted += col_desc + "\n"

        formatted += "\nRelationships:\n"
        for rel in self.schema.get('relationships', []):
            formatted += f"- {rel['source_table']}.{rel['source_column']} → {rel['target_table']}.{rel['target_column']}\n"

        return formatted

    def format_examples_for_prompt(self) -> str:
        examples = self.schema.get('examples', [])
        formatted = ""
        for ex in examples:
            formatted += f'\nUser Request: "{ex["user_request"]}"\nSQL:\n{ex["sql"]}\n'
        return formatted

    def format_rules_for_prompt(self) -> str:
        rules = self.schema.get('rules', [])
        return "\n- " + "\n- ".join(rules)

    
    
    
    
# Build custom sql query chain


def create_custom_sql_query_chain(schema_context: SchemaContext, llm_model) -> callable:
    """
    Creates a callable SQL generator function that uses the schema context and an LLM model.
    """

    def generate_sql(user_question: str) -> Dict[str, str]:
        schema_info = schema_context.format_for_prompt()
        examples_text = schema_context.format_examples_for_prompt()
        rules_text = schema_context.format_rules_for_prompt()
        sql_dialect = schema_context.get_dialect()

        prompt = f"""
                    You are an expert SQL developer.

                    Use the SQL dialect: {sql_dialect}

                    Here is the database schema and relationships you must use:
                    {schema_info}

                    Here are rules you must strictly follow when generating SQL:
                    {rules_text}

                    Here are examples of user requests and their corresponding SQL queries:
                    {examples_text}

                    Now, for the new user request:
                    "{user_question}"

                    Write the SQL query only. Follow the rules, examples, relationships, and SQL dialect exactly. No explanation.
                    """

        response = llm_model.invoke(prompt)
        sql_code = response.content.strip()

        # Optional: Add LIMIT if missing
        # if sql_dialect.lower() in ['postgresql', 'mysql', 'sqlite'] and "limit" not in sql_code.lower():
        #     sql_code = sql_code.rstrip(';') + "\nLIMIT 100;"

        return {"sql": sql_code}

    return generate_sql



# 1. Load Schema Context
schema_context = SchemaContext('database_context.yml')

# 2. Initialize Model
os.environ["OPENAI_API_KEY"] = yaml.safe_load(open('/Users/brandyn/Desktop/Business Science/credentials.yml'))['openai']

llm_model = ChatOpenAI(
    model = "gpt-4.1-nano",
    temperature = 0.7,
)


# 3. Create Custom SQL Query Chain
sql_query_chain = create_custom_sql_query_chain(schema_context, llm_model)

# 4. Ask a User Question
user_question = "Show me total revenue by product category over the last 12 months from the latest date in the table."

# 5. Generate SQL
result = sql_query_chain(user_question)

# 6. Print the SQL output
print("Generated SQL Query:")
print(result["sql"])

sql_query = result["sql"]
# DATABASE SETUP

PATH_DB = "sqlite:///database/leads_scored.db"

sql_engine = sql.create_engine(PATH_DB)

conn = sql_engine.connect()


# select all table names
pd.read_sql(sql_query, conn)
