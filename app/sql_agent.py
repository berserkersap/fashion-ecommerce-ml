from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
from langchain.llms import HuggingFacePipeline
from langchain.agents.agent_types import AgentType
from transformers import pipeline
import torch
import os
from typing import List, Dict
from sqlalchemy import create_engine
from dotenv import load_dotenv

load_dotenv()

# Database configuration
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_NAME = os.getenv("DB_NAME")
INSTANCE_CONNECTION_NAME = os.getenv("DB_HOST")

# Create SQLAlchemy URI
db_uri = f"postgresql+pg8000://{DB_USER}:{DB_PASSWORD}@/{DB_NAME}?unix_sock=/cloudsql/{INSTANCE_CONNECTION_NAME}/.s.PGSQL.5432"

# Initialize database connection
db = SQLDatabase.from_uri(db_uri)

def get_sql_agent():
    """
    Create and return a SQL agent that can execute natural language queries
    using an open source model (CodeLlama)
    """
    # Initialize CodeLlama for SQL generation
    llm = HuggingFacePipeline(
        pipeline=pipeline(
            "text-generation",
            model="codellama/CodeLlama-7b-instruct-hf",
            torch_dtype=torch.float16,
            device="cuda" if torch.cuda.is_available() else "cpu",
            model_kwargs={"temperature": 0.1}
        )
    )
    
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    
    return create_sql_agent(
        llm=llm,
        toolkit=toolkit,
        verbose=True,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    )

def natural_language_query(query: str) -> Dict:
    """
    Execute a natural language query against the database
    """
    agent = get_sql_agent()
    try:
        result = agent.run(query)
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}

# Example usage:
# result = natural_language_query("Find all products in the shoes category ordered by price") 