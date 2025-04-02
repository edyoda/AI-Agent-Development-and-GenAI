import os
import json
import sqlite3
import requests
from typing import Dict, Any
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlalchemy import create_engine, text

# Updated LangChain imports
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader, CSVLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.utilities import WikipediaAPIWrapper, SQLDatabase
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain, create_sql_query_chain
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.tools import BaseTool
from langchain.agents.output_parsers import JSONAgentOutputParser

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")

# Validate required environment variables
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is not set")
if not WEATHER_API_KEY:
    print("Warning: WEATHER_API_KEY environment variable is not set - weather tool will not work")

# Initialize components
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
llm = ChatOpenAI(temperature=0, openai_api_key=OPENAI_API_KEY, model_name="gpt-3.5-turbo")
db_engine = create_engine("sqlite:///company.db")
db = SQLDatabase(db_engine)

def setup_demo_database():
    """Initialize the demo database with sample data"""
    conn = sqlite3.connect('company.db')
    cursor = conn.cursor()
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS employees (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        department TEXT NOT NULL,
        position TEXT NOT NULL,
        salary REAL CHECK(salary > 0),
        hire_date TEXT NOT NULL,
        email TEXT UNIQUE
    )''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS products (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        category TEXT NOT NULL,
        price REAL CHECK(price > 0),
        stock INTEGER DEFAULT 0,
        release_date TEXT
    )''')
    
    # Insert sample data if tables are empty
    if not conn.execute("SELECT 1 FROM employees LIMIT 1").fetchone():
        employees = [
            (1, 'John Smith', 'Engineering', 'Senior Developer', 120000, '2020-05-15', 'john.smith@company.com'),
            (2, 'Sarah Johnson', 'Marketing', 'Marketing Manager', 95000, '2019-10-23', 'sarah.j@company.com'),
            (3, 'Michael Wong', 'Engineering', 'Software Engineer', 90000, '2021-02-10', 'michael.w@company.com'),
            (4, 'Lisa Brown', 'HR', 'HR Director', 110000, '2018-07-19', 'lisa.b@company.com'),
            (5, 'James Wilson', 'Sales', 'Sales Representative', 85000, '2022-01-05', 'james.w@company.com')
        ]
        cursor.executemany('INSERT INTO employees VALUES (?,?,?,?,?,?,?)', employees)
    
    if not conn.execute("SELECT 1 FROM products LIMIT 1").fetchone():
        products = [
            (1, 'Laptop Pro', 'Electronics', 1299.99, 45, '2023-01-15'),
            (2, 'Smartphone X', 'Electronics', 899.99, 120, '2023-03-10'),
            (3, 'Office Chair', 'Furniture', 249.95, 30, '2022-11-01'),
            (4, 'Desk Lamp', 'Home', 49.99, 75, '2023-02-20'),
            (5, 'Wireless Headphones', 'Electronics', 159.99, 90, '2023-04-05')
        ]
        cursor.executemany('INSERT INTO products VALUES (?,?,?,?,?,?)', products)
    
    conn.commit()
    conn.close()

def setup_knowledge_base(docs_dir: str = "./documents") -> Chroma:
    """Initialize the document knowledge base"""
    os.makedirs(docs_dir, exist_ok=True)
        
    if not os.listdir(docs_dir):
        sample_docs = [
            {"filename": "company_overview.txt", "content": "Our company..."},
            {"filename": "product_roadmap.txt", "content": "Q1 2023: Launch..."}
        ]
        for doc in sample_docs:
            with open(os.path.join(docs_dir, doc["filename"]), "w", encoding='utf-8') as f:
                f.write(doc["content"])
    
    # Load documents
    loaders = {
        ".txt": (TextLoader, {"encoding": "utf-8"}),
        ".pdf": (PyPDFLoader, {}),
        ".csv": (CSVLoader, {"encoding": "utf-8"})
    }
    
    documents = []
    for ext, (loader_cls, loader_args) in loaders.items():
        try:
            loader = DirectoryLoader(docs_dir, glob=f"**/*{ext}", loader_cls=loader_cls, loader_kwargs=loader_args)
            documents.extend(loader.load())
        except Exception as e:
            print(f"Warning: Failed to load {ext} files: {e}")
    
    if not documents:
        raise ValueError("No documents were loaded")
    
    # Split and store documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    doc_chunks = text_splitter.split_documents(documents)
    return Chroma.from_documents(documents=doc_chunks, embedding=embeddings, persist_directory="./chroma_db")

class WeatherTool(BaseTool):
    name: str = "get_weather"
    description: str = "Get current weather and recommendations for a location"
    
    def _run(self, location: str) -> str:
        """Get weather information as formatted string"""
        if not WEATHER_API_KEY:
            return "Weather API key not configured"
            
        try:
            response = requests.get(
                "https://api.weatherapi.com/v1/current.json",
                params={"key": WEATHER_API_KEY, "q": location, "aqi": "no"},
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            
            current = data["current"]
            loc = data["location"]
            condition = current["condition"]["text"].lower()
            
            # Generate recommendations
            umbrella = "rain" in condition or current.get("precip_mm", 0) > 0.5
            jacket = current["temp_c"] < 15
            windy = current["wind_kph"] > 20
            
            return (
                f"Weather in {loc['name']}, {loc['country']}:\n"
                f"- Temperature: {current['temp_c']}°C ({current['temp_f']}°F)\n"
                f"- Condition: {current['condition']['text']}\n"
                f"- Wind: {current['wind_kph']} km/h\n"
                f"- Humidity: {current['humidity']}%\n\n"
                "Recommendations:\n"
                f"{'- ☔ Take an umbrella (rain expected)' if umbrella else '- No umbrella needed'}\n"
                f"{'- 🧥 Wear a jacket' if jacket else ''}\n"
                f"{'- 🪁 Windy conditions' if windy else ''}\n"
                f"\nLast updated: {current['last_updated']}"
            )
        except Exception as e:
            return f"Error getting weather: {str(e)}"

def query_database(query: str) -> str:
    """Execute natural language query against database"""
    try:
        sql = create_sql_query_chain(llm, db).invoke({"question": query})
        with db_engine.connect() as conn:
            result = conn.execute(text(sql))
            return json.dumps([dict(row) for row in result], indent=2)
    except Exception as e:
        return f"Database error: {str(e)}"
    
def debug_wikipedia(query):
    wikipedia = WikipediaAPIWrapper(top_k_results=2)
    result = wikipedia.run(query)
    print(f"DEBUG - Wikipedia result: {result}")  # Add this line
    return result


def build_knowledge_assistant():
    """Configure and return the assistant agent"""
    setup_demo_database()
    vectorstore = setup_knowledge_base()
    
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="output"
    )
    
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        memory=memory,
        return_source_documents=True,
        verbose=False
    )
    
    tools = [
        Tool(
            name="Company Knowledge",
            func=lambda q: qa_chain.invoke({"question": q})["output"],
            description="For company policies and documents"
        ),
        Tool(
            name="Company Database",
            func=query_database,
            description="For employee and product data"
        ),
        WeatherTool(),
        Tool(
            name="Wikipedia",
            func=lambda q: WikipediaAPIWrapper(top_k_results=2).run(q),
            description="For general knowledge questions"
        )
    ]
    
    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        memory=memory,
        verbose=True,
        handle_parsing_errors=True
    )
    
    return agent

# FastAPI App
app = FastAPI(title="AI Knowledge Assistant API")

class QueryRequest(BaseModel):
    query: str

@app.post("/query")
async def query_assistant(request: QueryRequest):
    """API endpoint for queries"""
    try:
        agent = build_knowledge_assistant()
        response = agent.invoke({"input": request.query})
        return {"response": response["output"]}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

def main():
    """Command-line interface"""
    print("\nAI Knowledge Assistant (type 'exit' to quit)\n" + "="*50)
    agent = build_knowledge_assistant()
    
    while True:
        try:
            query = input("\nYou: ").strip()
            if query.lower() in ["exit", "quit"]:
                break
                
            # Process the query and extract the final answer
            response = agent.invoke({"input": query})
            # Properly handle the response structure
            if isinstance(response, dict):
                if "output" in response:
                    print(f"\nAssistant: {response['output']}\n" + "-"*50)
                elif "result" in response:
                    print(f"\nAssistant: {response['result']}\n" + "-"*50)
                else:
                    print("\nAssistant: Received unexpected response format")
                    print("Full response:", response)
            else:
                print(f"\nAssistant: {response}\n" + "-"*50)
                
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {str(e)}\nPlease try again.")


if __name__ == "__main__":
    main()