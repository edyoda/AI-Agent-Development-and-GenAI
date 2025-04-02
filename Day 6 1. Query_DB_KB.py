import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from langchain.utilities import SQLDatabase
from langchain_openai import OpenAI
from langchain.chains import create_sql_query_chain

# Load environment variables
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI
llm = OpenAI(api_key=API_KEY)

# Create SQLAlchemy engine and database object
db_engine = create_engine("sqlite:///company.db")
db = SQLDatabase(db_engine)

def setup_demo_database():
    """Set up a demo database with sample data"""
    with db_engine.connect() as conn:
        # Create employees table
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS employees (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                department TEXT NOT NULL,
                salary REAL NOT NULL,
                hire_date TEXT NOT NULL
            )
        """))
        
        # Create projects table
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS projects (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                budget REAL NOT NULL,
                status TEXT NOT NULL
            )
        """))
        
        # Create employee_projects table (many-to-many relationship)
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS employee_projects (
                employee_id INTEGER,
                project_id INTEGER,
                role TEXT NOT NULL,
                FOREIGN KEY (employee_id) REFERENCES employees (id),
                FOREIGN KEY (project_id) REFERENCES projects (id)
            )
        """))
        
        # Insert sample data if tables are empty
        result = conn.execute(text("SELECT COUNT(*) FROM employees"))
        if result.scalar() == 0:
            # Insert employees
            conn.execute(text("""
                INSERT INTO employees (name, department, salary, hire_date)
                VALUES 
                    ('John Smith', 'Engineering', 120000, '2020-01-15'),
                    ('Jane Doe', 'Marketing', 95000, '2021-03-20'),
                    ('Bob Johnson', 'Sales', 110000, '2019-11-05'),
                    ('Alice Brown', 'Engineering', 130000, '2020-06-10'),
                    ('Charlie Wilson', 'Marketing', 98000, '2021-08-15')
            """))
            
            # Insert projects
            conn.execute(text("""
                INSERT INTO projects (name, budget, status)
                VALUES 
                    ('AI Platform', 500000, 'In Progress'),
                    ('Mobile App', 300000, 'Completed'),
                    ('Website Redesign', 200000, 'In Progress'),
                    ('Data Analytics', 400000, 'Planning')
            """))
            
            # Insert employee_projects
            conn.execute(text("""
                INSERT INTO employee_projects (employee_id, project_id, role)
                VALUES 
                    (1, 1, 'Lead Developer'),
                    (1, 2, 'Developer'),
                    (2, 3, 'Project Manager'),
                    (3, 1, 'Business Analyst'),
                    (4, 1, 'Developer'),
                    (4, 4, 'Lead Developer')
            """))
            
        conn.commit()

def query_database(question: str):
    """Query the database using natural language"""
    try:
        # Create SQL chain
        sql_chain = create_sql_query_chain(llm, db)
        
        # Generate SQL query
        sql_query = sql_chain.invoke({"question": question})
        print(f"Generated SQL Query: {sql_query}\n")
        
        # Execute query
        result = db.run(sql_query)
        return result
        
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    # Set up demo database
    setup_demo_database()
    
    # Example queries
    questions = [
        "What is John Smith's salary?",
        "List all employees in the Engineering department",
        "What is the average salary by department?",
        "Which projects is John Smith working on?",
        "What is the total budget for all projects?",
        "Who are the lead developers?",
        "How many employees are working on the AI Platform project?",
        "What is the average salary of employees hired in 2020?"
    ]
    
    # Run queries
    for question in questions:
        print(f"\nQuestion: {question}")
        result = query_database(question)
        print(f"Result: {result}")