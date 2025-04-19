# Flexible AutoGen CSV Analyzer
# Takes any CSV file as input and returns top 5 analyses
# Requirements: pip install pyautogen pandas matplotlib scikit-learn seaborn

import os
import json
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any
import autogen
from autogen import Agent, UserProxyAgent, AssistantAgent, GroupChat, GroupChatManager
import tempfile

# Set up argument parser for command line usage
parser = argparse.ArgumentParser(description='Analyze any CSV file with AutoGen multi-agent system')
parser.add_argument('--csv_path', type=str, required=True, help='Path to the CSV file to analyze')

def create_temp_dir():
    """Create a temporary directory for file exchange between agents."""
    temp_dir = tempfile.mkdtemp()
    print(f"Temporary directory created at: {temp_dir}")
    return temp_dir

def setup_agents(work_dir):
    """Set up the AutoGen agents with appropriate configurations."""
    
    # Use provided API key or get from environment
    api_key = os.environ.get("OPENAI_API_KEY")
    
    # Configuration for AutoGen
    config_list = [
        {
            "model": "gpt-4",
            "api_key": api_key
        }
    ]

    # Define the user proxy agent that will initiate the process
    user_proxy = UserProxyAgent(
        name="User",
        system_message="A human user who wants to analyze a CSV file.",
        code_execution_config={"work_dir": work_dir}
    )

    # Define the coordinator agent
    coordinator = AssistantAgent(
        name="Coordinator",
        system_message="You are the coordinator agent responsible for orchestrating the analysis process. Your job is to delegate tasks to specialized agents and ensure they work together to produce the top 5 insights. Make sure each analysis is thorough and provide clear instructions to each agent.",
        llm_config={"config_list": config_list}
    )

    # Define the data processor agent (combines loading and profiling)
    data_processor = AssistantAgent(
        name="DataProcessor",
        system_message="You are the data processor agent. Your responsibilities: 1. Load the CSV file provided by the user 2. Perform initial validation, cleaning and preprocessing 3. Generate statistical summaries and visualizations of the dataset 4. Identify the types of data and potential analyses that would be valuable. When given a task, respond with Python code that loads, processes and profiles the CSV data.",
        llm_config={"config_list": config_list}
    )

    # Define the analysis agent
    analyzer = AssistantAgent(
        name="Analyzer",
        system_message="You are the analysis agent. Your responsibilities: 1. Perform comprehensive analysis on the dataset including: - Correlation analysis between variables - Trend analysis for any time-series data - Distribution analysis and outlier detection - Clustering to find natural groupings - Predictive modeling to identify key relationships 2. For each analysis, calculate a significance score based on statistical significance and potential business value 3. Generate clear visualizations for each major finding 4. Save each insight using the helper functions. When given a task, respond with Python code to analyze the data thoroughly.",
        llm_config={"config_list": config_list}
    )

    # Define the insights reporter agent
    insights_reporter = AssistantAgent(
        name="InsightsReporter",
        system_message="You are the insights reporter agent. Your responsibilities: 1. Review all analyses performed by the Analyzer 2. Select the top 5 most valuable insights based on: - Statistical significance - Business relevance - Novelty/unexpectedness - Actionability 3. Create a clear, concise report summarizing these insights 4. Include key visualizations and data points that support each insight 5. Ensure the report is easy for non-technical stakeholders to understand. When given a task, respond with a well-formatted markdown report of the top 5 insights.",
        llm_config={"config_list": config_list}
    )

    return user_proxy, coordinator, data_processor, analyzer, insights_reporter, config_list

def write_helper_functions(work_dir):
    """Write helper functions to a file that can be imported by all agents."""
    helper_code = '''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
import json
import os
from datetime import datetime

# Global variable to store insights
INSIGHTS_FILE = "insights.json"

def save_insight(title, description, importance_score, visualization_path=None, data=None):
    """Save an insight to a JSON file."""
    insight = {
        "title": title,
        "description": description,
        "importance_score": importance_score,
        "visualization_path": visualization_path,
        "data": data,
        "timestamp": datetime.now().isoformat()
    }
    
    # Check if insights file exists
    if os.path.exists(INSIGHTS_FILE):
        with open(INSIGHTS_FILE, "r") as f:
            insights = json.load(f)
    else:
        insights = []
    
    insights.append(insight)
    
    with open(INSIGHTS_FILE, "w") as f:
        json.dump(insights, f, indent=2)
    
    return f"Insight saved: {title} with importance score {importance_score}"

def get_all_insights():
    """Get all insights from the JSON file."""
    if os.path.exists(INSIGHTS_FILE):
        with open(INSIGHTS_FILE, "r") as f:
            return json.load(f)
    return []

def get_top_insights(n=5):
    """Get the top N insights by importance score."""
    insights = get_all_insights()
    insights.sort(key=lambda x: x.get('importance_score', 0), reverse=True)
    return insights[:n]

def detect_data_types(df):
    """Detect and categorize column data types."""
    type_info = {
        "numeric": [],
        "categorical": [],
        "datetime": [],
        "text": [],
        "boolean": [],
        "id": []
    }
    
    # Check each column
    for col in df.columns:
        # Skip columns with all missing values
        if df[col].isna().all():
            continue
            
        # Check if column name suggests it's an ID
        if col.lower().endswith('id') or col.lower().endswith('_id') or col.lower() == 'id':
            type_info["id"].append(col)
            continue
            
        # Check if boolean
        if df[col].dtype == bool or (df[col].nunique() == 2 and set(df[col].dropna().unique()).issubset({0, 1, True, False})):
            type_info["boolean"].append(col)
            continue
            
        # Check if numeric
        if pd.api.types.is_numeric_dtype(df[col]):
            type_info["numeric"].append(col)
            continue
            
        # Check if datetime
        try:
            pd.to_datetime(df[col])
            type_info["datetime"].append(col)
            continue
        except:
            pass
            
        # Check if categorical (few unique values) or text (many unique values)
        if df[col].dtype == 'object':
            n_unique = df[col].nunique()
            n_total = len(df[col].dropna())
            
            # If less than 10% of values are unique, consider it categorical
            if n_unique < min(20, n_total * 0.1):
                type_info["categorical"].append(col)
            else:
                type_info["text"].append(col)
                
    return type_info

def summarize_dataframe(df):
    """Generate a comprehensive summary of the dataframe."""
    summary = {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_values": df.isna().sum().to_dict(),
        "numeric_summary": {},
        "categorical_summary": {}
    }
    
    # Get data types
    data_types = detect_data_types(df)
    summary["column_types"] = data_types
    
    # Summarize numeric columns
    for col in data_types["numeric"]:
        summary["numeric_summary"][col] = {
            "min": float(df[col].min()) if not pd.isna(df[col].min()) else None,
            "max": float(df[col].max()) if not pd.isna(df[col].max()) else None,
            "mean": float(df[col].mean()) if not pd.isna(df[col].mean()) else None,
            "median": float(df[col].median()) if not pd.isna(df[col].median()) else None,
            "std": float(df[col].std()) if not pd.isna(df[col].std()) else None
        }
    
    # Summarize categorical columns
    for col in data_types["categorical"]:
        value_counts = df[col].value_counts().to_dict()
        # Convert keys to strings to ensure JSON serialization works
        value_counts = {str(k): v for k, v in value_counts.items()}
        summary["categorical_summary"][col] = value_counts
    
    return summary

def create_visualizations(df, output_dir="visualizations"):
    """Create standard visualizations based on data types."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    visualization_paths = []
    data_types = detect_data_types(df)
    
    # Correlation heatmap for numeric columns
    if len(data_types["numeric"]) > 1:
        plt.figure(figsize=(12, 10))
        correlation_matrix = df[data_types["numeric"]].corr()
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", 
                   square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
        plt.title("Correlation Heatmap")
        plt.tight_layout()
        path = f"{output_dir}/correlation_heatmap.png"
        plt.savefig(path)
        plt.close()
        visualization_paths.append(path)
    
    # Distribution plots for key numeric columns (up to 6)
    if data_types["numeric"]:
        key_numerics = data_types["numeric"][:min(6, len(data_types["numeric"]))]
        for col in key_numerics:
            plt.figure(figsize=(10, 6))
            sns.histplot(df[col].dropna(), kde=True)
            plt.title(f"Distribution of {col}")
            plt.tight_layout()
            path = f"{output_dir}/distribution_{col.replace(' ', '_')}.png"
            plt.savefig(path)
            plt.close()
            visualization_paths.append(path)
    
    # Bar charts for categorical columns (up to 6)
    if data_types["categorical"]:
        key_cats = data_types["categorical"][:min(6, len(data_types["categorical"]))]
        for col in key_cats:
            plt.figure(figsize=(10, 6))
            value_counts = df[col].value_counts().nlargest(10)  # Limit to top 10 categories
            sns.barplot(x=value_counts.index.astype(str), y=value_counts.values)
            plt.title(f"Frequency of {col}")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            path = f"{output_dir}/categorical_{col.replace(' ', '_')}.png"
            plt.savefig(path)
            plt.close()
            visualization_paths.append(path)
    
    # Time series plots if datetime columns exist
    if data_types["datetime"] and data_types["numeric"]:
        date_col = data_types["datetime"][0]  # Use first datetime column
        key_metrics = data_types["numeric"][:min(3, len(data_types["numeric"]))]
        
        for metric in key_metrics:
            plt.figure(figsize=(12, 6))
            # Convert to datetime and sort
            df_temp = df[[date_col, metric]].copy()
            df_temp[date_col] = pd.to_datetime(df_temp[date_col])
            df_temp = df_temp.sort_values(date_col)
            
            # Resample data if there are too many points
            if len(df_temp) > 100:
                # Determine appropriate frequency
                date_range = (df_temp[date_col].max() - df_temp[date_col].min()).days
                if date_range > 365*2:  # More than 2 years
                    freq = 'M'  # Monthly
                elif date_range > 30*2:  # More than 2 months
                    freq = 'W'  # Weekly
                else:
                    freq = 'D'  # Daily
                
                df_temp = df_temp.set_index(date_col)
                df_temp = df_temp.resample(freq).mean()
                plt.plot(df_temp.index, df_temp[metric])
            else:
                plt.plot(df_temp[date_col], df_temp[metric])
                
            plt.title(f"{metric} Over Time")
            plt.xticks(rotation=45)
            plt.tight_layout()
            path = f"{output_dir}/timeseries_{metric.replace(' ', '_')}.png"
            plt.savefig(path)
            plt.close()
            visualization_paths.append(path)
    
    return visualization_paths
'''
    
    with open(f"{work_dir}/helper_functions.py", "w") as f:
        f.write(helper_code)
    
    print(f"Helper functions written to {work_dir}/helper_functions.py")

def analyze_csv(csv_path):
    """Main function to analyze a CSV file using the multi-agent system."""
    # Create working directory
    work_dir = create_temp_dir()
    
    # Write helper functions
    write_helper_functions(work_dir)
    
    # Set up agents
    user_proxy, coordinator, data_processor, analyzer, insights_reporter, config_list = setup_agents(work_dir)
    
    # Define the group chat for agent collaboration
    groupchat = GroupChat(
        agents=[user_proxy, coordinator, data_processor, analyzer, insights_reporter],
        messages=[],
        max_round=15
    )
    
    # Create the group chat manager
    manager = GroupChatManager(groupchat=groupchat, llm_config={"config_list": config_list})
    
    # Initialize the conversation
    initial_message = f"""
    I want to analyze the CSV file located at '{csv_path}' using the multi-agent system.
    Please coordinate the analysis process to:
    1. Load and preprocess the data
    2. Perform comprehensive analyses (correlation, trends, clustering, etc.)
    3. Generate the top 5 most valuable insights from this data
    
    Coordinator, please start by instructing the DataProcessor to load and profile the file.
    """
    
    # Start the analysis process
    result = user_proxy.initiate_chat(manager, message=initial_message)
    
    print("\n" + "="*80)
    print("Analysis Complete!")
    print("="*80)
    
    # Extract the final report from the conversation
    final_report = None
    for message in reversed(groupchat.messages):
        if message["role"] == "InsightsReporter" and "# Top 5 Insights" in message["content"]:
            final_report = message["content"]
            break
    
    if final_report:
        report_path = os.path.join(work_dir, "final_report.md")
        with open(report_path, "w") as f:
            f.write(final_report)
        print(f"\nFinal report saved to: {report_path}")
        print(f"Visualizations can be found in: {work_dir}/visualizations")
        
        # Print the report
        print("\n")
        print(final_report)
    else:
        print("No final report was generated. Please check the conversation history.")
    
    return work_dir, final_report

if __name__ == "__main__":
    # Parse command line arguments
    args = parser.parse_args()
    
    try:
        # Run the analysis
        work_dir, report = analyze_csv(args.csv_path)
        print(f"\nAnalysis complete. Results saved to {work_dir}")
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
