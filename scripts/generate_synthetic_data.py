import os
import sys
import json
import random
from datetime import datetime, timedelta
from typing import List, Dict

# Add project root to path for parent directory module imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openai import OpenAI
from config import settings

client = OpenAI(api_key=settings.openai_api_key)

SYSTEM_PROMPT = """You are a Supply Chain Data Generator. 
Your goal is to generate realistic, diverse purchasing scenarios based on a specific CSV schema.

CSV SCHEMA (from 050425.csv):
Columns: ItemCode, ItemName, SupplierName, RiskLevel, CurrentStock, WksToOOS

EXAMPLE DATA:
100000,ItemA,SupplierA,High,100,25
100001,ItemB,SupplierA,Medium,90,41

Your goal is to generate realistic, diverse, and ROBUST purchasing scenarios. 

HISTORY CATEGORIES TO INCLUDE:
- Supplier: [Delivery Delays, Price Spikes, Quality Issues, MOQ changes, Communication response]
- Item: [Demand Volatility, Seasonal Spikes, Substitution History, Lead Time changes]

Each scenario object must have:
1. 'input_data': A list of items (JSON objects).
2. 'supplier': The name of the primary supplier.
3. 'supplier_history': Natural language text regarding the supplier's history.
4. 'item_history': Natural language text regarding the items' combined history.

ROBUSTNESS (Adversarial) SCENARIOS:
- Data Inconsistency: High Stock but low WksToOOS (Sudden spike).
- Conficting Context: Supplier is marked 'Reliable' in metadata but 'supplier_history' shows a recent failure.
- Vague History: History contains irrelevant or ambiguous information.
- Missing Values: Some minor fields are N/A or empty.
"""

def generate_scenario_batch(batch_size: int = 5) -> List[Dict]:
    """Generate synthetic data sets using GPT-4o."""
    prompt = f"Generate {batch_size} unique purchasing scenarios in JSON format. The root object MUST have a key 'scenarios' containing a list of objects. Each scenario object must have 'input_data', 'supplier', 'supplier_history', and 'item_history'."
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        response_format={ "type": "json_object" }
    )
    
    content = response.choices[0].message.content
    try:
        data = json.loads(content)
        scenarios = data.get("scenarios", [])
        if not scenarios:
            print(f"Warning: No scenarios found in response. Root keys: {list(data.keys())}")
        return scenarios
    except Exception as e:
        print(f"Error parsing JSON: {e}")
        print(f"Raw content: {content}")
        return []

def save_synthetic_data(scenarios: List[Dict], output_file: str = "training_data/synthetic_set.jsonl"):
    """Save generated scenarios to a JSONL file."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "a", encoding="utf-8") as f:
        for s in scenarios:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    print("Generating 30 synthetic scenarios...")
    # Generate 30 scenarios and save (6 batch calls)
    # Use 'w' mode on first save to overwrite existing file
    output_path = "training_data/synthetic_set.jsonl"
    if os.path.exists(output_path):
        os.remove(output_path)
    
    for i in range(6):
        print(f"Batch {i+1}/6 generating...")
        scenes = generate_scenario_batch(5)
        save_synthetic_data(scenes, output_path)
    print("Data generation complete. Check training_data/synthetic_set.jsonl")
