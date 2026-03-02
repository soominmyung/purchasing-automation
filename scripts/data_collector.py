import json
import os
import sys
from datetime import datetime

# Add project root to path for parent directory module imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.agents import run_purchasing_pipeline_graph
from config import settings

def process_synthetic_scenarios(input_file: str, output_file: str):
    """
    Reads synthetic data scenarios (JSONL) and runs the GPT-4o (Teacher) pipeline, saving results.
    """
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} not found.")
        return

    scenarios = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                scenarios.append(json.loads(line))

    print(f"Total scenarios to process: {len(scenarios)}")
    
    traces = []
    for i, scene in enumerate(scenarios):
        print(f"[{i+1}/{len(scenarios)}] Processing scenario for supplier: {scene.get('supplier')}...")
        
        # Construct pipeline input (include synthetic history)
        input_data = {
            "snapshot_date": datetime.now().strftime("%Y-%m-%d"),
            "supplier": scene.get("supplier"),
            "items": scene.get("input_data"),
            "supplier_history_override": scene.get("supplier_history"),
            "item_history_override": scene.get("item_history")
        }

        # Run Teacher model (GPT-4o)
        try:
            result = run_purchasing_pipeline_graph(input_data)
        except Exception as e:
            print(f"Error processing scenario {i+1}: {e}")
            continue
        
        # Construct training trace
        trace = {
            "instruction": "Analyze the purchasing data and supplier/item history to generate reports and drafts.",
            "input": {
                "inventory": input_data["items"],
                "supplier_history": input_data["supplier_history_override"],
                "item_history": input_data["item_history_override"]
            },
            "output": {
                "analysis": result.get("analysis_output"),
                "report": result.get("report_md"),
                "pr": result.get("pr_md"),
                "email": result.get("email_text"),
                "evaluation": result.get("evaluation_md")
            }
        }
        traces.append(trace)

    # Save results
    with open(output_file, 'w', encoding='utf-8') as f:
        for t in traces:
            f.write(json.dumps(t, ensure_ascii=False) + '\n')
    
    print(f"Successfully processed {len(traces)} scenarios. Saved to {output_file}")

if __name__ == "__main__":
    # Generate teacher responses using synthetic dataset
    synthetic_input = "training_data/synthetic_set.jsonl"
    final_output = f"training_data/teacher_dataset_{datetime.now().strftime('%Y%m%d')}.jsonl"
    
    process_synthetic_scenarios(synthetic_input, final_output)
