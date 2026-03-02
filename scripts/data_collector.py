import json
import os
import sys
from datetime import datetime

# 프로젝트 루트를 path에 추가하여 부모 디렉토리의 모듈을 가져올 수 있게 함
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.agents import run_purchasing_pipeline_graph
from config import settings

def process_synthetic_scenarios(input_file: str, output_file: str):
    """
    합성 데이터 시나리오(JSONL)를 읽어와서 GPT-4o(Teacher) 파이프라인을 실행하고 결과를 저장합니다.
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
        
        # 파이프라인 입력 구성 (합성 히스토리 포함)
        input_data = {
            "snapshot_date": datetime.now().strftime("%Y-%m-%d"),
            "supplier": scene.get("supplier"),
            "items": scene.get("input_data"),
            "supplier_history_override": scene.get("supplier_history"),
            "item_history_override": scene.get("item_history")
        }

        # Teacher 모델(GPT-4o) 실행
        try:
            result = run_purchasing_pipeline_graph(input_data)
        except Exception as e:
            print(f"Error processing scenario {i+1}: {e}")
            continue
        
        # 학습용 트레이스 구성
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

    # 결과 저장
    with open(output_file, 'w', encoding='utf-8') as f:
        for t in traces:
            f.write(json.dumps(t, ensure_ascii=False) + '\n')
    
    print(f"Successfully processed {len(traces)} scenarios. Saved to {output_file}")

if __name__ == "__main__":
    # 합성 데이터 세트를 사용하여 Teacher 응답 생성
    synthetic_input = "training_data/synthetic_set.jsonl"
    final_output = f"training_data/teacher_dataset_{datetime.now().strftime('%Y%m%d')}.jsonl"
    
    process_synthetic_scenarios(synthetic_input, final_output)
