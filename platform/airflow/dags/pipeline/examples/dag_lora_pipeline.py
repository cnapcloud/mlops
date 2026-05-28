import datetime
import os
import json
from airflow import DAG
from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
from airflow.providers.standard.operators.python  import BranchPythonOperator
from airflow.providers.standard.operators.empty import EmptyOperator

# 템플릿 경로 설정
import manifests
TEMPLATE_PATH =  os.path.join(manifests.MANIFEST_ROOT,'lora_pipeline_template.yaml')

# Data Analysis 스크립트
DATA_ANALYSIS_SCRIPT = """
import os
import json

print("1. [분석] 대화형 데이터셋 토큰 수량 및 정밀 분석 시작...")

dataset_status = "QUALIFIED"

xcom_output_path = "/airflow/xcom/return.json"
os.makedirs(os.path.dirname(xcom_output_path), exist_ok=True)

output_data = {
    "dataset_status": dataset_status,
    "raw_path": "/workspace/data/raw_dataset.jsonl"
}

with open(xcom_output_path, "w") as f:
    json.dump(output_data, f)

print(f"분석 완료. 결과: {output_data}")
"""


# 태스크 실패 시 호출될 공통 콜백 함수 (실패 고려)
def on_task_failure_callback(context):
    task_id = context['task_instance'].task_id
    execution_date = context['execution_date']
    error = context.get('exception')
    print(f"🚨 [ALERT] 태스크 실패 알림: '{task_id}'가 실패했습니다. 실행일시: {execution_date}. 에러내용: {error}")
    # 여기에 Slack Webhook이나 이메일 연동 코드를 삽입합니다.

default_args = {
    'owner': 'mlops_engineer',
    'start_date': datetime.datetime(2026, 1, 1),
    'retries': 1,                               # 1회 기본 재시도 구성
    'retry_delay': datetime.timedelta(minutes=5),# 재시도 대기 시간
    'on_failure_callback': on_task_failure_callback, # 실패 시 Alert 연동
}

with DAG(
    dag_id='example_lora_pipeline',
    default_args=default_args,
    schedule=None,
    catchup=False,
    tags=['llm', 'lora', 'peft'],
) as dag:

    # 1. 데이터 분석 (Data Analysis)
    # 데이터셋의 토큰 길이 분산, 결측치, 데이터 개수 분석
    data_analysis = KubernetesPodOperator(
        task_id='data_analysis',
        pod_template_file=TEMPLATE_PATH,
        cmds=['python', '-c'],
        arguments=[DATA_ANALYSIS_SCRIPT],
        do_xcom_push=True,
        on_finish_action="delete_succeeded_pod",
        get_logs=True,
    )
    
    # 1-2. 분석 결과에 따른 조건 분기 (Branching)
    def check_data_quality(**context):
        res = context['task_instance'].xcom_pull(task_ids='data_analysis')
        print(f"xcom raw value: {res}, type: {type(res)}")  # 타입 확인
        
        # 문자열로 올 경우 파싱
        if isinstance(res, str):
            import json
            res = json.loads(res)
        
        if res and res.get('dataset_status') == "QUALIFIED":
            return 'data_preprocessing'
        else:
            return 'pipeline_abort_by_data'

    branch_data_check = BranchPythonOperator(
        task_id='branch_data_check',
        python_callable=check_data_quality,
    )

    # 데이터 품질 미달 시 종료 경로
    pipeline_abort_by_data = EmptyOperator(
        task_id='pipeline_abort_by_data',
        on_success_callback=lambda context: print("데이터 품질 미달로 학습 파이프라인을 안전하게 종료합니다.")
    )

    # 2. 데이터 전처리 및 토큰화 (Data Preprocessing / Tokenization)
    # 프롬프트 템플릿(Alpaca 등) 주입 및 최대 토큰 길이에 맞게 잘라 파일로 캐싱
    data_preprocessing = KubernetesPodOperator(
        task_id='data_preprocessing',
        pod_template_file=TEMPLATE_PATH,
        cmds=['python', '-c'],
        arguments=[
            'import json, os\n'
            'print("2. [전처리] 프롬프트 엔지니어링 투입 및 Tokenization 수행 중...")\n'
            'os.makedirs("/airflow/xcom", exist_ok=True)\n'
            'with open("/airflow/xcom/return.json", "w") as f:\n'
            '    json.dump({"tokenized_path": "/workspace/data/processed_tokenized"}, f)\n'
        ],
        do_xcom_push=True,
        on_finish_action="delete_succeeded_pod",
        get_logs=True,
    )

    # 3. LoRA 학습 (LoRA Train) - ★가장 중요 및 리소스 다량 소모
    # GPU 리소스를 동적으로 YAML 스펙에 오버라이드 설정
    lora_training = KubernetesPodOperator(
        task_id='lora_training',
        pod_template_file=TEMPLATE_PATH,
        container_resources={
            'requests': {'memory': '100Mi', 'cpu': '100m'},
            'limits': {'memory': '1Gi', 'cpu': '1000m'} # GPU 1장 할당 필요('nvidia.com/gpu': '1')
        },
        cmds=['python', '-c'],
        arguments=[
            'import json, os\n'
            'print("3. [LoRA 학습] PEFT/HuggingFace 라이브러리를 사용해 고성능 파인튜닝 기동...");\n'
            'print("Epoch 1/3, Loss: 1.84... Epoch 3/3, Loss: 0.42")\n'
            'print("LoRA 어댑터 가중치(Adapter Weights) 저장 완료.")\n'
            'os.makedirs("/airflow/xcom", exist_ok=True)\n'
            'with open("/airflow/xcom/return.json", "w") as f:\n'
            '    json.dump({"adapter_dir": "/workspace/data/output/lora_adapter_v1"}, f)\n'
        ],
        do_xcom_push=True,
        on_finish_action="delete_succeeded_pod",
        get_logs=True,
    )

    # 4. 검증 및 평가 (Validation & Evaluation)
    # 검증 데이터셋(Validation Set)으로 Perplexity를 구하고, BLEU/ROUGE Score 또는 LLM-as-a-judge로 평가 수행
    model_evaluation = KubernetesPodOperator(
        task_id='model_evaluation',
        pod_template_file=TEMPLATE_PATH,
        container_resources={
            'requests': {'memory': '100Mi', 'cpu': '100m'},
            'limits': {'memory': '1Gi', 'cpu': '1000m'} # GPU 1장 할당 필요('nvidia.com/gpu': '1')
        },
        cmds=['python', '-c'],
        arguments=[
            'import json, os\n'
            'print("4. [평가] 추론 테스트 및 평가지표 계산 중...")\n'
            '# 가상의 가중치 평가 점수 (0.0 ~ 1.0)\n'
            'eval_loss = 0.35; accuracy_score = 0.88;\n'
            '# 평가 지표가 통과되었는지 여부 체크\n'
            'is_passed = "PASS" if accuracy_score >= 0.80 else "FAIL"\n'
            'os.makedirs("/airflow/xcom", exist_ok=True)\n'
            'with open("/airflow/xcom/return.json", "w") as f:\n'
            '    json.dump({"eval_status": is_passed, "accuracy": accuracy_score}, f)\n'
        ],
        do_xcom_push=True,
        on_finish_action="delete_succeeded_pod",
        get_logs=True,
    )

    # 4-2. 평가 점수 기반 분기 (Branching)
    def check_evaluation_score(**context):
        res = context['task_instance'].xcom_pull(task_ids='model_evaluation')
        
        if res.get('eval_status') == "PASS":
            return 'model_registration'
        else:
            return 'pipeline_abort_by_eval'

    branch_eval_check = BranchPythonOperator(
        task_id='branch_eval_check',
        python_callable=check_evaluation_score,
    )

    # 평가 실패(기준 미달) 시 종료 경로
    pipeline_abort_by_eval = EmptyOperator(
        task_id='pipeline_abort_by_eval',
        on_success_callback=lambda context: print("❌ [경고] 학습 모델의 정확도가 기준치 미달되어 등록을 취소합니다.")
    )

    # 5. 등록 (Model Registration)
    # 검증을 통과한 LoRA 어댑터 가중치를 중앙 가중치 저장소(HuggingFace Hub, MLflow, S3 등)로 업로드 및 배포 등록
    model_registration = KubernetesPodOperator(
        task_id='model_registration',
        pod_template_file=TEMPLATE_PATH,
        cmds=['python', '-c'],
        arguments=[
            'print("5. [등록] 검증이 통과된 LoRA 어댑터를 MLflow / HuggingFace Hub에 업로드합니다."); '
            'print("Successfully registered model as \'llm-lora-core:v1.0.0\'")'
        ],
        on_finish_action="delete_succeeded_pod",
        get_logs=True,
    )

    # 전체 성공 종착지
    pipeline_success_end = EmptyOperator(
        task_id='pipeline_success_end',
        trigger_rule='none_failed_min_one_success'
    )

    # --- 파이프라인 그래프 정의 ---
    # 데이터 분석 및 품질 체크 분기
    data_analysis >> branch_data_check
    branch_data_check >> data_preprocessing >> lora_training >> model_evaluation >> branch_eval_check
    
    # 1단계 조건 실패 경로 우회
    branch_data_check >> pipeline_abort_by_data >> pipeline_success_end

    # 4단계 조건 성공/실패 경로 분기
    branch_eval_check >> model_registration >> pipeline_success_end
    branch_eval_check >> pipeline_abort_by_eval >> pipeline_success_end