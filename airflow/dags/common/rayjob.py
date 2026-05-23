import datetime
from airflow.sdk import BaseSensorOperator
from kubernetes import client, config
import yaml

class RayJobSensor(BaseSensorOperator):
    # Airflow가 렌더링할 템플릿 필드 정의
    template_fields = ("manifest", "namespace")

    def __init__(self, task_id: str, manifest: str, namespace: str, **kwargs):
        super().__init__(task_id=task_id, **kwargs)
        self.manifest = manifest
        self.namespace = namespace
        self.job_name = None  # 생성된 잡 이름을 저장할 변수

    def execute(self, context):
        """[단계 1] 태스크가 시작되자마자 딱 한 번 RayJob을 생성합니다."""
        config.load_incluster_config()
        api = client.CustomObjectsApi()
        
        body = yaml.safe_load(self.manifest)
        # 타임스탬프를 조합해 고유한 잡 이름 생성
        self.job_name = f"{body['metadata']['name']}-{int(datetime.datetime.now().timestamp())}"
        body["metadata"]["name"] = self.job_name
        body["metadata"]["namespace"] = self.namespace

        self.log.info(f"Creating RayJob: {self.job_name}")
        try:
            api.create_namespaced_custom_object(
                group="ray.io", version="v1",
                namespace=self.namespace, plural="rayjobs", body=body,
            )
        except client.exceptions.ApiException as e:
            if e.status == 409:
                self.log.info(f"RayJob {self.job_name} already exists. Proceeding to poke.")
            else:
                raise
        
        # 💡 중요: 생성된 job_name을 XCom에 저장해서 뒤의 태스크(모델 등록)가 쓸 수 있게 합니다.
        context['ti'].xcom_push(key='return_value', value=self.job_name)

        # 부모(BaseSensorOperator)의 execute를 호출하여 [단계 2] poke(상태 폴링)를 시작합니다.
        super().execute(context)

    def poke(self, context):
        """[단계 2] RayJob이 끝날 때까지 주기적으로 상태를 체크합니다."""
        config.load_incluster_config()
        api = client.CustomObjectsApi()
        
        # execute 단계에서 생성된 job_name으로 조회합니다.
        obj = api.get_namespaced_custom_object(
            group="ray.io", version="v1",
            namespace=self.namespace,
            plural="rayjobs", name=self.job_name,
        )
        status = obj.get("status", {}).get("jobStatus")
        self.log.info(f"RayJob {self.job_name} status: {status}")
        
        if status in ("FAILED", "STOPPED"):
            raise Exception(f"RayJob failed: {status}")
            
        return status == "SUCCEEDED"