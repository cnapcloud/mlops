"""
NYC Taxi MLOps Pipeline - Step 3: Training
==========================================
목적: fare_amount(요금) + trip_duration_min(ETA) 두 모델을 학습하고
      MLflow Model Registry에 Staging으로 등록한다.

설계 원칙:
  - 모델은 항상 MLflow에 저장 (파일 저장 없음)
  - 로컬: 드라이버에서 학습 → mlflow.xgboost.log_model()
  - 분산: Ray 워커(rank=0) 내부에서 학습 완료 후 MLflow에 직접 저장
           드라이버는 run_id만 수신 → Registry Staging 등록

Import 전략:
  - 파일 상단: 드라이버에서 항상 필요한 것만 (mlflow, pandas, sklearn)
  - 함수 내부: ray 관련 (ray가 없는 환경에서 --local 실행 가능하도록)
  - 콜백 내부: 워커 프로세스에서 실행되는 코드 (반드시 콜백 안에서 import)

실행 모드:
  A. 로컬
     python step3_train.py --input data/raw/ --local

  B. KubeRay
     python step3_train.py --input data/raw/ \
       --ray-address ray://192.168.0.182:10001

의존 패키지:
    pip install ray[data,train] xgboost mlflow pyarrow pandas scikit-learn
"""

import argparse
import logging
import uuid
from pathlib import Path

# ── 드라이버 상단 import: ray 없이도 동작해야 하는 것만
import mlflow
import mlflow.xgboost
import pandas as pd
from mlflow import MlflowClient
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# 0. 설정
# ──────────────────────────────────────────────
DEFAULT_XGB_PARAMS = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "tree_method": "hist",
    "objective": "reg:squarederror",
    "random_state": 42,
}

FEATURE_COLS = [
    "pickup_hour",
    "pickup_dayofweek",
    "pickup_month",
    "trip_distance",
    "passenger_count",
    "PULocationID",
    "DOLocationID",
    "is_weekend",
    "is_rush_hour",
]

# (타겟 컬럼, MLflow experiment명, Registry 모델명)
TARGETS = [
    ("fare_amount",       "nyc-taxi-fare-training", "nyc-taxi-fare"),
    ("trip_duration_min", "nyc-taxi-eta-training",  "nyc-taxi-eta"),
]


# ──────────────────────────────────────────────
# 1. 피처 엔지니어링
# ──────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["tpep_pickup_datetime"]  = pd.to_datetime(df["tpep_pickup_datetime"])
    df["tpep_dropoff_datetime"] = pd.to_datetime(df["tpep_dropoff_datetime"])

    df["trip_duration_min"] = (
        df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]
    ).dt.total_seconds() / 60

    df["pickup_hour"]      = df["tpep_pickup_datetime"].dt.hour
    df["pickup_dayofweek"] = df["tpep_pickup_datetime"].dt.dayofweek
    df["pickup_month"]     = df["tpep_pickup_datetime"].dt.month
    df["is_weekend"]       = (df["pickup_dayofweek"] >= 5).astype(int)
    df["is_rush_hour"]     = df["pickup_hour"].isin([7, 8, 9, 17, 18, 19]).astype(int)

    df = df[
        (df["fare_amount"] > 0)       & (df["fare_amount"] < 300) &
        (df["trip_distance"] > 0)     & (df["trip_distance"] < 100) &
        (df["trip_duration_min"] > 0) & (df["trip_duration_min"] < 180) &
        (df["passenger_count"] >= 1)  & (df["passenger_count"] <= 6)
    ]

    all_cols = FEATURE_COLS + ["fare_amount", "trip_duration_min"]
    return df[all_cols].dropna().reset_index(drop=True)


# ──────────────────────────────────────────────
# 2. 지표 계산 (드라이버 / 워커 공용)
# ──────────────────────────────────────────────
def _compute_metrics(y_true, y_pred, label: str) -> dict:
    rmse = float(mean_squared_error(y_true, y_pred) ** 0.5)
    mae  = float(mean_absolute_error(y_true, y_pred))
    r2   = float(r2_score(y_true, y_pred))
    unit = "$" if "fare" in label else "min"
    log.info("[%s] RMSE=%.4f%s  MAE=%.4f%s  R²=%.4f", label, rmse, unit, mae, unit, r2)
    return {"rmse": round(rmse, 4), "mae": round(mae, 4), "r2": round(r2, 4)}


# ──────────────────────────────────────────────
# 3. 로컬 학습 + MLflow 저장
#    드라이버 프로세스에서 실행 → 모델 객체 직접 사용 가능
# ──────────────────────────────────────────────
def train_and_log_local(
    df: pd.DataFrame,
    target_col: str,
    xgb_params: dict,
    experiment: str,
    registered_model_name: str,
    tags: dict,
) -> tuple[str, dict]:
    """
    Returns: (run_id, metrics)
    xgboost import: ray가 없는 환경에서도 동작하도록 함수 내부에서
    """
    import xgboost as xgb  # ray 없는 환경 고려 → 함수 내부 import

    X = df[FEATURE_COLS]
    y = df[target_col]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    log.info("[%s] local  train=%d / val=%d", target_col, len(X_train), len(X_val))
    model = xgb.XGBRegressor(**xgb_params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=100)

    metrics = _compute_metrics(y_val, model.predict(X_val), target_col)

    mlflow.set_experiment(experiment)
    with mlflow.start_run() as run:
        mlflow.log_params({**xgb_params, "target": target_col})
        mlflow.log_metrics(metrics)
        mlflow.set_tags(tags)
        mlflow.xgboost.log_model(
            model,
            name="model",
            registered_model_name=registered_model_name,
        )
        run_id = run.info.run_id

    log.info("[%s] local run_id=%s", target_col, run_id)
    return run_id, metrics


# ──────────────────────────────────────────────
# 4. 분산 학습 + MLflow 저장
#    - ray, XGBoostTrainer: 함수 내부 import  (드라이버, ray 제어용)
#    - mlflow, xgboost:     콜백 내부 import  (워커 프로세스에서 실행)
#    - ray.train:           콜백 내부 import  (워커에서 report 호출)
# ──────────────────────────────────────────────
def train_and_log_distributed(
    df: pd.DataFrame,
    target_col: str,
    xgb_params: dict,
    experiment: str,
    registered_model_name: str,
    tags: dict,
    ray_address: str,
    num_workers: int,
    mlflow_uri: str,
) -> tuple[str, dict]:
    """
    Returns: (run_id, metrics)

    XGBoostTrainer(train_loop_per_worker) 방식:
      - train_loop_per_worker 가 필수 positional argument
      - 워커에서 xgb.train() 직접 호출 → rank=0 에서 MLflow 저장
      - ray.train.report() 로 run_id / metrics 를 드라이버에 전달
    """
    import ray
    from ray.train import RunConfig, ScalingConfig
    from ray.train.xgboost import XGBoostTrainer

    # 워커로 전달할 직렬화 가능한 설정
    mlflow_cfg = {
        "tracking_uri":          mlflow_uri,
        "experiment":            experiment,
        "registered_model_name": registered_model_name,
        "params":                {**xgb_params, "target": target_col},
        "tags":                  tags,
        "target_col":            target_col,
    }
    # n_estimators 는 xgb.train() 의 num_boost_round 로 사용
    num_boost_round = xgb_params.get("n_estimators", 300)
    # xgb.train() 에 넘길 파라미터 (n_estimators / random_state 는 XGBRegressor 전용)
    xgb_core_params = {
        k: v for k, v in xgb_params.items()
        if k not in ("n_estimators", "random_state")
    }

    # ── train_loop_per_worker: 워커 프로세스에서 실행
    #    클로저로 mlflow_cfg / xgb_core_params / num_boost_round / target_col 캡처
    def train_loop_per_worker() -> None:
        # 워커 프로세스 import
        import xgboost as xgb
        import ray.train as rt
        import mlflow as _mlflow
        import mlflow.xgboost as _mlflow_xgb
        import logging

        # 💡 로거 설정 (드라이버로 전송될 표준 로거)
        logger = logging.getLogger("ray.train")
        
        # 워커의 랭크 확인 (몇 번 워커인지 식별용)
        world_rank = rt.get_context().get_world_rank()
        logger.info(f"[Rank {world_rank}] 워커 프로세스 학습 루프 시작")

        # ── 데이터 수신
        logger.info(f"[Rank {world_rank}] 데이터셋 샤드 수신 중...")
        train_shard = rt.get_dataset_shard("train")
        val_shard   = rt.get_dataset_shard("validation")

        train_frame = train_shard.materialize().to_pandas()
        val_frame   = val_shard.materialize().to_pandas()
        logger.info(f"[Rank {world_rank}] 데이터 변환 완료 (Train: {len(train_frame)}행, Val: {len(val_frame)}행)")

        train_features = train_frame.drop(columns=[target_col])
        train_labels   = train_frame[target_col]
        val_features   = val_frame.drop(columns=[target_col])
        val_labels     = val_frame[target_col]

        dtrain = xgb.DMatrix(train_features, label=train_labels)
        dval   = xgb.DMatrix(val_features,   label=val_labels)

        # ── 학습
        logger.info(f"[Rank {world_rank}] 🏋️ XGBoost 학습 시작...")
        evals_result: dict = {}
        booster = xgb.train(
            xgb_core_params,
            dtrain          = dtrain,
            evals           = [(dval, "validation")],
            num_boost_round = num_boost_round,
            evals_result    = evals_result,
            verbose_eval    = 100,
        )

        val_rmse = evals_result.get("validation", {}).get("rmse", [None])[-1]
        logger.info(f"[Rank {world_rank}] XGBoost 학습 완료 (최종 Val RMSE: {val_rmse})")

        # ── rank=0 워커만 MLflow 저장
        if world_rank != 0:
            logger.info(f"[Rank {world_rank}] 💤 Rank 0이 아니므로 MLflow 저장을 건너뛰고 결과를 리포트합니다.")
            rt.report({"validation-rmse": val_rmse or 0.0})
            return

        # 여기서부터 Rank 0 워커의 MLflow 기록 영역입니다.
        logger.info(f"[Rank 0] MLflow 연동 시작 (Tracking URI: {mlflow_cfg['tracking_uri']})")
        
        try:
            cfg = mlflow_cfg
            _mlflow.set_tracking_uri(cfg["tracking_uri"])
            _mlflow.set_experiment(cfg["experiment"])

            logger.info(f"[Rank 0] MLflow Run 시작 중 (Experiment: {cfg['experiment']})")
            with _mlflow.start_run() as run:
                run_id = run.info.run_id
                logger.info(f"[Rank 0] MLflow Run 생성 완료 (Run ID: {run_id})")
                
                # 파라미터 및 태그 기록 로그
                _mlflow.log_params(cfg["params"])
                _mlflow.set_tags(cfg["tags"])
                logger.info(f"[Rank 0] 파라미터 및 태그 기록 완료")
                
                # 모델 등록 시작 지점
                logger.info(f"[Rank 0] MLflow에 모델 등록 중 (Model Name: {cfg['registered_model_name']})... 대기 시간이 걸릴 수 있습니다.")
                
                _mlflow_xgb.log_model(
                    booster,
                    name          = "model",
                    registered_model_name  = cfg["registered_model_name"],
                )
                logger.info(f"[Rank 0] MLflow에 모델 및 레지스트리 등록 최종 성공!")

        except Exception as e:
            # 🚨 MLflow 저장 중 에러가 발생하면 드라이버에 트레이스백을 찍도록 예외 처리
            logger.error(f"[Rank 0] MLflow 저장 중 에러 발생: {str(e)}", exc_info=True)
            raise e

        # run_id 를 드라이버로 전달
        rt.report({"validation-rmse": val_rmse or 0.0})
        logger.info(f"[Rank 0] 🏁 최종 결과 리포트 완료 및 워커 종료")

    # ── Ray 초기화
    ray.init(
        address          = ray_address,
        ignore_reinit_error = True,
        runtime_env      = {
            "pip": ["xgboost==3.2.0", "mlflow", "scikit-learn"],
            "env": {
                "RAY_DEFAULT_OBJECT_STORE_MEMORY_PROPORTION": "0.8"
            }            
        },
    )

    # pandas 단계에서 split → 각각 Ray Dataset 으로 변환
    # (ray.data.train_test_split 은 eager execute 로 autoscaler API 오류 발생)
    df_target = df[FEATURE_COLS + [target_col]]
    df_train  = df_target.sample(frac=0.8, random_state=42)
    df_val    = df_target.drop(df_train.index)
    train_ds  = ray.data.from_pandas(df_train.reset_index(drop=True))
    val_ds    = ray.data.from_pandas(df_val.reset_index(drop=True))

    run_name = f"nyc_taxi_{target_col}_{uuid.uuid4().hex[:8]}"

    trainer = XGBoostTrainer(
        train_loop_per_worker,
        datasets       = {"train": train_ds, "validation": val_ds},
        scaling_config = ScalingConfig(num_workers=num_workers, resources_per_worker={"CPU": 1}, use_gpu=False),
        run_config=RunConfig(name=run_name, storage_path="/shared/ray-checkpoints")
    )

    result = trainer.fit()

    # ── 드라이버: 방금 생성된 최신 MLflow Model Version에서 run_id 수신
    client = MlflowClient()
    versions = client.search_model_versions(f"name='{registered_model_name}'")
    if not versions:
        raise RuntimeError(f"[{target_col}] MLflow model version 을 찾지 못했습니다: {registered_model_name}")
    latest_version = max(versions, key=lambda v: int(v.version))
    run_id = latest_version.run_id

    val_rmse = (result.metrics or {}).get("validation-rmse")
    metrics  = {"rmse": round(float(val_rmse), 4)} if val_rmse is not None else {}

    log.info("[%s] distributed run_id=%s", target_col, run_id)
    return run_id, metrics

# ──────────────────────────────────────────────
# 5. Registry Staging 등록 (로컬/분산 공통)
#    log_model() 시점에 버전이 이미 생성되어 있음
#    → run_id로 해당 버전을 특정해 Staging으로 전환
# ──────────────────────────────────────────────
def register_to_staging(registered_model_name: str, run_id: str) -> str:
    """Returns: version string"""
    client   = MlflowClient()
    versions = client.search_model_versions(f"name='{registered_model_name}'")
    matched  = [v for v in versions if v.run_id == run_id]
    if not matched:
        raise RuntimeError(
            f"run_id={run_id} 에 해당하는 {registered_model_name} 버전을 찾을 수 없습니다."
        )
    version = matched[0].version

    # 기존 staging alias 제거
    try:
        client.delete_registered_model_alias(registered_model_name, "staging")
    except Exception:
        pass

    # 새 버전에 staging alias 부여
    client.set_registered_model_alias(registered_model_name, "staging", version)
    log.info("Registry: %s v%s → staging alias (run_id=%s)", registered_model_name, version, run_id)
    return version

# ──────────────────────────────────────────────
# 6. 메인
# ──────────────────────────────────────────────
def main(args: argparse.Namespace) -> None:
    mlflow.set_tracking_uri(args.mlflow_uri)

    p     = Path(args.input)
    files = sorted(p.glob("*.parquet")) if p.is_dir() else [p]
    df_raw = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    log.info("Loaded %d raw rows", len(df_raw))

    df = engineer_features(df_raw)
    log.info("Feature-engineered: %d rows", len(df))

    xgb_params = {**DEFAULT_XGB_PARAMS, "n_estimators": args.n_estimators}
    use_local  = args.local or not args.ray_address

    results = {}
    for target_col, experiment_name, registered_model_name in TARGETS:
        log.info("=" * 55)
        log.info("target=%s  mode=%s", target_col, "local" if use_local else "distributed")

        tags = {
            "pipeline_step": "train",
            "mode":          "local" if use_local else "distributed",
            "target":        target_col,
        }

        if use_local:
            run_id, metrics = train_and_log_local(
                df                    = df,
                target_col            = target_col,
                xgb_params            = xgb_params,
                experiment            = experiment_name,
                registered_model_name = registered_model_name,
                tags                  = tags,
            )
        else:
            run_id, metrics = train_and_log_distributed(
                df                    = df,
                target_col            = target_col,
                xgb_params            = xgb_params,
                experiment            = experiment_name,
                registered_model_name = registered_model_name,
                tags                  = tags,
                ray_address           = args.ray_address,
                num_workers           = args.num_workers,
                mlflow_uri            = args.mlflow_uri,
            )

        # 다음 evaluate step에서 Staging 등록을 하면 여기서는 스킵해야함
        # version = register_to_staging(registered_model_name, run_id)
        
        client   = MlflowClient()
        versions = client.search_model_versions(f"name='{registered_model_name}'")
        matched  = [v for v in versions if v.run_id == run_id]

        results[target_col] = {
            "metrics":          metrics,
            "run_id":           run_id,
            "registered_model": registered_model_name,
            "version":          matched[0].version if matched else None,
        }


    # ── 요약
    print("\n" + "=" * 60)
    print("  TRAINING SUMMARY")
    print("=" * 60)
    for target_col, info in results.items():
        unit  = "$" if "fare" in target_col else "min"
        label = "요금 예측 (Fare)" if "fare" in target_col else "ETA 예측"
        m     = info["metrics"]
        print(f"\n  [{label}]")
        if "rmse" in m: print(f"    RMSE    : {m['rmse']:.4f} {unit}")
        if "mae"  in m: print(f"    MAE     : {m['mae']:.4f} {unit}")
        if "r2"   in m: print(f"    R²      : {m['r2']:.4f}")
        print(f"    Run ID  : {info['run_id']}")
        print(f"    Registry: {info['registered_model']} v{info['version']}")
    print("\n" + "=" * 60)

    log.info("Step 3 complete. Models → MLflow Registry.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NYC Taxi – Step 3: Train")
    parser.add_argument("--input",        default="data/raw/",              help="Parquet 파일 또는 디렉토리")
    parser.add_argument("--local",        action="store_true",              help="로컬 실행 (Ray 없음)")
    parser.add_argument("--ray-address",  default=None,                     help="Ray 클러스터 주소")
    parser.add_argument("--num-workers",  type=int,  default=1,             help="Ray 워커 수")
    parser.add_argument("--n-estimators", type=int,  default=300,           help="XGBoost n_estimators")
    parser.add_argument("--mlflow-uri",   default="http://mlflow.cnapcloud.com", help="MLflow Tracking URI")
    args = parser.parse_args()
    main(args)