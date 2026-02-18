config: dict[str, str] = {
    "input_filename": "train_preprocessed.csv",
    "case_id_col": "case_id",
    "resource_id_col": "resource",
    "activity_col": "activity_name",
    "start_timestamp_col": "start_timestamp",
    "end_timestamp_col": "end_timestamp",
}

# Optimization settings
optimization_config = {
    "qmix_batch_size": 512,  # Increased from 32 for better GPU utilization
    "enable_model_eval_mode": True,  # Use model.eval() during inference to disable dropout
}
