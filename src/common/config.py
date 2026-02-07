RAW_EXPECTED_COLUMNS = [
    "Product ID",
    "Product Title",
    "Merchant ID",
    "Cluster ID",
    "Cluster Label",
    "Category ID",
    "Category Label",
]

FEATURE_COLUMNS = ["Product Title", "Merchant ID"]
TARGET_COLUMN = "Category Label"

PROCESSED_COLUMNS = FEATURE_COLUMNS + [TARGET_COLUMN]

S3_RAW_KEY = "raw/pricerunner/pricerunner_aggregate.csv"
S3_PROCESSED_KEY = "processed/pricerunner/processed.csv"
S3_SCHEMA_KEY = "processed/pricerunner/schema.json"
S3_CLASSES_KEY = "processed/pricerunner/classes.json"
S3_MODEL_KEY = "models/pricerunner/pipeline.joblib"
S3_METRICS_KEY = "models/pricerunner/metrics.json"
S3_MODEL_INFO_KEY = "models/pricerunner/model_info.json"

S3_MODELS_PREFIX = "models/pricerunner"
S3_MODEL_VERSIONS_PREFIX = "models/pricerunner/versions"
S3_MODEL_MARKERS_PREFIX = "models/pricerunner/markers"
S3_DEFAULT_POINTER_KEY = "models/pricerunner/default.json"

S3_INFERENCE_INPUT_PREFIX = "inference/input"
S3_INFERENCE_OUTPUT_PREFIX = "inference/output"
