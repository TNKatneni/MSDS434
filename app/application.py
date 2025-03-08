import os
import json
import boto3
from flask import Flask, request, jsonify
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from prometheus_flask_exporter import PrometheusMetrics

app = Flask(__name__)
application = app


# ✅ Define `/metrics` manually
@app.route('/metrics', methods=['GET'])
def metrics_endpoint():
    return generate_latest(), 200, {'Content-Type': CONTENT_TYPE_LATEST}


# ✅ Initialize Prometheus AFTER defining `/metrics`
metrics = PrometheusMetrics(app)

SM_ENDPOINT_NAME = os.environ.get("SM_ENDPOINT_NAME", "my-housing-endpoint")
AWS_REGION = "us-east-1"


@app.route('/', methods=['GET'])
def health_check():
    return jsonify({"message": "Housing Prediction API is up!"})


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON body provided"}), 400

    try:
        bedrooms = data["bedrooms"]
        bathrooms = data["bathrooms"]
        lot_size = data["lot_size"]
        house_size = data["house_size"]
    except KeyError as e:
        return jsonify({"error": f"Missing field {str(e)}"}), 400

    features = [bedrooms, bathrooms, lot_size, house_size]
    csv_payload = ",".join(str(x) for x in features)

    sagemaker_runtime = boto3.client(
        "sagemaker-runtime", region_name=AWS_REGION)
    try:
        response = sagemaker_runtime.invoke_endpoint(
            EndpointName=SM_ENDPOINT_NAME,
            Body=csv_payload,
            ContentType="text/csv"
        )  # noqa: E501
        result = json.loads(
            response['Body'].read().decode("utf-8")
        )  # noqa: E501

        if isinstance(result, dict):
            prediction = result.get("predictions",
            [None])[0] or list(result.values())[0]
        elif isinstance(result, list):
            prediction = result[0]
        else:
            prediction = result

        return jsonify({"prediction": prediction})

    except Exception as e:
        return jsonify({
            "error": f"SageMaker invocation failed: {str(e)}"
        }), 500  # noqa: E501


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port, debug=True)
