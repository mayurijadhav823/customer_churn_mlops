
import json
import joblib
import pandas as pd

from pathlib import Path
from django.conf import settings
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST

MODEL_PATH = Path(settings.BASE_DIR) / "models" / "churn_model.pkl"
model = joblib.load(MODEL_PATH)


@csrf_exempt
@require_POST
def predict_churn(request):
    try:
        data = json.loads(request.body)
        customer = pd.DataFrame([data])

        prediction = int(model.predict(customer)[0])
        probabilities = model.predict_proba(customer)[0]

        return JsonResponse({
            "prediction": "CHURN" if prediction == 1 else "NOT CHURN",
            "churn_probability": round(float(probabilities[1]) * 100, 2),
            "no_churn_probability": round(float(probabilities[0]) * 100, 2)
        })

    except (ValueError, TypeError, KeyError) as error:
        return JsonResponse({"error": str(error)}, status=400)
