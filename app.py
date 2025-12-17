import pandas as pd
import numpy as np
import io
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from catboost import CatBoostClassifier

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Загрузка модели
model = CatBoostClassifier()
try:
    model.load_model("model.cbm")
    print("Модель успешно загружена.")
except Exception as e:
    print(f"Ошибка загрузки модели: {e}")

# ----------------- НАСТРОЙКИ ПРИЗНАКОВ -----------------

# 1. Числовые признаки (будут принудительно приведены к типу float)
NUMERIC_FEATURES = ['BMI',
    'Sedentary Hours Per Day',
    'Systolic blood pressure',
    'Heart rate',
    'Triglycerides',
    'Exercise Hours Per Week',
    'Cholesterol',
    'Age',
    'Diastolic blood pressure',
    'Sleep Hours Per Day',
    'Blood sugar'
]

# 2. Категориальные признаки (будут приведены к типу string)
CAT_FEATURES = ['Physical Activity Days Per Week', 'Stress Level', 'Diet', 'Obesity']

# Общий список
REQUIRED_FEATURES = NUMERIC_FEATURES + CAT_FEATURES

# -------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def main(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    if file.filename.endswith('.xlsx'):
        df = pd.read_excel(io.BytesIO(contents))
    elif file.filename.endswith('.csv'):
        df = pd.read_csv(io.BytesIO(contents))
    else:
        return {"error": "Поддерживаются только файлы .xlsx и .csv"}

    if 'id' not in df.columns:
        return {"error": "В загруженном файле отсутствует столбец 'id'"}

    # Проверка наличия колонок
    try:
        df_model = df[REQUIRED_FEATURES].copy()
    except KeyError as e:
        return {"error": f"В файле не хватает колонок: {e}"}

    # ---------------- ПАЙПЛАЙН ОБРАБОТКИ ----------------
    
    # 1. Обработка ЧИСЛОВЫХ (Строго в float)
    for col in NUMERIC_FEATURES:
        if col in df_model.columns:
            series = pd.to_numeric(df_model[col], errors='coerce')
            df_model[col] = series.astype(float)

    # 2. Обработка КАТЕГОРИАЛЬНЫХ (пустые строки временно оставляем как NaN для детекции)
    for col in CAT_FEATURES:
        if col in df_model.columns:
            # Сначала заменяем пустые строки на NaN для корректной детекции пропусков
            df_model[col] = df_model[col].replace('', np.nan)
            # Убираем .0 в конце строки
            df_model[col] = df_model[col].astype(str).replace(r'\.0$', '', regex=True)
            # Заменяем 'nan' обратно на NaN
            df_model[col] = df_model[col].replace('nan', np.nan)

    # 3. Детекция пропусков (ищем NaN во всей таблице)
    rows_with_gaps = df_model.isnull().any(axis=1)

    # 4. Подготовка "чистых" данных (без пропусков)
    df_clean = df_model[~rows_with_gaps].copy()
    
    # 5. Подготовка категориальных признаков для модели (заполняем пустыми строками)
    for col in CAT_FEATURES:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna("").astype(str)

    # 6. Предсказание
    # Инициализируем все предсказания как 1 (для строк с пропусками)
    final_predictions = pd.Series(1, index=df_model.index)

    if len(df_clean) > 0:
        try:
            # Получаем порядок столбцов, который ждет модель
            expected_order = model.feature_names_
            df_clean_sorted = df_clean[expected_order]
            
            # Предсказываем только для чистых строк
            clean_preds = model.predict(df_clean_sorted)
            # Заменяем предсказания для строк БЕЗ пропусков
            final_predictions[~rows_with_gaps] = clean_preds
            # Строки С пропусками остаются = 1
        except KeyError as e:
            return {"error": f"Ошибка структуры данных: {e}"}
        except Exception as e:
            return {"error": f"Ошибка модели CatBoost: {str(e)}"}
    
    # ---------------- СБОРКА РЕЗУЛЬТАТА ----------------

    df_result = pd.DataFrame()
    df_result['id'] = df['id']
    df_result['prediction'] = final_predictions

    # Преобразуем в целые числа
    df_result['prediction'] = df_result['prediction'].astype('Int64')

    stream = io.StringIO()
    df_result.to_csv(stream, index=False, sep=',', na_rep='') 
    
    response = StreamingResponse(iter([stream.getvalue()]), media_type="text/csv")
    response.headers["Content-Disposition"] = "attachment; filename=predictions.csv"
    return response