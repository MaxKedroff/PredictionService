# ml_service.py
import os
import pickle
import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime
import logging
from typing import Dict, Any, List
import psycopg2
from  psycopg2.extras import RealDictCursor
from  psycopg2.pool import SimpleConnectionPool
import  json

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Разрешаем CORS для всех доменов

# Глобальные переменные для моделей
model = None
scaler = None
label_encoders = {}
feature_order = []
feature_importance = {}
model_config = {}
numeric_features = []
categorical_features = []

db_pool = None
db_config = {
    'host': os.environ.get('DB_HOST', 'localhost'),
    'port': os.environ.get('DB_PORT', '5432'),
    'database': 'RealEstateDb',
    'user': 'postgres',
    'password': 'vfrcbv2005',
    'min_conn': 1,
    'max_conn': 10
}

# test
def load_models():
    """Загрузка всех моделей и препроцессоров"""
    global model, scaler, label_encoders, feature_order, feature_importance, model_config
    global numeric_features, categorical_features

    try:
        # Путь к папке с моделями
        models_path = os.path.join(os.path.dirname(__file__), 'ml_models')

        if not os.path.exists(models_path):
            # Пробуем альтернативные пути
            alternatives = [
                '/content/drive/MyDrive/ml_models/',
                './ml_models/',
                '../ml_models/',
                '../../ml_models/'
            ]
            for alt in alternatives:
                if os.path.exists(alt):
                    models_path = alt
                    break
            else:
                raise Exception(f"Папка с моделями не найдена. Искали в: {models_path}")

        logger.info(f"Загрузка моделей из: {models_path}")

        # Загрузка основной модели
        model_path = os.path.join(models_path, 'price_prediction_model.pkl')
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            logger.info("✅ Основная модель загружена")
        else:
            raise Exception(f"Модель не найдена: {model_path}")

        # Загрузка скейлера
        scaler_path = os.path.join(models_path, 'scaler.pkl')
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            logger.info("✅ Scaler загружен")

        # Загрузка LabelEncoders
        label_encoders_path = os.path.join(models_path, 'label_mappings.json')
        if os.path.exists(label_encoders_path):
            import json
            with open(label_encoders_path, 'r', encoding='utf-8') as f:
                label_mappings = json.load(f)
            # Конвертируем маппинги в формат для использования
            label_encoders = label_mappings
            logger.info("✅ Label encoders загружены")

        # Загрузка порядка признаков
        feature_order_path = os.path.join(models_path, 'feature_order.json')
        if os.path.exists(feature_order_path):
            import json
            with open(feature_order_path, 'r', encoding='utf-8') as f:
                feature_order = json.load(f)
            logger.info(f"✅ Порядок признаков загружен ({len(feature_order)} признаков)")

        # Загрузка важности признаков
        importance_path = os.path.join(models_path, 'feature_importance.csv')
        if os.path.exists(importance_path):
            feature_importance_df = pd.read_csv(importance_path)
            feature_importance = dict(zip(feature_importance_df['feature'], feature_importance_df['importance']))
            logger.info("✅ Важность признаков загружена")

        # Загрузка конфигурации
        config_path = os.path.join(models_path, 'model_config.json')
        if os.path.exists(config_path):
            import json
            with open(config_path, 'r', encoding='utf-8') as f:
                model_config = json.load(f)
            logger.info("✅ Конфигурация модели загружена")

            # Извлекаем списки признаков из конфига
            numeric_features = model_config.get('numeric_features', [])
            categorical_features = model_config.get('categorical_features', [])

        logger.info("✅ Все модели успешно загружены!")
        return True

    except Exception as e:
        logger.error(f"❌ Ошибка загрузки моделей: {e}")
        return False

def init_db_pool():
    global db_pool
    try:
        db_pool = SimpleConnectionPool(
                     minconn = db_config['min_conn'],
                     maxconn = db_config['max_conn'],
                     host = db_config['host'],
                     port = db_config['port'],
                     database = db_config['database'],
                     user = db_config['user'],
                     password = db_config['password'],
                     cursor_factory = RealDictCursor)

        logger.info("✅ Пул соединений с PostgreSQL создан")
        return True
    except Exception as e:
        logger.error(f"❌ Ошибка подключения к БД: {e}")
        return False

def get_db_connection():
    if db_pool:
        return  db_pool.getconn()
    return None

def return_db_connection(conn):
    if db_pool and conn:
        db_pool.putconn(conn)


def calculate_similarity_with_prediction(target_flat: Dict, db_flat: Dict, model, preprocess_func,
                                         feature_importance: Dict = None) -> float:
    """
    Расчет сходства с использованием предсказанной цены модели (как в обучении)
    """
    try:
        # Предсказываем цену для квартиры из БД
        db_prediction = predict_for_flat(db_flat, model, preprocess_func)

        # 1. Сходство по предсказанной цене (вес 0.4)
        target_price = target_flat.get('target_price', 0)
        if target_price > 0 and db_prediction > 0:
            price_sim = 1 - min(1, abs(db_prediction - target_price) / max(target_price, 1))
        else:
            price_sim = 0.5

        # 2. Сходство по числовым признакам (вес 0.4)
        numeric_fields = ['flat_area', 'flat_rooms', 'flat_floor', 'flat_area_living', 'flat_area_kitchen']
        num_score = 0
        num_count = 0

        for field in numeric_fields:
            target_val = target_flat.get(field, 0)
            db_val = db_flat.get(field, 0)

            try:
                target_val = float(target_val) if target_val else 0
                db_val = float(db_val) if db_val else 0

                if target_val > 0 and db_val > 0:
                    diff = abs(db_val - target_val) / max(target_val, 1)
                    field_sim = 1 - min(1, diff)
                    num_score += field_sim
                    num_count += 1
                elif target_val == 0 and db_val == 0:
                    num_score += 1
                    num_count += 1
            except (ValueError, TypeError):
                pass

        if num_count > 0:
            num_score = num_score / num_count
        else:
            num_score = 0.5

        # 3. Сходство по категориальным признакам (вес 0.2)
        categorical_fields = ['flat_balcony', 'flat_loggia', 'flat_furniture', 'flat_status', 'city_id']
        cat_score = 0
        cat_count = 0

        for field in categorical_fields:
            target_val = str(target_flat.get(field, 'unknown')).strip()
            db_val = str(db_flat.get(field, 'unknown')).strip()

            if target_val == db_val:
                cat_score += 1
            cat_count += 1

        if cat_count > 0:
            cat_score = cat_score / cat_count
        else:
            cat_score = 0.5

        # Итоговый score
        total_score = 0.4 * price_sim + 0.4 * num_score + 0.2 * cat_score

        return total_score

    except Exception as e:
        logger.error(f"Ошибка расчета сходства: {e}")
        return 0.0

def calculate_similarity_for_service(target_flat: Dict, db_flat: Dict, model_prediction: float,
                                     feature_importance: Dict = None) -> float:
    """
    Расчет сходства между целевой квартирой и квартирой из БД
    Полностью повторяет логику из обучения моделей
    """

    # 1. Сходство по предсказанной цене (вес 0.4)
    # Для квартир из БД нужно было бы предсказать цену, но у нас ее нет
    # Поэтому используем реальную цену из БД как прокси для предсказанной
    db_price = float(db_flat.get('flat_price', 0))
    if db_price > 0 and model_prediction > 0:
        price_sim = 1 - min(1, abs(db_price - model_prediction) / max(model_prediction, 1))
    else:
        price_sim = 0.5

    # 2. Сходство по числовым признакам (вес 0.4)
    numeric_fields = ['flat_area', 'flat_rooms', 'flat_floor', 'flat_area_living', 'flat_area_kitchen']
    num_score = 0
    num_count = 0

    for field in numeric_fields:
        target_val = target_flat.get(field, 0)
        db_val = db_flat.get(field, 0)

        try:
            target_val = float(target_val) if target_val else 0
            db_val = float(db_val) if db_val else 0

            if target_val > 0 and db_val > 0:
                diff = abs(db_val - target_val) / max(target_val, 1)
                field_sim = 1 - min(1, diff)
                num_score += field_sim
                num_count += 1
            elif target_val == 0 and db_val == 0:
                num_score += 1
                num_count += 1
        except (ValueError, TypeError):
            pass

    if num_count > 0:
        num_score = num_score / num_count
    else:
        num_score = 0.5

    # 3. Сходство по категориальным признакам (вес 0.2)
    categorical_fields = ['flat_balcony', 'flat_loggia', 'flat_furniture', 'flat_status', 'city_id']
    cat_score = 0
    cat_count = 0

    for field in categorical_fields:
        target_val = str(target_flat.get(field, 'unknown')).strip()
        db_val = str(db_flat.get(field, 'unknown')).strip()

        if target_val == db_val:
            cat_score += 1
        cat_count += 1

    if cat_count > 0:
        cat_score = cat_score / cat_count
    else:
        cat_score = 0.5

    # Итоговый score (те же веса, что в обучении)
    # price_sim - 0.4, num_score - 0.4, cat_score - 0.2
    total_score = 0.4 * price_sim + 0.4 * num_score + 0.2 * cat_score

    return total_score

def preprocess_input(data: Dict[str, Any]) -> pd.DataFrame:
    """Предобработка входных данных"""
    global scaler, label_encoders, feature_order, numeric_features, categorical_features

    # Удаляем TYPES_RENOVATION если есть
    if 'TYPES_RENOVATION' in data:
        logger.info(f"Игнорируем поле TYPES_RENOVATION со значением: {data.pop('TYPES_RENOVATION')}")

    # 🔥 Преобразуем все булевы значения в числа
    data = data.copy()  # Создаем копию, чтобы не изменять оригинал
    for key, value in data.items():
        if isinstance(value, bool):
            data[key] = int(value)  # True -> 1, False -> 0
            logger.info(f"Преобразовано булево поле {key}: {value} -> {data[key]}")
        elif isinstance(value, str) and value.lower() in ['true', 'false']:
            data[key] = 1 if value.lower() == 'true' else 0
            logger.info(f"Преобразовано строковое булево поле {key}: {value} -> {data[key]}")

    # Создаем DataFrame из входных данных
    input_df = pd.DataFrame([data])

    # Обработка числовых признаков
    for col in numeric_features:
        if col in input_df.columns:
            # Конвертируем в число
            input_df[col] = pd.to_numeric(input_df[col], errors='coerce')
            # Заполняем пропуски 0
            input_df[col] = input_df[col].fillna(0)
        else:
            # Если колонки нет в данных, добавляем с 0
            input_df[col] = 0

    # Обработка категориальных признаков
    for col in categorical_features:
        if col in input_df.columns:
            # Приводим к строке
            input_df[col] = input_df[col].fillna('unknown').astype(str)

            # Применяем label encoding из сохраненного маппинга
            if col in label_encoders and 'transform' in label_encoders[col]:
                transform_map = label_encoders[col]['transform']
                input_df[col] = input_df[col].map(lambda x: transform_map.get(x, 0))
            else:
                logger.warning(f"Признак {col} не найден в label_encoders, используем 0")
                input_df[col] = 0
        else:
            # Если категориальной колонки нет, добавляем с 0
            input_df[col] = 0

    # Масштабируем числовые признаки
    if scaler and numeric_features:
        # Убеждаемся, что все числовые колонки присутствуют
        for col in numeric_features:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df[numeric_features] = scaler.transform(input_df[numeric_features])

    # Сортируем колонки в правильном порядке
    # Добавляем отсутствующие колонки
    for col in feature_order:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[feature_order]

    return input_df

@app.route('/health', methods=['GET'])
def health_check():
    """Проверка статуса сервиса"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'timestamp': datetime.now().isoformat()
    })


@app.route('/predict', methods=['POST'])
def predict():
    """Предсказание цены"""
    try:
        # Получаем данные из запроса
        data = request.json

        if not data:
            return jsonify({'error': 'No data provided'}), 400

        logger.info(f"Получен запрос на предсказание: {data.get('flat_id', 'unknown')}")

        # Предобработка данных
        input_data = preprocess_input(data)

        # Предсказание
        prediction = model.predict(input_data)[0]

        # Получаем реальную цену (если есть)
        actual_price = data.get('actual_price', None)

        # Формируем ответ
        response = {
            'predicted_price': float(prediction),
            'predicted_price_mln': float(prediction) / 1000000,
            'currency': 'RUB',
            'timestamp': datetime.now().isoformat(),
            'model_version': model_config.get('version', '1.0.0'),
            'model_name': model_config.get('model_name', 'Unknown')
        }

        # Если есть реальная цена - вычисляем отклонение
        if actual_price:
            deviation = prediction - actual_price
            deviation_percent = (deviation / actual_price) * 100

            response['actual_price'] = float(actual_price)
            response['deviation'] = float(deviation)
            response['deviation_percent'] = float(deviation_percent)

            # Статус цены
            if deviation_percent > 10:
                response['status'] = 'ЗАВЫШЕНА'
                response['recommendation'] = f'Цена выше рыночной. Рекомендуется снизить на {deviation_percent:.0f}%'
            elif deviation_percent < -10:
                response['status'] = 'ЗАНИЖЕНА'
                response['recommendation'] = f'Цена ниже рыночной на {-deviation_percent:.0f}%. Отличное предложение!'
            else:
                response['status'] = 'АДЕКВАТНАЯ'
                response['recommendation'] = 'Цена соответствует рынку. Хорошее предложение.'

        logger.info(f"Предсказание выполнено: {response['predicted_price']:.0f} руб")

        return jsonify(response)

    except Exception as e:
        logger.error(f"Ошибка при предсказании: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """Массовое предсказание цен"""
    try:
        data = request.json

        if not data or 'flats' not in data:
            return jsonify({'error': 'No flats data provided'}), 400

        flats = data['flats']
        results = []

        for flat in flats:
            try:
                input_data = preprocess_input(flat)
                prediction = model.predict(input_data)[0]

                result = {
                    'flat_id': flat.get('flat_id', 'unknown'),
                    'predicted_price': float(prediction),
                    'predicted_price_mln': float(prediction) / 1000000
                }

                # Если есть реальная цена
                if 'actual_price' in flat:
                    result['actual_price'] = float(flat['actual_price'])
                    result['deviation_percent'] = float(
                        (prediction - flat['actual_price']) / flat['actual_price'] * 100)

                results.append(result)
            except Exception as e:
                logger.error(f"Ошибка для квартиры {flat.get('flat_id', 'unknown')}: {e}")
                results.append({
                    'flat_id': flat.get('flat_id', 'unknown'),
                    'error': str(e)
                })

        return jsonify({
            'total': len(results),
            'results': results,
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"Ошибка при массовом предсказании: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/feature_importance', methods=['GET'])
def get_feature_importance():
    """Получение важности признаков"""
    return jsonify({
        'feature_importance': feature_importance,
        'top_features': dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10])
    })


@app.route('/model_info', methods=['GET'])
def get_model_info():
    """Информация о модели"""
    return jsonify({
        'model_config': model_config,
        'total_features': len(feature_order),
        'numeric_features': numeric_features,
        'categorical_features': categorical_features,
        'feature_order': feature_order
    })


if __name__ == '__main__':
    # Загружаем модели
    if load_models():
        # Запускаем сервер
        port = int(os.environ.get('PORT', 5002))
        app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
    else:
        logger.error("Не удалось загрузить модели. Сервер не запущен.")