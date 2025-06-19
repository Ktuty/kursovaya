from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
import os
import logging
import torch
from getpass import getpass

# Конфигурация модели
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.1"  # Или "mistralai/Mistral-7B-Instruct-v0.1"
CACHE_DIR = "../model_cache"
HF_TOKEN = None  # Будет запрошен, если не указан

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_hf_token():
    """Запрашивает токен Hugging Face, если он не установлен"""
    token = "YOUR_TOKEN_HERE"
    if token is None:
        logger.warning("Токен Hugging Face не найден в переменных окружения")
        token = getpass("Введите ваш токен Hugging Face (скопируйте с https://huggingface.co/settings/tokens): ")
    return token


def download_model():
    """Загружает модель и токенизатор с аутентификацией"""
    try:
        global HF_TOKEN

        # Получаем токен
        HF_TOKEN = get_hf_token()
        if not HF_TOKEN:
            logger.error("Токен Hugging Face не предоставлен")
            return None, None

        # Аутентификация
        login(token=HF_TOKEN)
        logger.info("Аутентификация на Hugging Face Hub успешна")

        # Создаем папку, если ее нет
        os.makedirs(CACHE_DIR, exist_ok=True)

        # Проверяем, есть ли уже модель в кэше
        model_path = os.path.join(CACHE_DIR, f"models--{MODEL_NAME.replace('/', '--')}")
        if os.path.exists(model_path):
            logger.info("Обнаружен кэш модели. Проверяем целостность...")

        logger.info(f"Загрузка модели Mistral: {MODEL_NAME}")
        logger.info("Размер модели: 7B параметров (~14GB в FP16)")

        # Проверяем наличие GPU
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Используемое устройство: {device}")

        # Загрузка токенизатора
        logger.info("Скачивание токенизатора...")
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            cache_dir=CACHE_DIR,
            load_in_4bit=True,
            device_map="auto",
            use_fast=True,
            token=HF_TOKEN  # Передаем токен для доступа
        )

        # Загрузка модели
        logger.info("Скачивание модели... (это может занять время и потребовать много памяти)")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            cache_dir=CACHE_DIR,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            low_cpu_mem_usage=True,
            token=HF_TOKEN  # Передаем токен для доступа
        ).to(device)

        logger.info(f"Модель и токенизатор успешно сохранены в: {os.path.abspath(CACHE_DIR)}")
        logger.info(f"Используемая память: ~14GB (на GPU в FP16) / ~28GB (на CPU в FP32)")

        return model, tokenizer

    except Exception as e:
        logger.error(f"Ошибка при загрузке модели: {str(e)}")
        return None, None


if __name__ == "__main__":
    # Два способа предоставить токен:
    # 1. Через переменную окружения: export HF_TOKEN="ваш_токен"
    # 2. Через интерактивный ввод при запуске

    model, tokenizer = download_model()

    if model and tokenizer:
        # Быстрая проверка работы
        device = "cuda" if torch.cuda.is_available() else "cpu"
        test_input = "Привет, как дела?"

        logger.info("\nТест генерации:")
        logger.info(f"Ввод: {test_input}")

        inputs = tokenizer(test_input, return_tensors="pt").to(device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            temperature=0.7,
            do_sample=True
        )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"Вывод: {generated_text}")
