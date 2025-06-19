import re
import random
import string
import torch
import os
import warnings
from datetime import datetime
from typing import Optional, Dict
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from tools.intent_classifier import IntentClassifier
from configs.config_dialogs.config_dialog import BOT_CONFIG
from configs.program_config import MODEL_DIR

# Отключение несущественных предупреждений
warnings.filterwarnings("ignore", category=UserWarning)


class ChatNeural:
    def __init__(self, model_dir: str = MODEL_DIR):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_dir = model_dir
        self.last_responses = []
        self.comparison_state = {}
        self._init_components()
        self._init_intent_classifier()
        self.comparison_state: Dict[int, dict] = {}  # Для хранения состояния сравнения

        # Инициализация компонентов с обработкой ошибок
        try:
            self._init_components()
            self._init_intent_classifier()
        except Exception as e:
            raise RuntimeError(f"Ошибка инициализации ChatNeural: {str(e)}")

    def _init_components(self):
        """Инициализация GPT-2 модели с обработкой ошибок"""
        try:
            if not os.path.exists(self.model_dir):
                raise FileNotFoundError(f"Директория с моделью не найдена: {self.model_dir}")

            self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_dir)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model = GPT2LMHeadModel.from_pretrained(self.model_dir).to(self.device)
            self.model.eval()
        except Exception as e:
            raise RuntimeError(f"Ошибка инициализации модели: {str(e)}")

    def _init_intent_classifier(self):
        """Инициализация классификатора намерений"""
        try:
            self.intent_classifier = IntentClassifier()
            try:
                self.intent_classifier.load_model()
            except FileNotFoundError:
                print("Модель классификатора не найдена, начинаем обучение...")
                self.intent_classifier.train()
        except Exception as e:
            print(f"Ошибка инициализации классификатора: {str(e)}")
            self.intent_classifier = None

    def _get_predefined_response(self, text: str) -> Optional[str]:
        """Поиск в базе ответов с улучшенным сопоставлением"""
        text_clean = self._normalize_text(text)

        for intent_name, intent_data in BOT_CONFIG['intents'].items():
            for example in intent_data['examples']:
                example_clean = self._normalize_text(example)

                # Точное совпадение или частичное вхождение
                if (text_clean == example_clean or
                        example_clean in text_clean or
                        text_clean in example_clean):
                    response = random.choice(intent_data['responses'])
                    if self._is_valid_response(response):
                        return response
        return None

    def _normalize_text(self, text: str) -> str:
        """Нормализация текста для сравнения"""
        return text.lower().translate(
            str.maketrans('', '', string.punctuation)
        ).strip()

    def _generate_response(self, text: str) -> Optional[str]:
        """Генерация ответа с улучшенными параметрами"""
        prompt = (
            f"Ты эксперт по мониторам, который помогает подобрать идеальный вариант под задачи клиента. "
            f"Дата: {datetime.now().strftime('%d.%m.%Y')}\n"
            f"Твоя задача — рекомендовать мониторы с диагональю 23.8\", IPS-матрицей, разрешением 1920x1080 и частотой 100Гц, "
            f"но адаптировать выбор под запрос пользователя (игры, офис, дизайн и т.д.).\n"
            f"**Правила:**\n"
            f"- Отвечай кратко (1-3 предложения), но убедительно.\n"
            f"- Подчеркивай выгоды для конкретного использования (например, для игр — плавность, для дизайна — цветопередачу).\n"
            f"- Рекомендуй модели: AOC 24G2U, ASUS TUF VG249Q, LG 24GN600-B, Dell S2421HGF, MSI Optix G241, BenQ MOBIUZ EX2410.\n"
            f"- Не упоминай фото/видеотехнику.\n"
            # f"- Если вопрос не о мониторах, вежливо скажи: 'Я специализируюсь только на мониторах.'\n"
            # f"- Если вопрос не о мониторах, не отвечай на этот вопрос и вежливо скажи: 'Я специализируюсь только на мониторах.'\n"
            f"**Примеры ответов:**\n"
            f"- Для игр: 'ASUS TUF VG249Q с частотой 100Гц обеспечит плавный геймплей без разрывов.'\n"
            f"- Для работы: 'Dell S2421HGF с IPS-матрицей снижает усталость глаз при долгой работе.'\n"
            f"- Для дизайна: 'BenQ MOBIUZ EX2410 с точной цветопередачей подойдет для графики.'\n"
            f"Вопрос клиента: {text}\n"
        )

        generation_config = {
            "max_new_tokens": 100,
            "temperature": 0.7,
            "top_k": 50,
            "top_p": 0.90,
            "repetition_penalty": 3.0,  # Исправлено - должно быть > 1.0
            "no_repeat_ngram_size": 3,
            "do_sample": True,
            "num_beams": 3,
            # "early_stopping": True
        }

        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(self.device)
            with torch.no_grad():
                outputs = self.model.generate(**inputs, **generation_config)
            response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            return self._postprocess_response(response)
        except Exception as e:
            print(f"Ошибка генерации: {str(e)}")
            return None

    def _postprocess_response(self, text: str) -> str:
        """Очистка ответа от лишнего текста и повторов"""
        # Удаление служебных фраз
        stop_phrases = ["Бот:", "Консультант:", "Ответ:", "Вы можете купить"]
        for phrase in stop_phrases:
            if phrase in text:
                text = text.split(phrase)[-1].strip()

        # Удаление повторяющихся предложений
        sentences = re.split(r'(?<=[.!?])\s+', text)
        unique_sentences = []
        for sent in sentences:
            if sent and sent not in unique_sentences:
                unique_sentences.append(sent)

        return ' '.join(unique_sentences[:2]).strip()

    def _is_valid_response(self, text: str) -> bool:
        """Проверка качества ответа"""
        if not text or len(text.split()) < 2:
            return False

        invalid_patterns = [
            r"(.+?)\1{2,}",
            r"\b(\w+)\s+\1\b",
            r"[^\w\s,.!?-]",
            r"\b\w{15,}\b"
        ]

        return not any(re.search(p, text) for p in invalid_patterns)

    def get_response(self, text: str) -> str:
        """Улучшенная система ответов"""
        try:
            # 1. Проверка на пустой текст
            if not text.strip():
                return "Пожалуйста, задайте вопрос о нашей продукции."

            # 2. Попытка получить предопределенный ответ
            predefined = self._get_predefined_response(text)
            if predefined:
                return predefined

            # 3. Анализ намерения (если классификатор доступен)
            intent, confidence = None, 0.0
            if self.intent_classifier:
                try:
                    intent, confidence = self.intent_classifier.predict(text)
                    print(confidence, intent)
                    if confidence > 0.8 and intent in BOT_CONFIG['intents']:
                        return random.choice(BOT_CONFIG['intents'][intent]['responses'])
                except Exception as e:
                    print(f"Ошибка классификатора: {str(e)}")

            # 4. Генерация ответа моделью (до 3 попыток)
            for _ in range(3):
                response = self._generate_response(text)
                if response and response not in self.last_responses:
                    self._update_response_history(response)
                    return response

            # 5. Фолбек-ответ
            return self._get_fallback_response(intent)

        except Exception as e:
            print(f"Критическая ошибка в get_response: {str(e)}")
            return "Извините, произошла ошибка. Пожалуйста, попробуйте еще раз."

    # Можем использовать intent для персонализации

    def _update_response_history(self, response: str):
        """Обновление истории ответов (последние 2)"""
        self.last_responses = [response] + self.last_responses[:1]

    def _get_fallback_response(self, intent: str = None) -> str:
        """Улучшенные фолбек-ответы с учетом намерения"""
        if intent == "price":
            return "Уточните, о какой модели монитора вас интересует цена?"
        elif intent == "comparison":
            return "Какие именно модели вы хотите сравнить?"

        fallbacks = [
            "Уточните, пожалуйста, ваш вопрос.",
            "Какие характеристики вас интересуют?",
            "Для уточнения информации можете посетить наш сайт:\nhttp://crimea/IsOurs"
        ]
        return random.choice(fallbacks)