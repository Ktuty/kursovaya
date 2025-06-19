import os
import logging
import asyncio
import tempfile
import random
from pathlib import Path
from typing import Optional, List, Dict
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from aiogram.types import Message, FSInputFile
from aiogram.enums import ContentType
from aiogram.utils.keyboard import InlineKeyboardBuilder
import speech_recognition as sr
from pydub import AudioSegment
from gtts import gTTS
from configs.program_config import MODEL_DIR, TELEGRAM_TOKEN
from tools.neyro import ChatNeural  # Импортируем вашу модель

# Конфигурация
TELEGRAM_TOKEN = TELEGRAM_TOKEN
MODEL_DIR = str(Path(__file__).parent / MODEL_DIR)
IMAGES_DIR = Path(__file__).parent / "images"

# Настройка логов
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("../logs/bot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ProductDatabase:
    """Класс для работы с базой данных товаров"""
    def __init__(self):
        self.products = self._load_products()

    def _load_products(self) -> List[Dict]:
        """Загружает список товаров"""
        return [
            {
                "id": 1,
                "name": "Dell S2721DGF",
                "price": 45990,
                "discount_price": 42990,
                "description": "27-дюймовый игровой монитор с QHD (2560x1440), 165Hz, IPS, AMD FreeSync Premium Pro",
                "features": ["165Hz", "1ms отклик", "HDR400", "Регулировка высоты"],
                "best_for": "геймеров и киберспортсменов",
                "image": "dell_s2721dgf.jpg",
                "promo_code": "DELL1000",
                "stores": ["DNS", "Ситилинк", "М.Видео"],
                "tags": ["игры", "высокая частота", "быстрый отклик"]
            },
            # Остальные товары...
        ]

    def get_product_by_id(self, product_id: int) -> Optional[Dict]:
        """Возвращает товар по ID"""
        return next((p for p in self.products if p['id'] == product_id), None)

    def get_random_products(self, count: int = 2) -> List[Dict]:
        """Возвращает случайные товары"""
        return random.sample(self.products, min(count, len(self.products)))

    def get_products_by_category(self, category: Optional[str]) -> List[Dict]:
        """Возвращает товары указанной категории"""
        if not category:
            return self.products

        category_map = {
            "игры": ["игры", "геймер", "киберспорт"],
            "дизайн": ["дизайн", "фото", "цвет", "калибр"],
            "офис": ["офис", "текст", "документ"]
        }

        keywords = category_map.get(category, [])
        return [
            p for p in self.products
            if any(kw in p['best_for'].lower() or kw in " ".join(p['tags']).lower() for kw in keywords)
        ]

class VoiceBot:
    def __init__(self):
        self.bot = Bot(token=TELEGRAM_TOKEN)
        self.dp = Dispatcher()
        self.neuro = ChatNeural(model_dir=MODEL_DIR)
        self.recognizer = sr.Recognizer()
        self._setup_handlers()
        self._check_dependencies()
        self.user_sessions = {}  # Для хранения состояния пользователей

    def _check_dependencies(self):
        """Проверка необходимых компонентов"""
        try:
            with tempfile.NamedTemporaryFile(suffix='.ogg', delete=False) as tmp:
                test_file = tmp.name
            silent_audio = AudioSegment.silent(duration=100)
            silent_audio.export(test_file, format="ogg")
            AudioSegment.from_ogg(test_file)
            logger.info("FFmpeg работает корректно")
        except Exception as e:
            logger.critical("FFmpeg не настроен правильно! Убедитесь что ffmpeg установлен и добавлен в PATH")
            raise RuntimeError("FFmpeg не настроен") from e
        finally:
            try:
                if os.path.exists(test_file):
                    os.remove(test_file)
            except Exception as e:
                logger.warning(f"Error deleting test file: {e}")

    def _setup_handlers(self):
        """Регистрация обработчиков сообщений"""
        self.dp.message(Command("start"))(self._handle_start)
        self.dp.message(Command("help"))(self._handle_help)
        self.dp.message(Command("products"))(self._handle_products)
        self.dp.message(Command("compare"))(self._handle_compare)
        self.dp.message(Command("promo"))(self._handle_promo)
        self.dp.callback_query()(self._handle_callback)
        self.dp.message(lambda m: m.content_type == ContentType.VOICE)(self._handle_voice)
        self.dp.message()(self._handle_text)

    async def _handle_start(self, message: Message):
        """Обработка команды /start"""
        try:
            await message.answer(
                "🖥️ Я бот-консультант по мониторам\n\n"
                "Задайте мне вопрос, например:\n"
                "• Игровой монитор до 50 тысяч\n"
                "• Лучший 4K для дизайнера\n"
                "• Сравните Dell и ASUS 27 дюймов\n\n"
                "Доступные команды:\n"
                "/products - все модели\n"
                "/compare - сравнить мониторы\n"
                "/promo - акции и скидки\n\n"
                "Вы можете писать текстом или отправлять голосовые сообщения!"
            )
        except Exception as e:
            logger.error(f"Start command error: {e}", exc_info=True)

    async def _handle_help(self, message: Message):
        """Обработка команды /help"""
        await message.answer(
            "ℹ️ Как получить лучший ответ:\n"
            "1. Укажите бюджет (например, 'до 40 тысяч')\n"
            "2. Назначение (игры, дизайн, офис)\n"
            "3. Желаемую диагональ\n\n"
            "Пример: 'Нужен монитор 27 дюймов для работы с фото, бюджет 60 тысяч'"
        )

    async def _handle_products(self, message: Message):
        """Показывает все товары с кнопками действий"""
        try:
            builder = InlineKeyboardBuilder()

            for product in self.neuro.db.products:
                builder.button(
                    text=f"{product['name']} - {product['discount_price']:,}₽",
                    callback_data=f"product_{product['id']}"
                )

            builder.adjust(1)

            await message.answer(
                "🎮 <b>Наши топовые модели мониторов:</b>\n"
                "Выберите модель для подробной информации:",
                reply_markup=builder.as_markup(),
                parse_mode="HTML"
            )

        except Exception as e:
            logger.error(f"Products command error: {e}", exc_info=True)
            await message.answer("❌ Не удалось загрузить информацию о продуктах")

    async def _handle_compare(self, message: Message):
        """Запускает процесс сравнения товаров"""
        builder = InlineKeyboardBuilder()

        for product in self.neuro.db.products:
            builder.button(
                text=product['name'],
                callback_data=f"compare_{product['id']}"
            )

        builder.adjust(2)

        await message.answer(
            "🔍 <b>Сравнение мониторов</b>\n"
            "Выберите 2 модели для сравнения:",
            reply_markup=builder.as_markup(),
            parse_mode="HTML"
        )

    async def _handle_promo(self, message: Message):
        """Показывает активные промокоды"""
        promo_text = "🎁 <b>Активные промокоды:</b>\n\n"

        for product in self.neuro.db.products:
            promo_text += (
                f"<b>{product['name']}</b>\n"
                f"Промокод: <code>{product['promo_code']}</code>\n"
                f"Скидка: {product['price'] - product['discount_price']:,}₽\n"
                f"Новая цена: {product['discount_price']:,}₽\n\n"
            )

        await message.answer(promo_text, parse_mode="HTML")

    async def _handle_callback(self, callback: types.CallbackQuery):
        """Обрабатывает все callback-запросы"""
        try:
            data = callback.data

            if data.startswith("product_"):
                product_id = int(data.split("_")[1])
                await self._show_product_details(callback, product_id)

            elif data.startswith("compare_"):
                product_id = int(data.split("_")[1])
                await self._process_comparison(callback, product_id)

            elif data == "buy":
                await callback.message.answer("🛒 Для покупки перейдите в один из магазинов:\n"
                                              "• https://www.dns-shop.ru\n"
                                              "• https://www.citilink.ru\n"
                                              "• https://www.mvideo.ru")

            elif data.startswith("stores_"):
                product_id = int(data.split("_")[1])
                product = self.neuro.db.get_product_by_id(product_id)

                stores = "\n".join([f"• {store}" for store in product['stores']])
                await callback.message.answer(
                    f"🏪 <b>Где купить {product['name']}:</b>\n\n"
                    f"{stores}\n\n"
                    f"Промокод: <code>{product['promo_code']}</code>",
                    parse_mode="HTML"
                )

            await callback.answer()

        except Exception as e:
            logger.error(f"Callback error: {e}", exc_info=True)
            await callback.answer("Произошла ошибка", show_alert=True)

    async def _show_product_details(self, callback: types.CallbackQuery, product_id: int):
        """Показывает детали продукта с кнопками действий"""
        product = self.neuro.db.get_product_by_id(product_id)
        if not product:
            await callback.answer("Товар не найден", show_alert=True)
            return

        features = "\n".join([f"• {f}" for f in product["features"]])
        stores = ", ".join(product["stores"])

        text = (
            f"<b>{product['name']}</b>\n"
            f"ID: {product['id']}\n\n"
            f"<b>Цена:</b>\n"
            f"<s>{product['price']:,}₽</s> → <b>{product['discount_price']:,}₽</b>\n"
            f"Промокод: <code>{product['promo_code']}</code>\n\n"
            f"<b>Описание:</b>\n"
            f"{product['description']}\n\n"
            f"<b>Характеристики:</b>\n"
            f"{features}\n\n"
            f"<b>Где купить:</b> {stores}"
        )

        builder = InlineKeyboardBuilder()
        builder.button(text="🛒 Купить", callback_data="buy")
        builder.button(text="🏪 Магазины", callback_data=f"stores_{product_id}")
        builder.button(text="🔍 Сравнить", callback_data=f"compare_{product_id}")
        builder.adjust(2)

        try:
            if product.get('image'):
                image_path = IMAGES_DIR / product['image']
                if image_path.exists():
                    photo = FSInputFile(image_path)
                    await callback.message.answer_photo(
                        photo=photo,
                        caption=text,
                        reply_markup=builder.as_markup(),
                        parse_mode="HTML"
                    )
                    return

        except Exception as e:
            logger.error(f"Error sending photo: {e}")

        await callback.message.answer(
            text,
            reply_markup=builder.as_markup(),
            parse_mode="HTML"
        )

    async def _process_comparison(self, callback: types.CallbackQuery, product_id: int):
        """Обрабатывает процесс сравнения товаров"""
        try:
            user_id = callback.from_user.id

            # Инициализация состояния сравнения для пользователя
            if user_id not in self.neuro.comparison_state:
                self.neuro.comparison_state[user_id] = {
                    'selected_ids': [],
                    'message_id': callback.message.message_id
                }

            # Добавляем выбранный товар
            if product_id not in self.neuro.comparison_state[user_id]['selected_ids']:
                self.neuro.comparison_state[user_id]['selected_ids'].append(product_id)

            # Если выбрано 2 товара - выполняем сравнение
            if len(self.neuro.comparison_state[user_id]['selected_ids']) >= 2:
                selected_ids = self.neuro.comparison_state[user_id]['selected_ids'][:2]
                comparison_text = self.neuro.compare_products(selected_ids)
                chart_path = self.neuro.generate_comparison_chart(selected_ids)

                if chart_path:
                    photo = FSInputFile(chart_path)
                    await callback.message.answer_photo(
                        photo=photo,
                        caption=comparison_text,
                        parse_mode="HTML"
                    )
                    os.remove(chart_path)
                else:
                    await callback.message.answer(comparison_text, parse_mode="HTML")

                # Очищаем состояние
                del self.neuro.comparison_state[user_id]
            else:
                product = self.neuro.db.get_product_by_id(product_id)
                await callback.answer(f"Выбран {product['name']}. Теперь выберите второй монитор для сравнения.")

        except Exception as e:
            logger.error(f"Comparison error: {e}", exc_info=True)
            await callback.message.answer("❌ Не удалось выполнить сравнение")
            if user_id in self.neuro.comparison_state:
                del self.neuro.comparison_state[user_id]

    async def _handle_voice(self, message: types.Message):
        """Обработка голосовых сообщений"""
        logger.info(f"Начало обработки голосового сообщения от пользователя {message.from_user.id}")
        ogg_path, wav_path = None, None

        try:
            if message.voice.duration > 30:
                await message.reply("❌ Сообщение должно быть короче 30 секунд")
                return

            file = await self.bot.get_file(message.voice.file_id)
            ogg_path = f"temp_voice_{message.from_user.id}.ogg"
            await self.bot.download_file(file.file_path, ogg_path)

            wav_path = await self._convert_audio(ogg_path)
            if not wav_path:
                raise ValueError("Ошибка конвертации аудио")

            text = await self._transcribe_audio(wav_path)
            if not text:
                raise ValueError("Не удалось распознать речь")

            await self._process_voice_message(text, message)

        except Exception as e:
            logger.error(f"Voice processing failed: {e}", exc_info=True)
            await message.reply("❌ Не удалось обработать голосовое сообщение")

        finally:
            logger.info(f"Завершение обработки голосового сообщения от пользователя {message.from_user.id}")
            for path in [ogg_path, wav_path]:
                try:
                    if path and os.path.exists(path):
                        os.remove(path)
                except Exception as e:
                    logger.warning(f"Error deleting {path}: {e}")

    async def _process_voice_message(self, text: str, message: Message) -> bool:
        """Обработка голосовых сообщений (отвечаем голосом)"""
        try:
            if not text or not isinstance(text, str):
                raise ValueError("Пустой или неверный тип текста")

            text = text.strip()
            if len(text) < 2:
                await message.answer("Пожалуйста, задайте более развернутый вопрос.")
                return False

            response = self.neuro.get_response(text)
            if not response:
                raise ValueError("Модель не вернула ответ")

            await self._send_voice_response(message, response)
            return True

        except Exception as e:
            logger.error(f"Voice processing error: {e}", exc_info=True)
            await message.answer("❌ Произошла ошибка. Попробуйте переформулировать вопрос.")
            return False

    async def _process_text_message(self, text: str, message: Message) -> bool:
        """Улучшенная обработка текстовых сообщений"""
        try:
            if not text or not isinstance(text, str):
                raise ValueError("Пустой или неверный тип текста")

            text = text.strip()
            if len(text) < 2:
                await message.answer("Пожалуйста, задайте более развернутый вопрос.")
                return False

            # Получаем ответ с обработкой ошибок
            try:
                response = self.neuro.get_response(text)
            except Exception as e:
                logger.error(f"Model error: {e}", exc_info=True)
                response = "Извините, произошла ошибка при обработке запроса. Пожалуйста, попробуйте еще раз."

            if not response:
                response = "Не удалось сгенерировать ответ. Пожалуйста, переформулируйте вопрос."

            # Отправка ответа с ограничением длины
            await message.answer(response[:4000], parse_mode="HTML")
            return True

        except Exception as e:
            logger.error(f"Text processing error: {e}", exc_info=True)
            await message.answer("❌ Произошла ошибка. Попробуйте переформулировать вопрос.")
            return False

    async def _handle_text(self, message: Message):
        """Обработка текстовых сообщений"""
        if not message.text:
            await message.answer("Пожалуйста, отправьте текстовое сообщение")
            return

        await self._process_text_message(message.text, message)

    async def _convert_audio(self, ogg_path: str) -> Optional[str]:
        """Конвертация OGG в WAV с обработкой ошибок"""
        try:
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                wav_path = tmp.name

            audio = AudioSegment.from_ogg(ogg_path)
            audio.export(wav_path, format="wav")

            if os.path.getsize(wav_path) == 0:
                raise ValueError("Конвертированный файл пуст")

            return wav_path

        except Exception as e:
            logger.error(f"Audio conversion error: {e}", exc_info=True)
            if 'wav_path' in locals() and os.path.exists(wav_path):
                os.remove(wav_path)
            return None

    async def _transcribe_audio(self, wav_path: str) -> Optional[str]:
        """Транскрибация аудио с обработкой ошибок"""
        try:
            with sr.AudioFile(wav_path) as source:
                audio = self.recognizer.record(source)
                text = self.recognizer.recognize_google(audio, language="ru-RU")
                logger.info(f"Распознанный текст: {text}")
                return text if text else None

        except sr.UnknownValueError:
            logger.warning("Речь не распознана")
        except sr.RequestError as e:
            logger.error(f"Ошибка сервиса распознавания: {e}")
        except Exception as e:
            logger.error(f"Transcription error: {e}", exc_info=True)
        return None

    async def _text_to_speech(self, text: str) -> Optional[str]:
        """Преобразует текст в голосовое сообщение (формат .ogg)"""
        try:
            with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp:
                output_path = tmp.name

            tts = gTTS(text=text, lang="ru", slow=False)
            tts.save(output_path)

            if os.path.getsize(output_path) == 0:
                raise ValueError("Сгенерированный аудиофайл пуст")

            return output_path

        except Exception as e:
            logger.error(f"Ошибка TTS: {e}", exc_info=True)
            return None

    async def _send_voice_response(self, message: Message, text: str):
        """Отправляет голосовой ответ"""
        voice_path = await self._text_to_speech(text)
        if not voice_path:
            return

        try:
            input_file = FSInputFile(voice_path, filename="response.ogg")
            await message.reply_voice(voice=input_file)
        except Exception as e:
            logger.error(f"Ошибка отправки голосового сообщения: {e}", exc_info=True)
        finally:
            if os.path.exists(voice_path):
                os.remove(voice_path)

    async def run(self):
        """Запуск бота"""
        try:
            logger.info("Starting bot...")
            await self.dp.start_polling(self.bot)
        except Exception as e:
            logger.critical(f"Bot crashed: {e}", exc_info=True)
        finally:
            await self.bot.session.close()

if __name__ == "__main__":
    bot = VoiceBot()
    asyncio.run(bot.run())
