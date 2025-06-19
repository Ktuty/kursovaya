import os
import torch
import importlib
import logging
from torch.optim import AdamW
from transformers import (
    get_linear_schedule_with_warmup, GPT2Tokenizer, GPT2LMHeadModel
)
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

# 1. Динамическая загрузка конфигурации
from configs.program_config import MODEL_LEARN, NEW_MODEL_NAME, VERSION_CONF_DIALOG

# Загружаем нужную версию конфига
try:
    config_module = importlib.import_module(
        f"configs.config_dialogs.config_dialog_{VERSION_CONF_DIALOG.replace('.', '_')}"
    )
    BOT_CONFIG = config_module.BOT_CONFIG
except ImportError as e:
    raise ImportError(f"Не удалось загрузить конфиг версии {VERSION_CONF_DIALOG}. Ошибка: {str(e)}")

logger = logging.getLogger(__name__)

# Настройки
BATCH_SIZE = 2
GRADIENT_ACCUM_STEPS = 2
MAX_LENGTH = 128

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cache_path = MODEL_LEARN
snapshot = sorted(os.listdir(cache_path))[-1]  # Берём последнюю версию
model_path = f"{cache_path}"
logger.info("Инициализация модели и токенизатора...")
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path).to(device)
tokenizer.pad_token = tokenizer.eos_token
print("Модель загружена:", model is not None)
input_text = "Привет"
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(**inputs)
print(tokenizer.decode(outputs[0]))
logger.info(f"Модель загружена на устройство: {device}")

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)

model.train()

# Подготовка данных
class ChatDataset(Dataset):
    def __init__(self, train_texts, tokenizer, max_length=128):
        self.texts = train_texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.bot_token = tokenizer.encode("\nБот:")[0]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            add_special_tokens=True
        )

        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()

        # Маска для обучения только на ответах бота
        labels = input_ids.clone()
        try:
            bot_pos = (input_ids == self.bot_token).nonzero()[0].item()
            labels[:bot_pos] = -100  # Игнорируем в loss
        except IndexError:
            pass

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

# Создание датасета
train_texts = [
    f"Пользователь: {ex}\nБот: {resp}"
    for intent in BOT_CONFIG['intents'].values()
    for ex, resp in zip(intent['examples'], intent['responses'])
]

train_dataset = ChatDataset(train_texts, tokenizer, MAX_LENGTH)
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

# Настройка обучения
optimizer = AdamW(model.parameters(), lr=5e-5)
total_steps = len(train_loader) * 3 // GRADIENT_ACCUM_STEPS
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(total_steps * 0.1),
    num_training_steps=total_steps
)

# Обучение
for epoch in range(3):
    model.train()
    total_loss = 0

    progress = tqdm(train_loader, desc=f'Epoch {epoch + 1}')
    for batch in progress:
        inputs = {k: v.to(device) for k, v in batch.items()}

        outputs = model(**inputs)
        loss = outputs.loss / GRADIENT_ACCUM_STEPS
        loss.backward()

        if (progress.n + 1) % GRADIENT_ACCUM_STEPS == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item()
        progress.set_postfix({'loss': loss.item()})

    # Сохранение модели
    output_dir = f"../model_cache/{NEW_MODEL_NAME}"
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"Эпоха {epoch + 1} завершена. Loss: {total_loss / len(train_loader):.4f}")
    logger.info(f"Модель сохранена в {output_dir}")

logger.info("Обучение завершено!")