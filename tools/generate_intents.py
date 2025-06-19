import random

# Конфигурационные параметры
MAX_QUESTIONS = 150
MAX_RESPONSES = 20

# Увеличим базовый набор вопросов
base_questions = [
    "Сколько стоит монитор?", "Какая цена на монитор?", "Цена монитора?",
    "Какой ценник у монитора?", "Сколько стоит этот экран?", "Какая стоимость дисплея?",
    "Какова цена этого монитора?", "Сколько просите за монитор?", "Ценник на монитор?",
    "Сколько стоит этот монитор?", "Какой прайс на монитор?", "Какая расценка на дисплей?",
    "Сколько берут за монитор?", "Какова стоимость экрана?", "Какой ценник на этот монитор?",
    "Во сколько обойдётся монитор?", "Какая цена у монитора?", "Сколько стоит такой монитор?",
    "Какой стоимости монитор?", "Цена на монитор?"
]

# Варианты типов и характеристик
monitor_types = [
    "", "игровой", "офисный", "домашний", "профессиональный", "бюджетный",
    "премиальный", "ультратонкий", "для дизайна", "для программистов", "для геймеров"
]

monitor_features = [
    "", "с IPS матрицей", "с VA матрицей", "с TN матрицей", "с OLED экраном",
    "с диагональю {size} дюймов", "с разрешением {resolution}", "с частотой {hz} Гц",
    "с HDR", "с изогнутым экраном", "с поддержкой G-Sync", "с поддержкой FreeSync"
]

feature_values = {
    "size": ["24", "27", "32", "34"],
    "resolution": ["Full HD", "2K", "4K"],
    "hz": ["60", "144", "240"]
}

# Синонимы для генерации вариантов
synonyms = {
    'стоит': ['цена', 'стоимость', 'ценник'],
    'монитор': ['экран', 'дисплей'],
    'какая': ['какова', 'сколько']
}


def generate_questions():
    questions = set()

    # Базовые вопросы
    for q in base_questions:
        if len(questions) >= MAX_QUESTIONS:
            break

        questions.add(q)

        # Вопросы с типами
        for mtype in monitor_types:
            if mtype and len(questions) < MAX_QUESTIONS:
                questions.add(q.replace("монитор", f"{mtype} монитор"))

        # Вопросы с характеристиками
        for feature in monitor_features:
            if feature and len(questions) < MAX_QUESTIONS:
                if "{" in feature:
                    for key in feature_values:
                        if f"{{{key}}}" in feature:
                            for value in feature_values[key]:
                                filled = feature.replace(f"{{{key}}}", value)
                                new_q = q.replace("монитор", f"монитор {filled}")
                                if len(questions) < MAX_QUESTIONS:
                                    questions.add(new_q)
                else:
                    new_q = q.replace("монитор", f"монитор {feature}")
                    if len(questions) < MAX_QUESTIONS:
                        questions.add(new_q)

    # Дополнительная синонимизация
    augmented = set(questions)
    for q in list(augmented):
        if len(augmented) >= MAX_QUESTIONS:
            break
        words = q.split()
        for i, word in enumerate(words):
            if word.lower() in synonyms:
                for synonym in synonyms[word.lower()]:
                    new_words = words.copy()
                    new_words[i] = synonym
                    new_q = ' '.join(new_words)
                    if len(augmented) < MAX_QUESTIONS:
                        augmented.add(new_q)

    return list(augmented)[:MAX_QUESTIONS]


# Генерация вопросов
questions = generate_questions()

# Система ответов
price_ranges = {
    # Типы
    "игровой": (15000, 50000),
    "офисный": (8000, 20000),
    "профессиональный": (30000, 150000),
    "премиальный": (50000, 300000),
    # Характеристики
    "IPS": (10000, 40000),
    "4K": (25000, 100000),
    "144": (20000, 60000),
    "240": (30000, 80000)
}

# 20 уникальных шаблонов ответов
response_templates = [
    "Стоимость такого монитора — от {min} до {max} рублей.",
    "Цена варьируется от {min} до {max} рублей.",
    "Примерный диапазон цен: {min}-{max} рублей.",
    "Такой монитор стоит от {min} до {max} руб.",
    "В магазинах цена от {min} до {max} рублей.",
    "Ориентировочная стоимость: {min}-{max} руб.",
    "Можно найти за {min}-{max} рублей.",
    "Средняя цена: {min}-{max} руб.",
    "Продаётся по цене от {min} до {max} руб.",
    "Ценник: {min}-{max} рублей.",
    "Стоимость в районе {min}-{max} руб.",
    "За такой просят от {min} до {max} рублей.",
    "Рыночная цена: {min}-{max} руб.",
    "Предложения от {min} до {max} рублей.",
    "Ценовой диапазон: {min}-{max} руб.",
    "Стоит примерно {min}-{max} рублей.",
    "Цены начинаются от {min} руб. до {max} руб.",
    "Продаётся в пределах {min}-{max} руб.",
    "Найдёте за {min}-{max} рублей.",
    "Стоимость составит {min}-{max} руб."
]


def generate_response(question):
    # Определяем тип
    monitor_type = next((t for t in price_ranges if t in question.lower()), "basic")

    # Базовый диапазон
    min_price, max_price = price_ranges.get(monitor_type, (8000, 15000))

    # Модификаторы характеристик
    for feature in ["IPS", "4K", "144", "240"]:
        if feature in question:
            f_min, f_max = price_ranges[feature]
            min_price = int((min_price + f_min) / 2)
            max_price = int((max_price + f_max) / 2)

    # Выбираем случайный шаблон из 20 вариантов
    template = random.choice(response_templates)
    return template.format(min=min_price, max=max_price)


# Группируем вопросы по типам для разнообразия ответов
question_groups = {}
for q in questions:
    key = next((t for t in price_ranges if t in q.lower()), "basic")
    if key not in question_groups:
        question_groups[key] = []
    question_groups[key].append(q)

# Распределяем 20 ответов по группам
responses_per_group = MAX_RESPONSES // len(question_groups)
response_map = {}

for i, (group, q_list) in enumerate(question_groups.items()):
    for j, q in enumerate(q_list):
        response_idx = i * responses_per_group + (j % responses_per_group)
        response_map[q] = response_templates[response_idx % MAX_RESPONSES].format(
            min=price_ranges.get(group, (8000, 15000))[0],
            max=price_ranges.get(group, (8000, 15000))[1]
        )

# Формируем финальный конфиг
BOT_CONFIG = {
    'intents': {
        'price_monitors': {
            'examples': questions,
            'responses': [response_map[q] for q in questions]
        }
    }
}

# Сохранение в файл
with open('output_config.py', 'w', encoding='utf-8') as f:
    f.write("BOT_CONFIG = {\n")
    f.write("    'intents': {\n")
    f.write("        'price_monitors': {\n")
    f.write("            'examples': [\n")
    for example in questions:
        f.write(f"                '{example}',\n")
    f.write("            ],\n")
    f.write("            'responses': [\n")
    for response in [response_map[q] for q in questions]:
        f.write(f"                '{response}',\n")
    f.write("            ],\n")
    f.write("        }\n")
    f.write("    }\n")
    f.write("}\n")

print(f"Сгенерировано {len(questions)} вопросов и {MAX_RESPONSES} вариантов ответов")