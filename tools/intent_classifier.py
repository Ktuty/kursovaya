import os
import re
from typing import Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class IntentClassifier:
    def __init__(self, config_path: str = "../configs/sk_config/config_dialog2.py",
                 model_path: str = "../configs/sk_config/intent_model.joblib"):
        self.model_path = model_path
        self.config_path = config_path
        self._prepare_data()

    def _prepare_data(self):
        """Подготовка данных для обучения из BOT_CONFIG"""
        from configs.config_dialogs.config_dialog import BOT_CONFIG

        self.texts = []
        self.labels = []

        for intent_name, intent_data in BOT_CONFIG['intents'].items():
            for example in intent_data['examples']:
                clean_text = self._clean_text(example)
                self.texts.append(clean_text)
                self.labels.append(intent_name)

    def _plot_metrics(self, y_test, y_pred, classes):
        """Визуализация метрик классификации"""
        plt.figure(figsize=(15, 10))

        # 1. Confusion Matrix
        plt.subplot(2, 2, 1)
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')

        # 2. Classification Report (Bar plot)
        plt.subplot(2, 2, 2)
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        metrics = ['precision', 'recall', 'f1-score']
        for i, metric in enumerate(metrics):
            values = [report[cls][metric] for cls in classes if cls in report]
            plt.bar(np.arange(len(values)) + i * 0.25, values, width=0.25, label=metric)

        plt.xticks(np.arange(len(classes)) + 0.25, classes, rotation=45)
        plt.title('Classification Metrics')
        plt.legend()
        plt.ylim(0, 1.1)

        # 3. Accuracy
        plt.subplot(2, 2, 3)
        accuracy = accuracy_score(y_test, y_pred)
        plt.bar(['Accuracy'], [accuracy], color='green')
        plt.ylim(0, 1.1)
        plt.title(f'Model Accuracy: {accuracy:.2f}')

        # 4. Class Distribution
        plt.subplot(2, 2, 4)
        class_counts = {cls: list(y_test).count(cls) for cls in classes}
        plt.bar(class_counts.keys(), class_counts.values(), color='orange')
        plt.title('Class Distribution in Test Set')
        plt.xticks(rotation=45)

        plt.tight_layout()
        plt.savefig('../configs/sk_config/metrics_plot.png')  # Сохраняем график
        plt.show()

    def train(self, test_size: float = 0.2, random_state: int = 42):
        """Обучение модели классификации намерений с визуализацией метрик"""
        # Разделение данных на train/test
        X_train, X_test, y_train, y_test = train_test_split(
            self.texts, self.labels,
            test_size=test_size,
            random_state=random_state
        )

        # Создание пайплайна
        self.model = Pipeline([
            ('tfidf', TfidfVectorizer(
                lowercase=True,
                max_features=10000,
                ngram_range=(1, 2),
                stop_words=None
            )),
            ('clf', LinearSVC(
                C=1.0,
                class_weight='balanced',
                dual=True,
                max_iter=1000,
                random_state=random_state
            ))
        ])

        # Обучение модели
        self.model.fit(X_train, y_train)

        # Предсказания и оценка качества
        y_pred = self.model.predict(X_test)

        # Уникальные классы
        classes = sorted(set(self.labels))

        # Вывод метрик в консоль
        print(f"\n{'=' * 50}")
        print(f"Model accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, zero_division=0))

        # Визуализация метрик
        self._plot_metrics(y_test, y_pred, classes)

        # Сохранение модели
        joblib.dump(self.model, self.model_path)
        return self.model

    def load_model(self):
        """Загрузка предобученной модели"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file {self.model_path} not found")
        self.model = joblib.load(self.model_path)
        return self.model

    def predict(self, text: str) -> Tuple[str, float]:
        """Предсказание намерения с улучшенной обработкой ошибок"""
        if not hasattr(self, 'model'):
            try:
                self.load_model()
            except Exception as e:
                print(f"Ошибка загрузки модели: {str(e)}")
                return "unknown", 0.0

        try:
            clean_text = self._clean_text(text)
            if not clean_text:
                return "unknown", 0.0

            intent = self.model.predict([clean_text])[0]
            decision_scores = self.model.decision_function([clean_text])[0]
            exp_scores = np.exp(decision_scores - np.max(decision_scores))
            probs = exp_scores / np.sum(exp_scores)
            confidence = np.max(probs)

            return intent, float(confidence)
        except Exception as e:
            print(f"Ошибка предсказания: {str(e)}")
            return "unknown", 0.0

    def _clean_text(self, text: str) -> str:
        """Очистка текста для классификации"""
        text = text.lower().strip()
        text = re.sub(r"[^\w\s]", "", text)
        return text


if __name__ == "__main__":
    classifier = IntentClassifier()
    print("Training intent classifier...")
    classifier.train()
    print(f"\nModel saved to {classifier.model_path}")
    print("Metrics plot saved to '../configs/sk_config/metrics_plot.png'")

    # Тестовые предсказания
    test_phrases = [
        "сколько стоит монитор",
        "какой монитор лучше для игр",
        "есть ли скидки",
        "покажите все модели"
    ]

    print("\nTest predictions:")
    for phrase in test_phrases:
        intent, confidence = classifier.predict(phrase)
        print(f"'{phrase}' -> {intent} (confidence: {confidence:.2f})")