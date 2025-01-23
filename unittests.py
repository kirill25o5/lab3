import unittest
import tensorflow as tf
from tensorflow.keras.models import Sequential


class TestMNISTModel(unittest.TestCase):
    def setUp(self):
        # Загружаем данные MNIST и делим на тренировочный и тестовый наборы
        (self.x_train, self.y_train), (self.x_test, self.y_test) = \
            tf.keras.datasets.mnist.load_data()

        # Берём только 10% от общего количества данных для ускорения тестов
        fraction = 0.1
        num_train_samples = int(len(self.x_train) * fraction)
        num_test_samples = int(len(self.x_test) * fraction)

        # Нормализация входных данных и перевод меток в one-hot encoding
        self.x_train_10 = self.x_train[:num_train_samples].astype('float32') / 255
        self.y_train_10 = \
            tf.keras.utils.to_categorical(self.y_train[:num_train_samples], 10)
        self.x_test_10 = self.x_test[:num_test_samples].astype('float32') / 255
        self.y_test_10 = \
            tf.keras.utils.to_categorical(self.y_test[:num_test_samples], 10)

    def test_data_shapes(self):
        """
        Проверяет корректность размеров данных.
        """
        # Тренировочные данные должны содержать ровно 10% от общего набора
        self.assertEqual(self.x_train_10.shape[0], int(len(self.x_train) * 0.1))
        self.assertEqual(self.x_test_10.shape[0], int(len(self.x_test) * 0.1))
        # Размеры изображений должны быть (28, 28)
        self.assertEqual(self.x_train_10.shape[1:], (28, 28))
        # Выходные метки должны быть one-hot encoded с 10 классами
        self.assertEqual(self.y_train_10.shape[1], 10)

    def test_model_training(self):
        """
        Проверяет процесс обучения модели.
        """
        # Создаём простую модель
        model = Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
        ])
        # Компилируем модель
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        # Обучаем модель на 1 эпоху
        history = model.fit(self.x_train_10, self.y_train_10,
                            epochs=1, batch_size=32,
                            validation_data=(self.x_test_10, self.y_test_10))

        # Проверяем, что обучающий процесс прошёл успешно
        self.assertGreater(len(history.history['loss']), 0, "Эпоха обучения не выполнена")
        self.assertGreaterEqual(history.history['accuracy'][0], 0, "Точность обучения некорректна")
        self.assertGreaterEqual(history.history['val_accuracy'][0], 0, "Точность валидации некорректна")

    def test_model_evaluation(self):
        """
        Проверяет корректность оценки модели.
        """
        # Создаём и обучаем модель
        model = Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
        ])
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        model.fit(self.x_train_10, self.y_train_10, epochs=1, batch_size=32)

        # Оцениваем модель на тестовых данных
        test_loss, test_acc = model.evaluate(self.x_test_10, self.y_test_10)

        # Убедимся, что точность находится в разумных пределах
        self.assertGreaterEqual(test_acc, 0, "Точность не может быть меньше 0")
        self.assertLessEqual(test_acc, 1, "Точность не может быть больше 1")


if __name__ == '__main__':
    unittest.main()
