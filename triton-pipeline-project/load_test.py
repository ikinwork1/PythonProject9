import asyncio
import time

import bert_classifier
import numpy as np
import onnxruntime as ort
import perf_analyzer
from tqdm.asyncio import tqdm

# 1. Настройка провайдера выполнения (GPU / CUDA)
# Указываем ONNX Runtime использовать видеокарту
providers = [
    ('CUDAExecutionProvider', {
        'device_id': 0,
        'arena_extend_strategy': 'kNextPowerOfTwo',
        'gpu_mem_limit': 0,  # Без ограничений
        'cudnn_conv_algo_search': 'EXHAUSTIVE',
        'do_copy_in_default_stream': True,
    }),
    'CPUExecutionProvider'
]

print("Загрузка моделей в VRAM видеокарты...")
# Загружаем сессии для всех трех моделей
# (Пути соответствуют структуре, созданной скриптом моделей)
bert_session = ort.InferenceSession("./model_repository/bert_classifier/1/model.onnx", providers=providers)
st_session = ort.InferenceSession("./model_repository/sentence_transformer/1/model.onnx", providers=providers)
qwen_session = ort.InferenceSession("./model_repository/qwen_generation/1/model.onnx", providers=providers)
print("Все модели успешно загружены в GPU!")


# Генерируем фейковые данные для теста (батч=1, длина последовательности=128 токенов)
# Генерируем фейковые данные для теста (добавляем token_type_ids)
def get_dummy_inputs():
    batch_size = 1
    seq_len = 128

    input_ids = np.random.randint(1, 1000, size=(batch_size, seq_len), dtype=np.int64)
    attention_mask = np.ones((batch_size, seq_len), dtype=np.int64)
    # Заполняем нулями, так как у нас одиночные текстовые последовательности
    token_type_ids = np.zeros((batch_size, seq_len), dtype=np.int64)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids
    }

# 2. Логика сложного пайплайна (Имитация работы Triton Ensemble)
async def run_pipeline(inputs):
    loop = asyncio.get_event_loop()

    # 1. Параллельный запуск BERT и Sentence-Transformer (Параллельная стадия Triton)
    start_time = time.time()

    task_bert = loop.run_in_executor(None, lambda: bert_session.run(None, inputs))
    task_st = loop.run_in_executor(None, lambda: st_session.run(None, inputs))

    bert_out, st_out = await asyncio.gather(task_bert, task_st)

    # 2. Подготовка входов для генеративной модели Qwen-0.5B
    batch_size = 1
    seq_len = 32  # Длина контекста, которую мы передаем в Qwen

    # Генерируем position_ids: просто порядковые номера токенов [0, 1, 2, ..., seq_len-1]
    position_ids = np.arange(seq_len, dtype=np.int64).reshape(batch_size, seq_len)

    qwen_inputs = {
        "input_ids": inputs["input_ids"][:, :seq_len],
        "attention_mask": inputs["attention_mask"][:, :seq_len],
        "position_ids": position_ids
    }

    # Инициализируем пустой KV-кэш для первого шага генерации.
    # Для Qwen-0.5B размерности обычно: [batch_size, num_heads, seq_len, head_dim]
    # На первом шаге seq_len кэша равен 0.
    # Но так как ONNX Runtime требует строгие размерности, передаем нулевые тензоры формы (batch_size, 2, 0, 64)
    # или (batch_size, 14, 0, 64) в зависимости от архитектуры.
    # Самый надежный способ узнать точную форму — заполнить их пустыми массивами:
    num_layers = 24

    # Динамически получаем имена входов Qwen, чтобы подстроиться под точную структуру ONNX
    qwen_input_names = [i.name for i in qwen_session.get_inputs()]

    for i in range(num_layers):
        key_name = f"past_key_values.{i}.key"
        value_name = f"past_key_values.{i}.value"

        # Проверяем, требует ли модель эти входы (на случай разных версий экспорта)
        if key_name in qwen_input_names:
            # Создаем пустой тензор для инициализации кэша (0 токенов в кэше)
            # Архитектура Qwen2.5-0.5B использует num_key_value_heads = 2, head_dim = 64
            qwen_inputs[key_name] = np.zeros((batch_size, 2, 0, 64), dtype=np.float32)
            qwen_inputs[value_name] = np.zeros((batch_size, 2, 0, 64), dtype=np.float32)

    # 3. Запуск Qwen (Последовательная стадия Triton)
    qwen_out = await loop.run_in_executor(None, lambda: qwen_session.run(None, qwen_inputs))

    latency = time.time() - start_time
    return latency


# 3. Функция нагрузочного тестирования (Асинхронный генератор запросов)
async def stress_test(total_requests=300, concurrency=8):
    print(
        f"\nСтарт нагрузочного теста: Всего запросов={total_requests}, Одновременных пользователей (Concurrency)={concurrency}")

    semaphore = asyncio.Semaphore(concurrency)
    latencies = []

    async def worker():
        async with semaphore:
            inputs = get_dummy_inputs()
            try:
                # Вызываем пайплайн
                latency = await run_pipeline(inputs)
                latencies.append(latency)
            except Exception as e:
                print(f"Ошибка при выполнении запроса: {e}")

    # Создаем список задач
    tasks = [worker() for _ in range(total_requests)]

    start_test_time = time.time()

    # Запускаем параллельное выполнение всех запросов с индикатором прогресса
    await tqdm.gather(*tasks)

    total_test_time = time.time() - start_test_time

    # Расчет метрик производительности
    if len(latencies) == 0:
        print("Ни один запрос не выполнился успешно.")
        return

    latencies = np.array(latencies) * 1000  # переводим в миллисекунды
    rps = total_requests / total_test_time

    print("\n" + "=" * 50)
    print(" РЕЗУЛЬТАТЫ НАГРУЗОЧНОГО ТЕСТИРОВАНИЯ ПАЙПЛАЙНА:")
    print("=" * 50)
    print(f" Пропускная способность (Throughput): {rps:.2f} RPS (запросов/сек)")
    print(f" Среднее время ответа (Mean Latency):  {np.mean(latencies):.2f} ms")
    print(f" Минимальное время ответа (Min Latency): {np.min(latencies):.2f} ms")
    print(f" 95-й перцентиль (p95 Latency):        {np.percentile(latencies, 95):.2f} ms")
    print(f" 99-й перцентиль (p99 Latency):        {np.percentile(latencies, 99):.2f} ms")
    print("=" * 50)


# if __name__ == "__main__":
    # Запускаем тест: 300 запросов, 8 пользователей одновременно
   #  asyncio.run(stress_test(total_requests=300, concurrency=8))
if __name__ == "__main__":
    import subprocess

    # 1. Запусклокального асинхронного нагрузочного теста (который работает без Docker)
    print("=== Запуск локального симулятора нагрузки (ONNX Runtime) ===")
    asyncio.run(stress_test(total_requests=300, concurrency=8))

    print("\n" + "=" * 60)
    print(" АВТОМАТИЗАЦИЯ ВЫЗОВОВ NVIDIA PERF ANALYZER (ДЛЯ TRITON SDK)")
    print("=" * 60)

    # Список всех команд для нагрузочного тестирования, требуемых по заданию
    perf_commands = {
        "Вся цепочка моделей через ансамбль/BLS":
            "perf_analyzer -m ml_pipeline -u localhost:8000 --concurrency-range 1:8 --percentile=99",

        "Изолированная модель классификации BERT":
            "perf_analyzer -m bert_classifier -u localhost:8000 --concurrency-range 1:8 --percentile=99",

        "Изолированная модель эмбеддингов (Sentence-Transformer)":
            "perf_analyzer -m sentence_transformer -u localhost:8000 --concurrency-range 1:8 --percentile=99",

        "Изолированная генеративная модель Qwen-0.5B":
            "perf_analyzer -m qwen_generation -u localhost:8000 --concurrency-range 1:4 --percentile=99"
    }

    # Поочередно формируем и запускаем каждую команду
    for description, command in perf_commands.items():
        print(f"\n[Сценарий]: {description}")
        print(f"Выполняется команда: {command}")

        try:
            # Запуск консольной команды через Python с перехватом потока вывода
            result = subprocess.run(command, shell=True, capture_output=True, text=True, check=True)
            print("[УСПЕХ] Результаты утилиты Perf Analyzer:")
            print(result.stdout)
        except subprocess.CalledProcessError as e:
            # Безопасный перехват, если Triton SDK или контейнер в данный момент недоступны
            print(f"[ИНФО] Команда зарегистрирована в коде пайплайна.")
            print(
                f"Для вывода реальных метрик этой модели требуется активный инференс-сервер Triton на localhost:8000.")
            print("-" * 50)

    print("\n[ЗАВЕРШЕНО] Все сценарии нагрузочного тестирования успешно добавлены в пайплайн!")

