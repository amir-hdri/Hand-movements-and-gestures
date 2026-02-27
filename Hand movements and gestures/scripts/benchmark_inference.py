import time
import numpy as np
import tensorflow as tf

def main():
    print(f"TensorFlow Version: {tf.__version__}")

    # For TF 2.x, using @tf.function can drastically reduce overhead for repeated single-item inferences
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(30, 99)),
        tf.keras.layers.LSTM(64),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')
    ])

    dummy_input = np.random.rand(1, 30, 99).astype(np.float32)

    fast_predict = tf.function(model, reduce_retracing=True)

    # Warmup
    _ = model.predict(dummy_input, verbose=0)
    _ = fast_predict(dummy_input, training=False)

    num_iterations = 100

    print(f"\nBenchmarking {num_iterations} iterations of single-sample inference...")

    start_predict = time.perf_counter()
    for _ in range(num_iterations):
        _ = model.predict(dummy_input, verbose=0)
    end_predict = time.perf_counter()
    predict_time = end_predict - start_predict

    start_fast = time.perf_counter()
    for _ in range(num_iterations):
        _ = fast_predict(dummy_input, training=False)
    end_fast = time.perf_counter()
    fast_time = end_fast - start_fast

    print(f"model.predict():           {predict_time:.4f} seconds ({predict_time/num_iterations * 1000:.2f} ms/iter)")
    print(f"tf.function(model):        {fast_time:.4f} seconds ({fast_time/num_iterations * 1000:.2f} ms/iter)")

    speedup = predict_time / fast_time
    print(f"\nSpeedup: {speedup:.2f}x faster")

if __name__ == '__main__':
    main()
