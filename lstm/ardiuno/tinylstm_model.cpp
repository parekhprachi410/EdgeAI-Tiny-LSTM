#include "tinylstm_model.h"
#include "tinylstm_weights.h"
#include <math.h>

// Keras LSTM gate order: input, forget, cell, output
// gates = [i, f, c, o]

static float sigmoid(float x) {
  return 1.0f / (1.0f + expf(-x));
}

static float relu(float x) {
  return x > 0.0f ? x : 0.0f;
}

static float normalizeTemp(float temp) {
  return (temp - DATA_MIN) / (DATA_MAX - DATA_MIN);
}

static float denormalizeTemp(float value) {
  return value * (DATA_MAX - DATA_MIN) + DATA_MIN;
}

float predictTinyLSTM(float inputWindow[WINDOW_SIZE]) {

  float h[LSTM_UNITS];
  float c[LSTM_UNITS];

  for (int i = 0; i < LSTM_UNITS; i++) {
    h[i] = 0.0f;
    c[i] = 0.0f;
  }

  // =========================
  // LSTM FORWARD PASS
  // =========================
  for (int t = 0; t < WINDOW_SIZE; t++) {

    float x = normalizeTemp(inputWindow[t]);

    float new_h[LSTM_UNITS];
    float new_c[LSTM_UNITS];

    for (int unit = 0; unit < LSTM_UNITS; unit++) {

      float gate_i = lstm_bias[unit];
      float gate_f = lstm_bias[LSTM_UNITS + unit];
      float gate_c = lstm_bias[2 * LSTM_UNITS + unit];
      float gate_o = lstm_bias[3 * LSTM_UNITS + unit];

      // Input kernel contribution
      gate_i += x * lstm_kernel_flat[unit];
      gate_f += x * lstm_kernel_flat[LSTM_UNITS + unit];
      gate_c += x * lstm_kernel_flat[2 * LSTM_UNITS + unit];
      gate_o += x * lstm_kernel_flat[3 * LSTM_UNITS + unit];

      // Recurrent contribution
      for (int j = 0; j < LSTM_UNITS; j++) {
        gate_i += h[j] * lstm_recurrent_kernel_flat[j * 4 * LSTM_UNITS + unit];
        gate_f += h[j] * lstm_recurrent_kernel_flat[j * 4 * LSTM_UNITS + LSTM_UNITS + unit];
        gate_c += h[j] * lstm_recurrent_kernel_flat[j * 4 * LSTM_UNITS + 2 * LSTM_UNITS + unit];
        gate_o += h[j] * lstm_recurrent_kernel_flat[j * 4 * LSTM_UNITS + 3 * LSTM_UNITS + unit];
      }

      float i_gate = sigmoid(gate_i);
      float f_gate = sigmoid(gate_f);
      float c_gate = tanhf(gate_c);
      float o_gate = sigmoid(gate_o);

      new_c[unit] = f_gate * c[unit] + i_gate * c_gate;
      new_h[unit] = o_gate * tanhf(new_c[unit]);
    }

    for (int unit = 0; unit < LSTM_UNITS; unit++) {
      h[unit] = new_h[unit];
      c[unit] = new_c[unit];
    }
  }

  // =========================
  // DENSE 1: 16 -> 8
  // =========================
  float dense1[DENSE1_UNITS];

  for (int i = 0; i < DENSE1_UNITS; i++) {
    float sum = dense1_bias[i];

    for (int j = 0; j < LSTM_UNITS; j++) {
      sum += h[j] * dense1_weights_flat[j * DENSE1_UNITS + i];
    }

    dense1[i] = relu(sum);
  }

  // =========================
  // DENSE 2: 8 -> 1
  // =========================
  float output = dense2_bias[0];

  for (int i = 0; i < DENSE1_UNITS; i++) {
    output += dense1[i] * dense2_weights_flat[i];
  }

  return denormalizeTemp(output);
}