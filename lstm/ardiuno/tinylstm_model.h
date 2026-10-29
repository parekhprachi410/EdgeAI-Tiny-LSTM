#ifndef TINYLSTM_MODEL_H
#define TINYLSTM_MODEL_H

#define WINDOW_SIZE 10
#define LSTM_UNITS 16
#define DENSE1_UNITS 8

#define DATA_MIN 7.312f
#define DATA_MAX 38.938f

float predictTinyLSTM(float inputWindow[WINDOW_SIZE]);

#endif