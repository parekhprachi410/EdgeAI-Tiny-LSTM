#include <WiFi.h>
#include <HTTPClient.h>
#include <time.h>

#include <OneWire.h>
#include <DallasTemperature.h>

#include "tinylstm_model.h"

// ===============================
// WIFI CONFIG
// ===============================

const char* WIFI_SSID = "OPPO A5s";
const char* WIFI_PASSWORD = "sldc$123";

const char* GOOGLE_SCRIPT_URL =
"https://script.google.com/macros/s/AKfycby1fQJkQwVoWhyWypFbAEojjO0n0tcpT1ohR_QUz6f_3LwgoHx4jLlcq0WpkI_ACj-5/exec";

// ===============================
// TIME CONFIG
// ===============================

const char* NTP_SERVER = "pool.ntp.org";
const long GMT_OFFSET_SEC = 19800;
const int DAYLIGHT_OFFSET_SEC = 0;

// ===============================
// DS18B20 CONFIG
// ===============================

#define ONE_WIRE_BUS 4

OneWire oneWire(ONE_WIRE_BUS);
DallasTemperature sensors(&oneWire);

// ===============================
// MODEL BUFFER
// ===============================

float tempWindow[WINDOW_SIZE];

int windowIndex = 0;
bool windowFilled = false;

float lastPredictedTemp = 0.0f;
float lastValidTemp = 0.0f;

bool hasLastValidTemp = false;

// ===============================
// TIMING
// ===============================

unsigned long lastLogTime = 0;

const unsigned long LOG_INTERVAL = 360000;    // 6 minutes
const unsigned long WARMUP_INTERVAL = 2000;   // 2 sec during first 10 readings

// ===============================
// WIFI
// ===============================

void connectWiFi()
{
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

    Serial.print("Connecting to WiFi");

    while (WiFi.status() != WL_CONNECTED)
    {
        delay(500);
        Serial.print(".");
    }

    Serial.println();
    Serial.println("WiFi Connected");
    Serial.print("IP Address: ");
    Serial.println(WiFi.localIP());
}

// ===============================
// WINDOW BUFFER
// ===============================

void updateWindow(float temp)
{
    tempWindow[windowIndex] = temp;

    windowIndex++;

    if (windowIndex >= WINDOW_SIZE)
    {
        windowIndex = 0;
        windowFilled = true;
    }
}

void getOrderedWindow(float orderedWindow[WINDOW_SIZE])
{
    int index = windowIndex;

    for (int i = 0; i < WINDOW_SIZE; i++)
    {
        orderedWindow[i] = tempWindow[index];

        index++;

        if (index >= WINDOW_SIZE)
        {
            index = 0;
        }
    }
}

// ===============================
// DATE AND TIME
// ===============================

void getDateTime(String &dateStr, String &timeStr)
{
    struct tm timeinfo;

    dateStr = "NA";
    timeStr = "NA";

    if (getLocalTime(&timeinfo))
    {
        char dateBuffer[11];
        char timeBuffer[9];

        strftime(dateBuffer, sizeof(dateBuffer), "%Y-%m-%d", &timeinfo);
        strftime(timeBuffer, sizeof(timeBuffer), "%H:%M:%S", &timeinfo);

        dateStr = String(dateBuffer);
        timeStr = String(timeBuffer);
    }
}

// ===============================
// GOOGLE SHEET LOGGING
// ===============================

void sendToGoogleSheet(
    float realTemp,
    float predictedTemp,
    String status,
    float diff
)
{
    if (WiFi.status() != WL_CONNECTED)
    {
        connectWiFi();
    }

    String dateStr;
    String timeStr;
    getDateTime(dateStr, timeStr);

    String url = String(GOOGLE_SCRIPT_URL);
    url += "?date=" + dateStr;
    url += "&time=" + timeStr;
    url += "&timestamp=" + String(millis());
    url += "&real_temp=" + String(realTemp, 3);
    url += "&predicted_temp=" + String(predictedTemp, 3);
    url += "&status=" + status;
    url += "&diff=" + String(diff, 3);

    Serial.println(url);

    HTTPClient http;
    http.setFollowRedirects(HTTPC_FORCE_FOLLOW_REDIRECTS);
    http.begin(url);

    int httpResponseCode = http.GET();

    Serial.print("HTTP Response: ");
    Serial.println(httpResponseCode);

    Serial.print("Server Response: ");
    Serial.println(http.getString());

    http.end();
}

// ===============================
// SETUP
// ===============================

void setup()
{
    Serial.begin(115200);

    sensors.begin();

    connectWiFi();

    configTime(
        GMT_OFFSET_SEC,
        DAYLIGHT_OFFSET_SEC,
        NTP_SERVER
    );

    Serial.println("TinyLSTM Hydroponics Forecasting Started");
    Serial.println("Warm-up mode: collecting first 10 readings quickly");
}

// ===============================
// LOOP
// ===============================

void loop()
{
    unsigned long currentInterval;

    if (windowFilled)
    {
        currentInterval = LOG_INTERVAL;
    }
    else
    {
        currentInterval = WARMUP_INTERVAL;
    }

    if (millis() - lastLogTime >= currentInterval)
    {
        lastLogTime = millis();

        sensors.requestTemperatures();

        float currentTemp = sensors.getTempCByIndex(0);

        bool sensorConnected = true;

        if (
            currentTemp == DEVICE_DISCONNECTED_C ||
            currentTemp < -50.0f ||
            currentTemp > 100.0f
        )
        {
            sensorConnected = false;
        }

        float orderedWindow[WINDOW_SIZE];
        float predictedTemp = lastPredictedTemp;
        float diff = 0.0f;

        if (sensorConnected)
        {
            updateWindow(currentTemp);

            if (windowFilled)
            {
                getOrderedWindow(orderedWindow);

                predictedTemp = predictTinyLSTM(orderedWindow);

                if (hasLastValidTemp)
                {
                    float delta = currentTemp - lastValidTemp;

                    if (abs(delta) > 0.5f)
                    {
                        predictedTemp =
                            currentTemp + (delta * 0.3f);
                    }
                }

                lastPredictedTemp = predictedTemp;
                diff = currentTemp - predictedTemp;
            }
            else
            {
                predictedTemp = currentTemp;
                lastPredictedTemp = predictedTemp;
                diff = 0.0f;
            }

            lastValidTemp = currentTemp;
            hasLastValidTemp = true;

            Serial.print("Real Temp: ");
            Serial.print(currentTemp);
            Serial.print(" C | Predicted: ");
            Serial.print(predictedTemp);
            Serial.print(" C | Diff: ");
            Serial.print(diff);

            if (windowFilled)
            {
                Serial.println(" | Status: connected | Mode: 6-min logging");
            }
            else
            {
                Serial.println(" | Status: connected | Mode: warm-up");
            }

            sendToGoogleSheet(
                currentTemp,
                predictedTemp,
                "connected",
                diff
            );
        }
        else
        {
            if (windowFilled)
            {
                getOrderedWindow(orderedWindow);

                predictedTemp = predictTinyLSTM(orderedWindow);

                lastPredictedTemp = predictedTemp;

                updateWindow(predictedTemp);
            }
            else
            {
                predictedTemp = lastPredictedTemp;
            }

            Serial.print("Sensor disconnected | Predicted: ");
            Serial.print(predictedTemp);
            Serial.println(" C");

            sendToGoogleSheet(
                -999.0f,
                predictedTemp,
                "disconnected",
                0.0f
            );
        }
    }
}