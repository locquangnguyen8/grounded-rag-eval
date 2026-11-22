# Sensor Network

Each HEMP station uses a multiparameter sonde that measures water temperature, salinity, dissolved oxygen, turbidity and chlorophyll-a fluorescence.

Sondes record a reading every 15 minutes. Readings are sent to the central database over a LoRaWAN telemetry link. If the link fails, the sonde stores up to 30 days of readings locally and uploads them when the link recovers.

Biofouling is the main cause of sensor drift. Field technicians clean the sensors and replace wiper brushes during each scheduled visit, at least every 21 days. Dissolved oxygen sensors are recalibrated every 42 days, or sooner if a field check differs from the sonde reading by more than 0.5 mg/L.

Salinity is reported in practical salinity units (PSU). Turbidity is reported in nephelometric turbidity units (NTU).
