# Data Platform

HEMP readings are stored in a PostgreSQL database with the TimescaleDB extension, which partitions time-series data by time for fast queries.

A REST API serves quality-controlled data to the dashboard and to researchers. Each API key is limited to 60 requests per minute.

Ingestion, quality control and model jobs run as scheduled containers. The full database is backed up nightly, and backups are kept for 35 days.
