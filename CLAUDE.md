# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

HSE Finam Hackathon project for financial forecasting based on stock market data and news.

## Data Structure

The repository contains historical financial data in `forecast_data/`:

### Stock Market Data (candles.csv, candles_2.csv)
- OHLCV (Open, High, Low, Close, Volume) candle data
- Columns: `open`, `close`, `high`, `low`, `volume`, `begin`, `ticker`
- Time series data starting from 2020-06-19
- Multiple Russian stock tickers (e.g., AFLT - Aeroflot)

### News Data (news.csv, news_2.csv)
- Financial news articles with timestamps
- Columns: index, `publish_date`, `title`, `publication`
- News articles related to Russian companies and markets
- Dates aligned with stock data timeframe (2020+)

## Data Characteristics

- Large datasets: news.csv is ~92MB, candles.csv is ~1.2MB
- Russian language content in news articles
- Focus on Russian stock market companies (Gazprom, MMK, MTS, etc.)
- Data suitable for time series forecasting and sentiment analysis tasks
