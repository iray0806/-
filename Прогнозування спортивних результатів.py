"""
Required:
pip install pandas numpy tensorflow scikit-learn matplotlib requests python-dotenv plotly pywebview
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import sqlite3
import pandas as pd
import numpy as np
import requests
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Conv1D, MaxPooling1D, Flatten
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
import warnings
import os
from dotenv import load_dotenv
from threading import Thread
import logging
import sys
import json
import webview  # Import pywebview

# Налаштування логування
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('football_predictor.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Завантаження змінних середовища
load_dotenv()
warnings.filterwarnings('ignore')

def calculate_league_coefficient(league_tier):
    """Розрахунок коефіцієнту ліги"""
    return round(1 / league_tier, 2)

class DateValidator:
    """Клас для валідації дат та сезонів"""
    @staticmethod
    def validate_date(date):
        """Перевірка та конвертація дати"""
        try:
            if isinstance(date, str):
                return pd.to_datetime(date)
            elif isinstance(date, pd.Series):
                return pd.to_datetime(date.iloc[0])
            elif isinstance(date, pd.Timestamp):
                return date
            else:
                return pd.to_datetime(date)
        except Exception as e:
            print(f"Помилка валідації дати: {date}, тип: {type(date)}")
            raise

    @staticmethod
    def get_next_year(date):
        """Безпечне отримання наступного року"""
        try:
            date = DateValidator.validate_date(date)
            return int(date.year) + 1
        except Exception as e:
            print(f"Помилка отримання наступного року для дати: {date}")
            raise

    @staticmethod
    def format_season(first_year):
        """Форматування сезону"""
        try:
            # Переконуємося, що у нас int
            first_year = int(first_year)
            # Отримуємо наступний рік
            second_year = first_year + 1
            # Беремо останні дві цифри
            second_year_short = str(second_year)[-2:]
            # Формуємо рядок сезону
            return str(first_year) + '/' + str(second_year_short)
        except Exception as e:
            print(f"Помилка форматування сезону: {first_year}, тип: {type(first_year)}")
            raise

class DebugPrinter:
    """Клас для виведення відладочної інформації"""
    def __init__(self, text_widget):
        self.text_widget = text_widget
    
    def print(self, message):
        """Додавання повідомлення у віджет та логи"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        formatted_message = f"[{timestamp}] {message}\n"
        
        self.text_widget.configure(state='normal')
        self.text_widget.insert(tk.END, formatted_message)
        self.text_widget.see(tk.END)
        self.text_widget.configure(state='disabled')
        
        logger.debug(message)

class FootballDataCollector:
    def __init__(self, api_key, debug_printer):
        """Ініціалізація колектора даних"""
        self.api_key = api_key
        self.base_url = "https://api.football-data.org/v4"
        self.headers = {"X-Auth-Token": api_key}
        self.debug_printer = debug_printer
        self.create_database()
    
    def create_database(self):
        """Створення бази даних"""
        try:
            with sqlite3.connect('football_data.db') as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS matches (
                        match_id TEXT PRIMARY KEY,
                        date TEXT,
                        team_id INTEGER,
                        team_name TEXT,
                        league_tier INTEGER,
                        competition_id INTEGER,
                        competition_name TEXT,
                        home_away TEXT,
                        goals_for INTEGER,
                        goals_against INTEGER,
                        result TEXT
                    )
                ''')
                conn.commit()
            self.debug_printer.print("База даних створена успішно")
        except Exception as e:
            self.debug_printer.print(f"Помилка створення БД: {str(e)}")
            raise

    @staticmethod
    def normalize_team_name(name):
        """Нормалізація назви команди для пошуку"""
        import unicodedata
        import re
        
        name = unicodedata.normalize('NFKD', name).encode('ASCII', 'ignore').decode('ASCII')
        name = re.sub(r'[^a-zA-Z0-9\s]', '', name).lower().strip()
        return ' '.join(name.split())

    @staticmethod
    def team_names_match(search_term, team_name):
        """Перевірка відповідності назв команд"""
        if not team_name:
            return False
        search_norm = FootballDataCollector.normalize_team_name(search_term)
        team_norm = FootballDataCollector.normalize_team_name(team_name)
        
        if search_norm == team_norm:
            return True
            
        words = search_norm.split()
        team_words = team_norm.split()
        
        return any(word in team_words for word in words)

    def get_team_matches(self, team_code):
        """Отримання матчів команди"""
        self.debug_printer.print(f"Пошук команди за кодом: {team_code}")
        
        try:
            competitions_url = f"{self.base_url}/competitions"
            self.debug_printer.print("Отримання списку змагань...")
            competitions_response = requests.get(competitions_url, headers=self.headers)
            
            if competitions_response.status_code != 200:
                raise Exception(f"Помилка отримання списку змагань: {competitions_response.status_code}")
                
            competitions = competitions_response.json().get('competitions', [])
            self.debug_printer.print(f"Знайдено {len(competitions)} змагань")
            
            for competition in competitions:
                self.debug_printer.print(f"Пошук в {competition['name']}...")
                
                teams_url = f"{self.base_url}/competitions/{competition['id']}/teams"
                teams_response = requests.get(teams_url, headers=self.headers)
                
                if teams_response.status_code == 200:
                    teams = teams_response.json().get('teams', [])
                    
                    for team in teams:
                        if (team.get('tla') == team_code.upper() or 
                            self.team_names_match(team_code, team.get('name')) or 
                            self.team_names_match(team_code, team.get('shortName'))):
                            
                            team_id = team['id']
                            self.debug_printer.print(
                                f"Знайдено команду: {team['name']} (ID: {team_id}) "
                                f"в лізі {competition['name']}"
                            )
                            return self.get_matches_by_team_id(team_id)
            
            raise ValueError(f"Команду з кодом {team_code} не знайдено в жодній лізі")
                
        except requests.exceptions.RequestException as e:
            error_msg = f"Помилка мережі: {str(e)}"
            self.debug_printer.print(error_msg)
            raise

    def get_matches_by_team_id(self, team_id, seasons=5):
        """Отримання матчів за ID команди"""
        matches = []
        current_date = datetime.now()
        
        for i in range(seasons):
            season_start = current_date.replace(year=current_date.year - i, month=7, day=1)
            season_end = season_start.replace(year=season_start.year + 1, month=6, day=30)
            
            url = f"{self.base_url}/teams/{team_id}/matches"
            params = {
                'dateFrom': season_start.strftime('%Y-%m-%d'),
                'dateTo': season_end.strftime('%Y-%m-%d')
            }
            
            self.debug_printer.print(f"Завантаження сезону {season_start.year}/{season_end.year}")
            
            try:
                response = requests.get(url, headers=self.headers, params=params)
                if response.status_code == 200:
                    season_matches = response.json().get('matches', [])
                    self.debug_printer.print(f"Отримано {len(season_matches)} матчів")
                    matches.extend(season_matches)
                else:
                    self.debug_printer.print(f"Помилка отримання матчів: {response.status_code}")
                    
            except Exception as e:
                self.debug_printer.print(f"Помилка: {str(e)}")
                continue
        
        if not matches:
            self.debug_printer.print("Матчів не знайдено")
            return pd.DataFrame()
        
        processed_matches = []
        for match in matches:
            match_data = self.process_match_data(match, team_id)
            if match_data:
                processed_matches.append(match_data)
        
        df = pd.DataFrame(processed_matches)
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
        
        return df

    def process_match_data(self, match, team_id):
        """Обробка даних матчу"""
        try:
            match_date = pd.to_datetime(match['utcDate'])
            
            home_team = match['homeTeam']
            away_team = match['awayTeam']
            score = match.get('score', {}).get('fullTime', {})
            
            if not score:
                return None
                
            is_home = home_team['id'] == team_id
            team_name = home_team['name'] if is_home else away_team['name']
            goals_for = score.get('home' if is_home else 'away')
            goals_against = score.get('away' if is_home else 'home')
            
            if goals_for is None or goals_against is None:
                return None
            
            return {
                'match_id': str(match['id']),
                'date': match_date,
                'team_id': team_id,
                'team_name': team_name,
                'league_tier': self.get_league_tier(match['competition']['id']),
                'competition_id': match['competition']['id'],
                'competition_name': match['competition']['name'],
                'home_away': 'home' if is_home else 'away',
                'goals_for': goals_for,
                'goals_against': goals_against,
                'result': self.get_match_result(goals_for, goals_against)
            }
                
        except Exception as e:
            self.debug_printer.print(f"Помилка обробки матчу: {str(e)}")
            return None

    def get_league_tier(self, competition_id):
        """Визначення рівня ліги"""
        top_leagues = [2021, 2014, 2019, 2002, 2015]  # IDs топ-5 ліг
        return 1 if competition_id in top_leagues else 2

    @staticmethod
    def get_match_result(goals_for, goals_against):
        """Визначення результату матчу"""
        if goals_for > goals_against:
            return 'W'
        elif goals_for < goals_against:
            return 'L'
        return 'D'

class PredictionModel:
    def __init__(self, debug_printer):
        self.lstm_model = None
        self.dense_model = None
        self.cnn_model = None
        self.scaler = StandardScaler()
        self.debug_printer = debug_printer
        self.sequence_length = 10
        
        self.debug_printer.print("Ініціалізація моделей прогнозування")

    def prepare_features(self, df):
        """Підготовка ознак для моделей"""
        self.debug_printer.print("Підготовка ознак для моделей")
        
        numeric_columns = ['goals_for', 'goals_against']
        features = df[numeric_columns].values
        
        # Нормалізація з урахуванням рівня ліги
        league_coefficients = df['league_tier'].apply(calculate_league_coefficient).values
        features = features * league_coefficients.reshape(-1, 1)
        
        return self.scaler.fit_transform(features)

    def create_sequences(self, X, y, sequence_length):
        """Створення послідовностей для тренування"""
        X_seq, y_seq = [], []
        for i in range(len(X) - sequence_length):
            X_seq.append(X[i:(i + sequence_length)])
            y_seq.append(y[i + sequence_length])
        return np.array(X_seq), np.array(y_seq)

    def train_models(self, df, sequence_length=10):
        """Тренування всіх моделей"""
        self.debug_printer.print("Початок тренування моделей")
        self.sequence_length = sequence_length
        
        try:
            # Підготовка даних
            X = self.prepare_features(df)
            y = df['goals_for'].values
            
            X_seq, y_seq = self.create_sequences(X, y, sequence_length)
            
            self.debug_printer.print(f"Розмір послідовностей - X: {X_seq.shape}, y: {y_seq.shape}")
            
            if len(X_seq) == 0:
                raise ValueError("Недостатньо даних для тренування")
            
            input_shape = (sequence_length, X.shape[1])
            self.create_models(input_shape)
            
            lstm_input = X_seq
            dense_input = X_seq.reshape(X_seq.shape[0], -1)
            cnn_input = X_seq
            
            self.debug_printer.print("Тренування LSTM моделі...")
            self.lstm_model.fit(lstm_input, y_seq, epochs=50, batch_size=32, verbose=0)
            
            self.debug_printer.print("Тренування Dense моделі...")
            self.dense_model.fit(dense_input, y_seq, epochs=50, batch_size=32, verbose=0)
            
            self.debug_printer.print("Тренування CNN моделі...")
            self.cnn_model.fit(cnn_input, y_seq, epochs=50, batch_size=32, verbose=0)
            
            self.debug_printer.print("Тренування завершено успішно")
            
        except Exception as e:
            error_msg = f"Помилка під час тренування: {str(e)}"
            self.debug_printer.print(error_msg)
            raise ValueError(error_msg)

    def create_models(self, input_shape):
        """Створення всіх моделей"""
        sequence_length, n_features = input_shape
        self.debug_printer.print(f"Створення моделей з input_shape={input_shape}")
        
        # LSTM модель
        self.lstm_model = Sequential([
            LSTM(64, return_sequences=True, input_shape=input_shape),
            LSTM(32, return_sequences=True),
            LSTM(16),
            Dense(1)
        ])
        self.lstm_model.compile(optimizer='adam', loss='mse')
        
        # Dense модель
        self.dense_model = Sequential([
            Dense(64, activation='relu', input_shape=(sequence_length * n_features,)),
            Dense(32, activation='relu'),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        self.dense_model.compile(optimizer='adam', loss='mse')
        
        # CNN модель
        self.cnn_model = Sequential([
            Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
            MaxPooling1D(pool_size=2),
            Flatten(),
            Dense(32, activation='relu'),
            Dense(1)
        ])
        self.cnn_model.compile(optimizer='adam', loss='mse')
        
        self.debug_printer.print("Моделі створено успішно")

    def predict(self, features):
        """Прогнозування одного значення"""
        try:
            features_scaled = self.scaler.transform(features)
            sequence_length = self.sequence_length
            
            if len(features_scaled) < sequence_length:
                self.debug_printer.print(f"Попередження: доступно тільки {len(features_scaled)} матчів")
                pad_size = sequence_length - len(features_scaled)
                pad_data = np.mean(features_scaled, axis=0)
                features_scaled = np.vstack([np.tile(pad_data, (pad_size, 1)), features_scaled])
            
            X_seq = features_scaled[-sequence_length:].reshape(1, sequence_length, -1)
            X_dense = X_seq.reshape(1, -1)
            
            lstm_pred = self.lstm_model.predict(X_seq, verbose=0)
            dense_pred = self.dense_model.predict(X_dense, verbose=0)
            cnn_pred = self.cnn_model.predict(X_seq, verbose=0)
            
            ensemble_pred = (lstm_pred + dense_pred + cnn_pred) / 3
            return ensemble_pred
            
        except Exception as e:
            self.debug_printer.print(f"Помилка прогнозування: {str(e)}")
            raise

    def predict_sequence(self, features, n_steps):
        """Прогнозування послідовності значень"""
        predictions = []
        current_features = features.copy()
        
        try:
            for _ in range(n_steps):
                next_pred = self.predict(current_features)[0][0]
                predictions.append(next_pred)
                new_row = np.array([[next_pred, predictions[-1]]])
                current_features = np.vstack([current_features[1:], new_row])
            
            return np.array(predictions)
            
        except Exception as e:
            self.debug_printer.print(f"Помилка прогнозування послідовності: {str(e)}")
            raise

    def calculate_prediction_interval(self, prediction, confidence=0.95):
        """Розрахунок інтервалу передбачення"""
        std = np.std([prediction[0][0], prediction[1][0], prediction[2][0]])
        z_score = 1.96  # для 95% довірчого інтервалу
        interval = z_score * std
        mean_pred = np.mean([prediction[0][0], prediction[1][0], prediction[2][0]])
        
        return mean_pred - interval, mean_pred + interval

class FootballPredictor(tk.Tk):
    def __init__(self, api_key):
        super().__init__()
        
        self.title("Футбольний прогнозист")
        self.geometry("1400x900")

        # Створення головного контейнера
        self.main_container = ttk.PanedWindow(self, orient='horizontal')
        self.main_container.pack(fill='both', expand=True, padx=5, pady=5)

        # Створення правої та лівої панелей
        self.create_left_panel()
        self.create_right_panel()
        
        # Ініціалізація
        self.debug_printer.print("Ініціалізація програми...")
        self.data_collector = FootballDataCollector(api_key, self.debug_printer)
        self.model = PredictionModel(self.debug_printer)
        self.data = None
        self.temp_files = []  # Для відстеження тимчасових файлів
        
        self.debug_printer.print("Програму ініціалізовано успішно")

    def create_left_panel(self):
        """Створення лівої панелі інтерфейсу"""
        left_panel = ttk.Frame(self.main_container)
        self.main_container.add(left_panel, weight=1)
        
        # Панель введення
        input_frame = ttk.LabelFrame(left_panel, text="Введення даних")
        input_frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Label(input_frame, text="Код команди:").pack(side='left', padx=5)
        self.team_code_entry = ttk.Entry(input_frame, width=5)
        self.team_code_entry.pack(side='left', padx=5)
        
        help_button = ttk.Button(input_frame, text="?", width=3, 
                               command=self.show_team_codes)
        help_button.pack(side='left', padx=2)
        
        self.load_button = ttk.Button(input_frame, text="Завантажити дані",
                                    command=self.start_loading)
        self.load_button.pack(side='left', padx=5)
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(input_frame, length=200, 
                                          mode='determinate',
                                          variable=self.progress_var)
        self.progress_bar.pack(side='left', padx=5)
        
        # Панель прогнозів
        self.prediction_frame = ttk.LabelFrame(left_panel, text="Прогнози та аналіз")
        self.prediction_frame.pack(fill='both', expand=True, padx=5, pady=5)

    def create_right_panel(self):
        """Створення правої панелі інтерфейсу"""
        right_panel = ttk.Frame(self.main_container)
        self.main_container.add(right_panel, weight=2)
        
        # Створюємо вкладки для графіків
        self.plots_notebook = ttk.Notebook(right_panel)
        self.plots_notebook.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Вкладки для різних графіків
        self.goals_frame = ttk.Frame(self.plots_notebook)
        self.points_frame = ttk.Frame(self.plots_notebook)
        self.form_frame = ttk.Frame(self.plots_notebook)
        
        self.plots_notebook.add(self.goals_frame, text='Голи')
        self.plots_notebook.add(self.points_frame, text='Очки')
        self.plots_notebook.add(self.form_frame, text='Форма')
        
        # Панель відладки
        debug_frame = ttk.LabelFrame(right_panel, text="Відладкова інформація")
        debug_frame.pack(fill='x', padx=5, pady=5)
        
        self.debug_text = scrolledtext.ScrolledText(debug_frame, height=8)
        self.debug_text.pack(fill='x', padx=5, pady=5)
        self.debug_printer = DebugPrinter(self.debug_text)

    def show_team_codes(self):
        """Показ підказки з кодами команд"""
        codes = """Приклади кодів команд:

Англія (EPL):
LIV - Liverpool
MUN - Manchester United
ARS - Arsenal
CHE - Chelsea
MCI - Manchester City
TOT - Tottenham

Іспанія (La Liga):
MAD - Real Madrid
BAR - Barcelona
ATM - Atletico Madrid
VAL - Valencia

Німеччина (Bundesliga):
BAY - Bayern Munich
BVB - Borussia Dortmund
RBL - RB Leipzig
SCF - SC Freiburg

Італія (Serie A):
JUV - Juventus
MIL - AC Milan
INT - Inter
ROM - Roma
NAP - Napoli

Франція (Ligue 1):
PSG - Paris Saint-Germain
OL  - Olympique Lyonnais
OM  - Olympique Marseille
MON - Monaco"""
        
        messagebox.showinfo("Коди команд", codes)

    def start_loading(self):
        """Початок завантаження даних в окремому потоці"""
        team_code = self.team_code_entry.get().strip()
        if not team_code:
            messagebox.showerror("Помилка", "Введіть код команди")
            return
        
        self.load_button.configure(state='disabled')
        self.progress_var.set(0)
        
        try:
            thread = Thread(target=self.load_data, args=(team_code,))
            thread.daemon = True
            thread.start()
        except Exception as e:
            self.debug_printer.print(f"Помилка запуску завантаження: {str(e)}")
            self.load_button.configure(state='normal')
            messagebox.showerror("Помилка", str(e))

    def load_data(self, team_code):
        """Завантаження даних команди"""
        try:
            self.debug_printer.print(f"Початок завантаження даних для {team_code}")
            self.progress_var.set(20)
            
            self.data = self.data_collector.get_team_matches(team_code)
            if self.data.empty:
                raise ValueError("Дані не знайдено")
                
            self.debug_printer.print("Обробка дат...")
            self.data['date'] = pd.to_datetime(self.data['date'])
            
            self.debug_printer.print("Формування сезонів...")
            self.data['season'] = self.data['date'].apply(lambda x: 
                DateValidator.format_season(x.year if x.month >= 7 else x.year - 1))
                
            self.debug_printer.print(f"Унікальні сезони: {self.data['season'].unique()}")
            
            self.progress_var.set(50)
            
            self.debug_printer.print("Тренування моделей...")
            self.model.train_models(self.data)
            self.progress_var.set(80)
            
            self.update_plots()
            self.show_predictions()
            self.progress_var.set(100)
            
        except Exception as e:
            self.debug_printer.print(f"Помилка: {str(e)}")
            messagebox.showerror("Помилка", str(e))
        finally:
            self.load_button.configure(state='normal')

    def show_predictions(self):
        """Відображення прогнозів"""
        try:
            for widget in self.prediction_frame.winfo_children():
                widget.destroy()
            
            if self.data is None or self.data.empty:
                return

            current_season = self.get_season_str(self.data['date'].max())
            current_year = int(current_season.split('/')[0])
            next_season = f"{current_year+1}/{str(current_year+2)[-2:]}"
            
            season_data = self.data[self.data['season'] == current_season]
            matches_left = 38 - len(season_data)

            # Створюємо текстовий віджет для прогнозів
            pred_text = scrolledtext.ScrolledText(self.prediction_frame, height=15, width=40)
            pred_text.pack(fill='both', expand=True, padx=5, pady=5)

            # Прогноз на наступний матч
            features = self.data[['goals_for', 'goals_against']].values
            next_match_pred = self.model.predict(features)[0][0]
            recent_form = season_data['result'].map({'W': 3, 'D': 1, 'L': 0}).tail(5).mean()
            
            pred_text.insert(tk.END, "=== ПРОГНОЗ НА НАСТУПНИЙ МАТЧ ===\n\n")
            pred_text.insert(tk.END, f"Очікувані голи: {next_match_pred:.1f}\n")
            pred_text.insert(tk.END, f"Поточна форма: {recent_form:.2f} очка за матч\n")
            win_prob = min(max(recent_form / 3.0, 0.1), 0.9) * 100
            pred_text.insert(tk.END, f"Ймовірність перемоги: {win_prob:.1f}%\n\n")

            # Прогноз до кінця поточного сезону
            if matches_left > 0:
                goals_predictions = self.model.predict_sequence(features, matches_left)
                expected_points = matches_left * recent_form
                current_points = season_data['result'].map({'W': 3, 'D': 1, 'L': 0}).sum()
                
                pred_text.insert(tk.END, f"=== ПРОГНОЗ ДО КІНЦЯ СЕЗОНУ {current_season} ===\n\n")
                pred_text.insert(tk.END, f"Залишилось матчів: {matches_left}\n")
                pred_text.insert(tk.END, f"Очікувані голи (середнє): {np.mean(goals_predictions):.2f}\n")
                pred_text.insert(tk.END, f"Очікувані очки: +{expected_points:.1f}\n")
                pred_text.insert(tk.END, f"Поточні очки: {current_points}\n")
                pred_text.insert(tk.END, f"Прогноз підсумкових очків: {current_points + expected_points:.1f}\n")
                
                # Прогноз місця в таблиці
                total_points = current_points + expected_points
                if total_points >= 80:
                    position = "1-4 (Ліга Чемпіонів)"
                elif total_points >= 65:
                    position = "5-7 (Єврокубки)"
                elif total_points >= 45:
                    position = "8-12 (Середина таблиці)"
                elif total_points >= 35:
                    position = "13-16 (Нижня частина)"
                else:
                    position = "17-20 (Зона вильоту)"
                pred_text.insert(tk.END, f"Очікуване місце: {position}\n\n")

            # Прогноз на наступний сезон
            pred_text.insert(tk.END, f"=== ПРОГНОЗ НА СЕЗОН {next_season} ===\n\n")
            
            next_season_goals = self.model.predict_sequence(features[-10:], 38)
            avg_goals = np.mean(next_season_goals)
            
            # Використовуємо середню форму за останні 10 матчів з невеликим покращенням
            last_10_matches_form = self.data['result'].map({'W': 3, 'D': 1, 'L': 0}).tail(10).mean()
            expected_points_per_game = min(max(last_10_matches_form * 1.1, 1.0), 2.5)
            expected_season_points = expected_points_per_game * 38
            
            pred_text.insert(tk.END, f"Очікувані голи за матч: {avg_goals:.2f}\n")
            pred_text.insert(tk.END, f"Очікувані очки за матч: {expected_points_per_game:.2f}\n")
            pred_text.insert(tk.END, f"Прогноз очок за сезон: {expected_season_points:.1f}\n")
            
            # Прогноз результату наступного сезону
            if expected_season_points >= 80:
                outcome = "Боротьба за чемпіонство"
            elif expected_season_points >= 65:
                outcome = "Боротьба за Лігу Чемпіонів"
            elif expected_season_points >= 55:
                outcome = "Боротьба за Єврокубки"
            elif expected_season_points >= 45:
                outcome = "Середина таблиці"
            elif expected_season_points >= 35:
                outcome = "Боротьба за виживання"
            else:
                outcome = "Висока ймовірність вильоту"
                
            pred_text.insert(tk.END, f"Очікуваний результат: {outcome}\n")
            
            pred_text.configure(state='disabled')
            
        except Exception as e:
            self.debug_printer.print(f"Помилка в show_predictions: {str(e)}")
            raise

    

    def generate_future_dates(self, start_date, periods):
        """Генерація майбутніх дат"""
        if isinstance(start_date, pd.Timestamp):
            # Конвертуємо в datetime і додаємо timedelta для кожної дати
            base_date = start_date.to_pydatetime()
            return pd.date_range(start=base_date, periods=periods+1, freq='7D')[1:]
        else:
            # Якщо дата вже в форматі datetime
            return pd.date_range(start=start_date, periods=periods+1, freq='7D')[1:]

    def get_season_start_year(self, date):
        """Отримання року початку сезону"""
        if isinstance(date, str):
            date = pd.to_datetime(date)
        month = date.month
        year = date.year
        return year if month >= 8 else year - 1

    def get_season_str(self, date):
        """Отримання строки сезону"""
        start_year = self.get_season_start_year(date)
        return f"{start_year}/{str(start_year + 1)[-2:]}"

    def generate_season_dates(self, season_start_year, total_matches=38):
        """Генерація дат для сезону"""
        season_start = pd.Timestamp(f"{season_start_year}-08-01")
        return pd.date_range(start=season_start, periods=total_matches, freq='7D')

    def update_plots(self):
        """Оновлення всіх графіків"""
        try:
            self.debug_printer.print("Початок оновлення графіків...")
            self.debug_printer.print(f"Дата останнього матчу: {self.data['date'].max()}")
            
            self.update_goals_plot()
            self.update_points_plot()
            self.update_form_plot()
            
        except Exception as e:
            self.debug_printer.print(f"Помилка оновлення графіків: {str(e)}")
            raise

    def update_goals_plot(self):
        """Оновлення графіка голів з реалістичними прогнозами"""
        try:
            for widget in self.goals_frame.winfo_children():
                widget.destroy()

            fig = go.Figure()

            # Групуємо дані по сезонах
            self.data['season'] = self.data['date'].apply(self.get_season_str)
            seasons = self.data['season'].unique()

            # Показуємо історичні дані по сезонах
            for season in seasons:
                season_data = self.data[self.data['season'] == season]
                
                fig.add_trace(go.Scatter(
                    x=season_data['date'],
                    y=season_data['goals_for'],
                    name=f'Забиті голи {season}',
                    mode='lines+markers',
                    line=dict(color='blue'),
                    marker=dict(size=6)
                ))
                
                fig.add_trace(go.Scatter(
                    x=season_data['date'],
                    y=season_data['goals_against'],
                    name=f'Пропущені голи {season}',
                    mode='lines+markers',
                    line=dict(color='red'),
                    marker=dict(size=6)
                ))

            # Прогнози
            current_season = self.get_season_str(self.data['date'].max())
            current_season_data = self.data[self.data['season'] == current_season]
            matches_left = 38 - len(current_season_data)
            
            if matches_left > 0:
                last_date = self.data['date'].max()
                
                # Базова статистика поточного сезону
                avg_goals_for = current_season_data['goals_for'].mean()
                avg_goals_against = current_season_data['goals_against'].mean()
                
                future_dates = pd.date_range(start=last_date, periods=matches_left + 1, freq='7D')[1:]
                
                # Генеруємо типові значення для голів
                typical_scores = [0, 1, 1, 1, 1, 1, 2, 2, 2, 3]  # Типовий розподіл голів
                
                # Прогноз на поточний сезон
                scored_goals = []
                conceded_goals = []
                
                for _ in range(matches_left):
                    # Вибираємо випадкове значення з типових результатів
                    # з корекцією на середню результативність команди
                    scored = np.random.choice(typical_scores)
                    if avg_goals_for > 2:  # Якщо команда результативна
                        scored = max(1, scored)
                    
                    conceded = np.random.choice(typical_scores)
                    if avg_goals_against < 1:  # Якщо команда надійна в захисті
                        conceded = min(2, conceded)
                    
                    scored_goals.append(scored)
                    conceded_goals.append(conceded)
                
                fig.add_trace(go.Scatter(
                    x=future_dates,
                    y=scored_goals,
                    name=f'Прогноз забитих {current_season}',
                    line=dict(dash='dash', color='blue')
                ))
                
                fig.add_trace(go.Scatter(
                    x=future_dates,
                    y=conceded_goals,
                    name=f'Прогноз пропущених {current_season}',
                    line=dict(dash='dash', color='red')
                ))

                # Прогноз на наступний сезон
                next_season_year = int(current_season.split('/')[0]) + 1
                next_season_dates = self.generate_season_dates(next_season_year)
                
                next_season_scored = []
                next_season_conceded = []
                
                for _ in range(38):
                    scored = np.random.choice(typical_scores)
                    if avg_goals_for > 2:
                        scored = max(1, scored)
                    
                    conceded = np.random.choice(typical_scores)
                    if avg_goals_against < 1:
                        conceded = min(2, conceded)
                    
                    next_season_scored.append(scored)
                    next_season_conceded.append(conceded)

                fig.add_trace(go.Scatter(
                    x=next_season_dates,
                    y=next_season_scored,
                    name=f'Прогноз забитих {next_season_year}/{str(next_season_year + 1)[-2:]}',
                    line=dict(dash='dot', color='blue')
                ))
                
                fig.add_trace(go.Scatter(
                    x=next_season_dates,
                    y=next_season_conceded,
                    name=f'Прогноз пропущених {next_season_year}/{str(next_season_year + 1)[-2:]}',
                    line=dict(dash='dot', color='red')
                ))

            fig.update_layout(
                title='Голи по сезонах',
                xaxis_title='Дата',
                yaxis_title='Кількість голів',
                hovermode='x unified',
                showlegend=True,
                xaxis=dict(
                    tickformat='%Y-%m',
                    dtick='M1',
                    tickangle=45
                ),
                yaxis=dict(range=[0, 4]),
                plot_bgcolor='white'
            )

            self.add_plotly_chart(fig, self.goals_frame)

        except Exception as e:
            self.debug_printer.print(f"Помилка в update_goals_plot: {str(e)}")
            raise

    def update_points_plot(self):
        """Оновлення графіка очок з розділенням по сезонах"""
        try:
            for widget in self.points_frame.winfo_children():
                widget.destroy()

            fig = go.Figure()

            # Групуємо дані по сезонах
            self.data['season'] = self.data['date'].apply(self.get_season_str)
            self.data['points'] = self.data['result'].map({'W': 3, 'D': 1, 'L': 0})
            
            # Показуємо очки окремо для кожного сезону
            for season in self.data['season'].unique():
                season_data = self.data[self.data['season'] == season].copy()
                season_data = season_data.sort_values('date')
                season_points = season_data['points'].cumsum()
                
                fig.add_trace(go.Scatter(
                    x=season_data['date'],
                    y=season_points,
                    name=f'Очки {season}',
                    mode='lines+markers',
                    line=dict(width=2),
                    marker=dict(size=8)
                ))

            # Прогнози
            current_season = self.get_season_str(self.data['date'].max())
            current_season_data = self.data[self.data['season'] == current_season]
            matches_left = 38 - len(current_season_data)
            
            if matches_left > 0:
                last_date = self.data['date'].max()
                last_points = current_season_data['points'].cumsum().iloc[-1]
                historical_form = current_season_data['points'].rolling(window=5).mean()
                recent_form = historical_form.iloc[-1] if not historical_form.empty else 1.5
                
                # Прогноз на поточний сезон з урахуванням форми та волатильності
                future_dates = pd.date_range(start=last_date, periods=matches_left + 1, freq='7D')[1:]
                points_prediction = [last_points]
                current_form = recent_form
                
                for match in range(matches_left):
                    # Розраховуємо ймовірності результатів на основі форми
                    win_prob = min(max(current_form / 3, 0.2), 0.6)
                    draw_prob = 0.3
                    lose_prob = 1 - win_prob - draw_prob
                    
                    # Симулюємо результат
                    result = np.random.choice([3, 1, 0], p=[win_prob, draw_prob, lose_prob])
                    points_prediction.append(points_prediction[-1] + result)
                    
                    # Оновлюємо форму
                    form_change = (result - current_form) * 0.2
                    current_form = max(min(current_form + form_change, 2.5), 0.5)
                
                fig.add_trace(go.Scatter(
                    x=future_dates,
                    y=points_prediction[1:],
                    name=f'Прогноз {current_season}',
                    line=dict(dash='dash', width=2)
                ))

            # Прогноз на наступний сезон
            next_season_year = int(current_season.split('/')[0]) + 1
            next_season_dates = self.generate_season_dates(next_season_year)
            
            # Аналіз історичних даних для сезонності
            historical_points = []
            for season in self.data['season'].unique()[:-1]:  # Виключаємо поточний сезон
                season_data = self.data[self.data['season'] == season]
                season_points = season_data['points'].cumsum().iloc[-1]
                historical_points.append(season_points)
            
            # Генерація прогнозу з урахуванням історичних трендів
            historical_avg = np.mean(historical_points) if historical_points else 60
            expected_improvement = 1.1  # Очікуване покращення
            
            next_season_points = [0]
            current_form = recent_form * expected_improvement
            
            for match in range(38):
                # Сезонний фактор (краща форма в середині сезону)
                season_factor = 1 + 0.1 * np.sin(np.pi * match / 19)
                
                # Розраховуємо ймовірності з урахуванням сезонності
                win_prob = min(max(current_form * season_factor / 3, 0.2), 0.6)
                draw_prob = 0.3
                lose_prob = 1 - win_prob - draw_prob
                
                # Симулюємо результат
                result = np.random.choice([3, 1, 0], p=[win_prob, draw_prob, lose_prob])
                next_season_points.append(next_season_points[-1] + result)
                
                # Оновлюємо форму
                form_change = (result - current_form) * 0.1
                current_form = max(min(current_form + form_change, 2.5), 0.5)
            
            fig.add_trace(go.Scatter(
                x=next_season_dates,
                y=next_season_points[1:],
                name=f'Прогноз {next_season_year}/{str(next_season_year + 1)[-2:]}',
                line=dict(dash='dot', width=2)
            ))

            fig.update_layout(
                title='Набрані очки по сезонах',
                xaxis_title='Дата',
                yaxis_title='Очки',
                hovermode='x unified',
                showlegend=True,
                xaxis=dict(
                    tickformat='%Y-%m',
                    dtick='M1',
                    tickangle=45
                ),
                plot_bgcolor='white'
            )

            self.add_plotly_chart(fig, self.points_frame)

        except Exception as e:
            self.debug_printer.print(f"Помилка в update_points_plot: {str(e)}")
            raise

    def update_form_plot(self):
        """Оновлення графіка форми з реалістичними прогнозами"""
        try:
            for widget in self.form_frame.winfo_children():
                widget.destroy()

            fig = go.Figure()

            # Групуємо дані по сезонах
            self.data['season'] = self.data['date'].apply(self.get_season_str)
            self.data['points'] = self.data['result'].map({'W': 3, 'D': 1, 'L': 0})
            
            # Показуємо історичні дані
            for season in self.data['season'].unique():
                season_data = self.data[self.data['season'] == season].copy()
                season_data = season_data.sort_values('date')
                season_data['form'] = season_data['points'].rolling(window=5, min_periods=1).mean()
                
                fig.add_trace(go.Scatter(
                    x=season_data['date'],
                    y=season_data['form'],
                    name=f'Форма {season}',
                    mode='lines+markers',
                    line=dict(width=2),
                    marker=dict(size=6)
                ))

            # Прогнози
            current_season = self.get_season_str(self.data['date'].max())
            current_season_data = self.data[self.data['season'] == current_season]
            matches_left = 38 - len(current_season_data)

            if matches_left > 0:
                last_date = self.data['date'].max()
                
                # Поточна форма команди
                current_form = current_season_data['points'].tail(5).mean()
                if pd.isna(current_form):
                    current_form = 1.5

                # Прогноз на залишок поточного сезону
                future_dates = pd.date_range(start=last_date, periods=matches_left + 1, freq='7D')[1:]
                
                # Генеруємо 2-3 тренди для поточного сезону
                current_predictions = []
                form = current_form
                remaining_matches = matches_left
                
                while remaining_matches > 0:
                    # Визначаємо довжину тренду (від 4 до 8 матчів)
                    trend_length = min(remaining_matches, np.random.randint(4, 9))
                    
                    # Визначаємо цільове значення форми для тренду
                    if form < 1.5:  # Якщо форма низька, тенденція до покращення
                        target = np.random.uniform(1.5, 2.3)
                    elif form > 2.0:  # Якщо форма висока, можливий спад
                        target = np.random.uniform(1.3, 1.8)
                    else:  # Середня форма - випадковий напрямок
                        target = np.random.uniform(1.0, 2.5)
                    
                    # Генеруємо значення для тренду
                    for _ in range(trend_length):
                        # Рух до цільового значення з реалістичними коливаннями
                        pull_to_target = (target - form) * 0.3
                        random_change = np.random.uniform(-0.4, 0.4)
                        
                        form += pull_to_target + random_change
                        form = np.clip(form, 0.7, 2.7)
                        current_predictions.append(form)
                    
                    remaining_matches -= trend_length

                fig.add_trace(go.Scatter(
                    x=future_dates,
                    y=current_predictions,
                    name=f'Прогноз {current_season}',
                    line=dict(dash='dash', width=2)
                ))

                # Прогноз на наступний сезон (залишаємо як є)
                next_season_year = int(current_season.split('/')[0]) + 1
                next_season_dates = self.generate_season_dates(next_season_year)
                
                next_season_predictions = []
                form = 1.5  # Починаємо з середньої форми
                trend = 1
                trend_length = 0
                
                for _ in range(38):
                    if trend_length == 0:
                        trend = np.random.choice([-1, 1])
                        trend_length = np.random.randint(3, 7)
                        target = np.random.uniform(1.0, 2.5)
                    
                    pull_to_target = (target - form) * 0.3
                    random_change = np.random.uniform(-0.4, 0.4)
                    
                    form += pull_to_target + random_change
                    form = np.clip(form, 0.5, 2.8)
                    next_season_predictions.append(form)
                    
                    trend_length -= 1

                fig.add_trace(go.Scatter(
                    x=next_season_dates,
                    y=next_season_predictions,
                    name=f'Прогноз {next_season_year}/{str(next_season_year + 1)[-2:]}',
                    line=dict(dash='dot', width=2)
                ))

            # Додаємо орієнтири
            fig.add_hline(y=2, line_dash="dot", line_color="green", 
                         annotation_text="Хороша форма")
            fig.add_hline(y=1, line_dash="dot", line_color="orange", 
                         annotation_text="Середня форма")

            fig.update_layout(
                title='Форма команди по сезонах',
                xaxis_title='Дата',
                yaxis_title='Середні очки за матч (5 ігор)',
                hovermode='x unified',
                showlegend=True,
                yaxis=dict(range=[0, 3]),
                xaxis=dict(
                    tickformat='%Y-%m',
                    dtick='M1',
                    tickangle=45
                ),
                plot_bgcolor='white'
            )

            self.add_plotly_chart(fig, self.form_frame)

        except Exception as e:
            self.debug_printer.print(f"Помилка в update_form_plot: {str(e)}")
            raise

    def add_plotly_chart(self, fig, parent_frame):
        """Додавання графіка"""
        try:
            # Створюємо тимчасовий HTML файл
            import tempfile
            temp_file = tempfile.NamedTemporaryFile(
                delete=False,
                suffix='.html',
                mode='w',
                encoding='utf-8'
            )
            
            self.temp_files.append(temp_file.name)
            fig.write_html(temp_file.name, include_plotlyjs='cdn')
            
            chart_frame = ttk.Frame(parent_frame)
            chart_frame.pack(fill='both', expand=True)
            
            ttk.Button(
                chart_frame,
                text="Відкрити графік у браузері",
                command=lambda: self.open_in_browser(temp_file.name)
            ).pack(pady=5)
            
        except Exception as e:
            self.debug_printer.print(f"Помилка при створенні графіка: {str(e)}")
            messagebox.showerror("Помилка", "Не вдалося створити графік")
    
    def open_in_browser(self, file_path):
        """Безпечне відкриття файлу в браузері"""
        try:
            import webbrowser
            webbrowser.open(f'file://{os.path.abspath(file_path)}')
        except Exception as e:
            self.debug_printer.print(f"Помилка відкриття в браузері: {str(e)}")
            
    def cleanup(self):
        """Очищення тимчасових файлів"""
        for file_path in self.temp_files:
            try:
                os.remove(file_path)
            except Exception as e:
                self.debug_printer.print(f"Помилка видалення файлу {file_path}: {str(e)}")
        self.temp_files.clear()
        
    def get_football_season(self, date):
        """Визначення футбольного сезону для дати"""
        try:
            date = DateValidator.validate_date(date)
            year = int(date.year)
            first_year = year if date.month >= 7 else year - 1
            return DateValidator.format_season(first_year)
        except Exception as e:
            self.debug_printer.print(f"Помилка в get_football_season: {str(e)}")
            raise
            
    

def main():
    try:
        # Встановлюємо глобальні налаштування для plotly
        import plotly.io as pio
        pio.templates.default = "plotly_white"
        
        # Запускаємо програму
        api_key = os.getenv('FOOTBALL_API_KEY', 'd9dbe5c72a61471eb54e1f7ad9f54347')
        app = FootballPredictor(api_key)
        app.mainloop()
    except Exception as e:
        print(f"Критична помилка: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()