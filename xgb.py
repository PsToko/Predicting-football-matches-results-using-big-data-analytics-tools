import tkinter as tk
from tkinter import ttk
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, when, desc
import numpy as np
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils.class_weight import compute_class_weight
import time
from functools import lru_cache
import threading
from queue import Queue
from pyspark.sql import functions as F
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Initialize Spark
spark = SparkSession.builder \
    .appName("FootballPredictor") \
    .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
    .config("spark.executor.memory", "2g") \
    .config("spark.driver.memory", "2g") \
    .getOrCreate()

# Load data
try:
    matches_df = spark.read.option("header", True).csv("matches2.csv")
    matches_df = matches_df.withColumn("date", col("date"))
    matches_df.cache()
    print("Data loaded successfully")
except Exception as e:
    print(f"Error loading data: {e}")
    raise

# Get team list
try:
    teams = sorted([row.team for row in matches_df.select("team").distinct().collect()])
except:
    teams = []


class SafeFeatureExtractor:
    @lru_cache(maxsize=10000)
    def get_team_stats(self, team, n_matches=5):
        try:
            stats = matches_df.filter(col("team") == team) \
                .orderBy(desc("date")) \
                .limit(n_matches) \
                .agg(
                avg(when(col("result") == "W", 1).otherwise(0)).alias("win_pct"),
                avg(when(col("result") == "D", 1).otherwise(0)).alias("draw_pct"),
                avg(col("xg")).alias("avg_xg"),
                avg(col("xga")).alias("avg_xga"),
                avg("gf").alias("avg_goals")
            ).collect()[0]
            return {
                "win_pct": stats.win_pct * 100 if stats.win_pct else 0,
                "draw_pct": stats.draw_pct * 100 if stats.draw_pct else 0,
                "avg_xg": stats.avg_xg if stats.avg_xg else 0,
                "avg_xga": stats.avg_xga if stats.avg_xga else 0,
                "avg_goals": stats.avg_goals if stats.avg_goals else 0
            }
        except:
            return {"win_pct": 0, "draw_pct": 0, "avg_xg": 0, "avg_goals": 0}

    def get_formation_stats(self, team_name, is_opponent=False):
        try:
            # Get team's most used formation in last 20 matches
            team_matches = matches_df.filter(F.col("team") == team_name) \
                .orderBy(F.col("date").desc()) \
                .limit(20)

            most_used = team_matches.groupBy("formation").count() \
                .orderBy(F.col("count").desc()) \
                .first()

            if not most_used or not most_used["formation"]:
                return {"formation": "Unknown", "winrate": 0.0}

            most_used_formation = most_used["formation"]

            # Get opponent matches against this team (last 20)
            opponent_matches = matches_df.filter(F.col("opponent") == team_name) \
                .orderBy(F.col("date").desc()) \
                .limit(20)

            # Filter for matches where opponent faced this formation
            vs_formation_matches = opponent_matches.filter(F.col("formation") == most_used_formation)

            # Calculate winrate with reversed results (L=win, W=lose for opponents)
            winrate_expr = avg(
                when(F.col("result") == "L", 1)  # Opponent win (reversed)
                .when(F.col("result") == "W", 0)  # Opponent loss (reversed)
                .otherwise(0.5)  # Draw remains the same
            ).alias("winrate")

            winrate_row = vs_formation_matches.agg(winrate_expr).collect()[0]
            winrate = winrate_row["winrate"] if winrate_row["winrate"] is not None else 0

            return {
                "formation": most_used_formation,
                "winrate": winrate * 100,  # Convert to percentage
                "formation_count": most_used["count"],
                "total_matches": team_matches.count()
            }
        except Exception as e:
            print(f"Formation stats error for {team_name}: {e}")
            return {"formation": "Unknown", "winrate": 0.0}

    def get_power_ranking(self, team, recent_n=20):
        try:
            matches = matches_df.filter(col("team") == team) \
                .orderBy(col("date").desc()) \
                .limit(recent_n)

            stats = matches.withColumn("points", when(col("result") == "W", 3)
                                       .when(col("result") == "D", 1)
                                       .otherwise(0)) \
                .agg(F.avg("points").alias("avg_points")) \
                .collect()[0]
            return stats["avg_points"] if stats["avg_points"] is not None else 0.0
        except:
            return 0.0

    def get_h2h_stats(self, home_team, away_team, limit=10):
        try:
            h2h_matches = matches_df.filter(
                ((col("team") == home_team) & (col("opponent") == away_team)) |
                ((col("team") == away_team) & (col("opponent") == home_team))
            ).orderBy(col("date").desc()).limit(limit).collect()

            if not h2h_matches:
                return {
                    "home_win_rate": 0.5,
                    "draw_rate": 0.0,
                    "away_win_rate": 0.5,
                    "total_matches": 0
                }

            home_wins = 0
            draws = 0
            away_wins = 0
            total = len(h2h_matches)

            for row in h2h_matches:
                team = row["team"]
                opponent = row["opponent"]
                result = row["result"]

                if team == home_team and opponent == away_team:
                    if result == "W":
                        home_wins += 1
                    elif result == "D":
                        draws += 1
                    else:  # L
                        away_wins += 1
                elif team == away_team and opponent == home_team:
                    if result == "W":  # Away team won
                        away_wins += 1
                    elif result == "D":
                        draws += 1
                    else:  # L - Home team won
                        home_wins += 1

            return {
                "home_win_rate": home_wins / total if total > 0 else 0.5,
                "draw_rate": draws / total if total > 0 else 0.0,
                "away_win_rate": away_wins / total if total > 0 else 0.5,
                "total_matches": total
            }
        except Exception as e:
            print(f"H2H error: {e}")
            return {
                "home_win_rate": 0.5,
                "draw_rate": 0.0,
                "away_win_rate": 0.5,
                "total_matches": 0
            }

    def get_home_away_form_stats(self, team, venue="Home", n_matches=5):
        try:
            if venue == "Home":
                filtered = matches_df.filter((col("team") == team) & (col("venue") == "Home"))
            else:
                filtered = matches_df.filter((col("team") == team) & (col("venue") == "Away"))

            filtered = filtered.orderBy(desc("date")).limit(n_matches)

            stats = filtered.agg(
                avg(when(col("result") == "W", 1).otherwise(0)).alias("win_pct"),
                avg(col("xg")).alias("avg_xg"),
                avg(col("xga")).alias("avg_xga")
            ).collect()[0]

            return {
                "win_pct": stats["win_pct"] * 100 if stats["win_pct"] is not None else 0,
                "avg_xg": stats["avg_xg"] if stats["avg_xg"] is not None else 0,
                "avg_xga": stats["avg_xga"] if stats["avg_xga"] is not None else 0
            }
        except Exception as e:
            print(f"Venue form stats error for {team}: {e}")
            return {"win_pct": 0, "avg_xg": 0, "avg_xga": 0}


feature_extractor = SafeFeatureExtractor()


class ModelManager:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.ready = False
        self.queue = Queue()
        self.feature_names = [
            "home_win_pct", "home_draw_pct", "home_avg_xg", "home_avg_goals",
            "home_avg_xga", "away_avg_xga", "away_win_pct", "away_draw_pct",
            "away_avg_xg", "away_avg_goals", "home_formation_winrate",
            "away_formation_winrate", "home_power", "away_power",
            "home_form_win_pct", "home_form_avg_xg", "home_form_avg_xga",
            "away_form_win_pct", "away_form_avg_xg", "away_form_avg_xga",
            "h2h_home_win_rate", "h2h_away_win_rate", "h2h_draw_rate",
            "net_attack", "net_defense", "power_diff",
            "xg_momentum", "win_draw_ratio"
        ]

    def background_train(self):
        try:
            print("Starting training with XGBoost...")

            sample = matches_df.sample(fraction=1.0).collect()
            X, y = [], []

            for row in sample:
                try:
                    home = row.team if row.venue == "Home" else row.opponent
                    away = row.opponent if row.venue == "Home" else row.team

                    home_stats = feature_extractor.get_team_stats(home)
                    away_stats = feature_extractor.get_team_stats(away)
                    formation_stats_home = feature_extractor.get_formation_stats(home)
                    formation_stats_away = feature_extractor.get_formation_stats(away)
                    home_power = feature_extractor.get_power_ranking(home)
                    away_power = feature_extractor.get_power_ranking(away)
                    h2h_stats = feature_extractor.get_h2h_stats(home, away)
                    home_form = feature_extractor.get_home_away_form_stats(home, venue="Home")
                    away_form = feature_extractor.get_home_away_form_stats(away, venue="Away")

                    features = [
                        home_stats["win_pct"], home_stats["draw_pct"],
                        home_stats["avg_xg"], home_stats["avg_goals"],
                        home_stats["avg_xga"], away_stats["avg_xga"],
                        away_stats["win_pct"], away_stats["draw_pct"],
                        away_stats["avg_xg"], away_stats["avg_goals"],
                        formation_stats_home["winrate"],
                        formation_stats_away["winrate"],
                        home_power, away_power,
                        home_form["win_pct"], home_form["avg_xg"], home_form["avg_xga"],
                        away_form["win_pct"], away_form["avg_xg"], away_form["avg_xga"],
                        h2h_stats["home_win_rate"], h2h_stats["away_win_rate"], h2h_stats["draw_rate"],
                        home_stats["avg_xg"] - away_stats["avg_xga"],
                        away_stats["avg_xg"] - home_stats["avg_xga"],
                        home_power - away_power,
                        home_form["avg_xg"] - away_form["avg_xg"],
                        home_stats["win_pct"] / (home_stats["draw_pct"] + 0.1)
                    ]

                    if row.result == "W":
                        label = 0 if row.venue == "Home" else 2
                    elif row.result == "D":
                        label = 1
                    else:
                        label = 2 if row.venue == "Home" else 0

                    X.append(features)
                    y.append(label)
                except Exception as e:
                    print(f"Skipping row: {e}")
                    continue

            X = np.array(X)
            y = np.array(y)

            self.scaler.fit(X)
            X_scaled = self.scaler.transform(X)

            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42
            )

            # Calculate class weights
            classes = np.unique(y_train)
            weights = compute_class_weight('balanced', classes=classes, y=y_train)
            class_weights = dict(zip(classes, weights))

            # XGBoost parameters
            self.model = XGBClassifier(
                objective='multi:softprob',
                num_class=3,
                n_estimators=200,
                max_depth=5,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.1,
                scale_pos_weight=weights[0]/weights[1],  # Adjust for class imbalance
                random_state=42
            )

            # Train model
            self.model.fit(X_train, y_train)

            # Evaluate
            y_pred = self.model.predict(X_test)
            print("\nXGBoost Evaluation:")
            print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
            print(classification_report(y_test, y_pred))

            # Feature importance
            plt.figure(figsize=(10, 8))
            plt.barh(self.feature_names, self.model.feature_importances_)
            plt.title("Feature Importance")
            plt.tight_layout()
            plt.show()

            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                        display_labels=["Home Win", "Draw", "Away Win"])
            disp.plot(cmap='Blues')
            plt.title("Confusion Matrix")
            plt.show()

            self.ready = True
            self.queue.put(True)

        except Exception as e:
            print(f"Training failed: {e}")
            self.queue.put(False)

    def predict(self, home_team, away_team):
        if not self.ready:
            return {"error": "Model not ready"}

        try:
            home_stats = feature_extractor.get_team_stats(home_team)
            away_stats = feature_extractor.get_team_stats(away_team)
            formation_stats_home = feature_extractor.get_formation_stats(home_team)
            formation_stats_away = feature_extractor.get_formation_stats(away_team)
            home_power = feature_extractor.get_power_ranking(home_team)
            away_power = feature_extractor.get_power_ranking(away_team)
            h2h_stats = feature_extractor.get_h2h_stats(home_team, away_team)
            home_form = feature_extractor.get_home_away_form_stats(home_team, venue="Home")
            away_form = feature_extractor.get_home_away_form_stats(away_team, venue="Away")

            # Get opponent formation analysis
            home_vs_away_formation = feature_extractor.get_formation_stats(away_team, is_opponent=True)
            away_vs_home_formation = feature_extractor.get_formation_stats(home_team, is_opponent=True)

            print("\n=== FEATURE STATISTICS ===")
            print(f"\nHome Team ({home_team}):")
            print(f"- Win %: {home_stats['win_pct']:.1f}%")
            print(f"- Avg xG: {home_stats['avg_xg']:.2f}")
            print(f"- Most Used Formation: {formation_stats_home['formation']} "
                  f"({formation_stats_home['formation_count']}/{formation_stats_home['total_matches']} matches)")
            print(f"- Opponents' Win Rate vs this Formation: {formation_stats_home['winrate']:.1f}%")
            print(f"- Home Form (Win %): {home_form['win_pct']:.1f}%")
            print(f"- Power Ranking: {home_power:.2f}")

            print(f"\nAway Team ({away_team}):")
            print(f"- Win %: {away_stats['win_pct']:.1f}%")
            print(f"- Avg xG: {away_stats['avg_xg']:.2f}")
            print(f"- Most Used Formation: {formation_stats_away['formation']} "
                  f"({formation_stats_away['formation_count']}/{formation_stats_away['total_matches']} matches)")
            print(f"- Opponents' Win Rate vs this Formation: {formation_stats_away['winrate']:.1f}%")
            print(f"- Away Form (Win %): {away_form['win_pct']:.1f}%")
            print(f"- Power Ranking: {away_power:.2f}")

            print(f"\nHead-to-Head (Last {h2h_stats['total_matches']} matches):")
            print(f"- {home_team} Win Rate: {h2h_stats['home_win_rate'] * 100:.1f}%")
            print(f"- Draw Rate: {h2h_stats['draw_rate'] * 100:.1f}%")
            print(f"- {away_team} Win Rate: {h2h_stats['away_win_rate'] * 100:.1f}%")

            print(f"\nFormation Analysis:")
            print(f"- {home_team}'s win rate vs {away_team}'s formation: {home_vs_away_formation['winrate']:.1f}%")
            print(f"- {away_team}'s win rate vs {home_team}'s formation: {away_vs_home_formation['winrate']:.1f}%")

            features = np.array([[
                home_stats["win_pct"], home_stats["draw_pct"],
                home_stats["avg_xg"], home_stats["avg_goals"],
                home_stats["avg_xga"], away_stats["avg_xga"],
                away_stats["win_pct"], away_stats["draw_pct"],
                away_stats["avg_xg"], away_stats["avg_goals"],
                formation_stats_home["winrate"],
                formation_stats_away["winrate"],
                home_power, away_power,
                home_form["win_pct"], home_form["avg_xg"], home_form["avg_xga"],
                away_form["win_pct"], away_form["avg_xg"], away_form["avg_xga"],
                h2h_stats["home_win_rate"], h2h_stats["away_win_rate"], h2h_stats["draw_rate"],
                home_stats["avg_xg"] - away_stats["avg_xga"],
                away_stats["avg_xg"] - home_stats["avg_xga"],
                home_power - away_power,
                home_form["avg_xg"] - away_form["avg_xg"],
                home_stats["win_pct"] / (home_stats["draw_pct"] + 0.1)
            ]])

            features_scaled = self.scaler.transform(features)
            probs = self.model.predict_proba(features_scaled)[0]

            return {
                "Home Win": f"{probs[0] * 100:.1f}%",
                "Draw": f"{probs[1] * 100:.1f}%",
                "Away Win": f"{probs[2] * 100:.1f}%",
                "features": {
                    "home_team": home_team,
                    "away_team": away_team,
                    "home_stats": home_stats,
                    "away_stats": away_stats,
                    "h2h_stats": h2h_stats,
                    "formation_stats_home": formation_stats_home,
                    "formation_stats_away": formation_stats_away,
                    "home_form": home_form,
                    "away_form": away_form,
                    "home_vs_away_formation": home_vs_away_formation,
                    "away_vs_home_formation": away_vs_home_formation
                }
            }
        except Exception as e:
            return {"error": str(e)}


model_manager = ModelManager()
training_thread = threading.Thread(target=model_manager.background_train, daemon=True)
training_thread.start()


class FootballPredictorApp:
    def __init__(self, root):
        self.root = root
        self.setup_ui()

    def setup_ui(self):
        self.root.title("Football Match Predictor (XGBoost)")

        # Team selection
        ttk.Label(self.root, text="Home Team:").grid(row=0, column=0, padx=5, pady=5)
        self.home_var = tk.StringVar()
        self.home_dropdown = ttk.Combobox(self.root, textvariable=self.home_var, values=teams)
        self.home_dropdown.grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(self.root, text="Away Team:").grid(row=1, column=0, padx=5, pady=5)
        self.away_var = tk.StringVar()
        self.away_dropdown = ttk.Combobox(self.root, textvariable=self.away_var, values=teams)
        self.away_dropdown.grid(row=1, column=1, padx=5, pady=5)

        # Prediction button
        self.predict_btn = ttk.Button(self.root, text="Predict", command=self.on_predict)
        self.predict_btn.grid(row=2, column=0, columnspan=2, pady=10)

        # Status label
        self.status_var = tk.StringVar(value="Initializing...")
        ttk.Label(self.root, textvariable=self.status_var).grid(row=3, column=0, columnspan=2)

        # Results display
        self.result_var = tk.StringVar()
        self.result_label = ttk.Label(self.root, textvariable=self.result_var, wraplength=300)
        self.result_label.grid(row=4, column=0, columnspan=2, padx=5, pady=5)

        # Check training status periodically
        self.check_training_status()

    def check_training_status(self):
        if not model_manager.queue.empty():
            success = model_manager.queue.get()
            if success:
                self.status_var.set("Model ready for predictions")
            else:
                self.status_var.set("Model training failed")
        else:
            self.status_var.set("Training model...")
            self.root.after(1000, self.check_training_status)

    def plot_stats(self, features):
        plt.close('all')
        home_team = features["home_team"]
        away_team = features["away_team"]
        home_stats = features["home_stats"]
        away_stats = features["away_stats"]
        h2h_stats = features["h2h_stats"]

        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        fig.suptitle(f"Team Statistics: {home_team} vs {away_team}")

        # Win Percentage
        axes[0, 0].bar([home_team, away_team],
                       [home_stats["win_pct"], away_stats["win_pct"]],
                       color=['blue', 'red'])
        axes[0, 0].set_title("Win Percentage")
        axes[0, 0].set_ylim(0, 100)

        # Average Expected Goals (xG)
        axes[0, 1].bar([home_team, away_team],
                       [home_stats["avg_xg"], away_stats["avg_xg"]],
                       color=['blue', 'red'])
        axes[0, 1].set_title("Average Expected Goals (xG)")

        # Home/Away Form
        home_form = features.get("home_form", {"win_pct": 0})
        away_form = features.get("away_form", {"win_pct": 0})
        axes[1, 0].bar(["Home Form", "Away Form"],
                       [home_form["win_pct"], away_form["win_pct"]],
                       color=['blue', 'red'])
        axes[1, 0].set_title("Home/Away Win Percentage")
        axes[1, 0].set_ylim(0, 100)

        # Head-to-Head (now showing all three outcomes)
        h2h_labels = [f"{home_team} Win", "Draw", f"{away_team} Win"]
        h2h_values = [h2h_stats["home_win_rate"] * 100,
                      h2h_stats["draw_rate"] * 100,
                      h2h_stats["away_win_rate"] * 100]
        axes[1, 1].bar(h2h_labels, h2h_values, color=['blue', 'gray', 'red'])
        axes[1, 1].set_title("Head-to-Head Results")
        axes[1, 1].set_ylim(0, 100)

        plt.tight_layout()
        plt.show(block=False)

    def on_predict(self):
        home = self.home_var.get()
        away = self.away_var.get()

        if not home or not away:
            self.result_var.set("Please select both teams")
            return

        if not model_manager.ready:
            self.result_var.set("Model not ready yet")
            return

        self.result_var.set("Calculating prediction...")
        self.root.update()

        try:
            prediction = model_manager.predict(home, away)
            if "error" in prediction:
                self.result_var.set(f"Error: {prediction['error']}")
            else:
                result_text = (
                    f"Prediction for {home} vs {away}:\n"
                    f"Home Win: {prediction['Home Win']}\n"
                    f"Draw: {prediction['Draw']}\n"
                    f"Away Win: {prediction['Away Win']}"
                )
                self.result_var.set(result_text)

                if "features" in prediction:
                    self.plot_stats(prediction["features"])
        except Exception as e:
            self.result_var.set(f"Prediction failed: {e}")


if __name__ == "__main__":
    root = tk.Tk()
    app = FootballPredictorApp(root)
    root.mainloop()
    spark.stop()