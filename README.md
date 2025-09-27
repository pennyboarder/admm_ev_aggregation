# ADMM EV Aggregation

電気自動車（EV）の充電スケジューリングと電力市場での予備力提供を最適化するためのADMM（Alternating Direction Method of Multipliers）アルゴリズムの実装です。

## 構成

プロジェクトは以下のモジュールに分割されています：

### `models.py`
- `EVAgent`: 各EVの特性（容量、SOC、充電能力など）
- `GlobalParams`: 全体パラメータ（時間解像度、劣化コストなど）
- `MarketInputs`: 市場データ（電力価格、予備力価格）

### `utils.py`
- `build_blocks()`: 5分単位のタイムスロットを30分ブロックに分割
- `blocks_to_slot_mask()`: ブロック単位の制約をスロット単位のマスクに変換
- `build_post_windows()`: 各ブロック後のリカバリー時間窓を定義

### `local_solver.py`
各EV個別の充電スケジュール最適化を行います。
- 決定変数: 充電電力、予備力容量、リカバリーエネルギー
- 制約: SOC上下限、予備力実現可能性、リカバリー制約
- 目的関数: 予備力収入 - エネルギー購入費 - 劣化コスト - ADMMペナルティ項

### `admm_coordinator.py`
ADMM協調アルゴリズムを実装し、以下の3つの合意制約を満たします：
1. 予備力入札量の達成
2. リカバリーエネルギー配分
3. ベースライン追従（入札ブロックのみ）

### `demo.py`
デモシナリオ生成（ランダムなEVフリート、時間帯別電力価格など）

### `main.py`
メイン実行スクリプト

## 実行方法

### 基本実行（可視化付き）
```bash
python main.py
```

### カスタム実行例
```python
from demo import build_demo
from admm_coordinator import admm_cut_with_baseline
from visualization import *

# データ生成
evs, gp, mi, R_bid, enforce_blocks, P_base = build_demo(n_evs=6, hours=12)

# 最適化実行
results = admm_cut_with_baseline(evs, gp, mi, R_bid, P_base, enforce_blocks)

# 可視化（画面表示のみ）
plot_ev_charging_schedule(evs, gp, results)  # 個別EV充電スケジュール
plot_soc_evolution(evs, gp, results)         # SOC進化
plot_aggregate_analysis(evs, gp, mi, results, P_base)  # 集約分析
plot_convergence_history(results)           # 収束履歴

# 可視化（画像保存付き）
plot_ev_charging_schedule(evs, gp, results, save_path="output/charging.png")
plot_soc_evolution(evs, gp, results, save_path="output/soc.png")

# 全グラフを一括保存
save_all_plots(evs, gp, mi, results, P_base, 
               output_dir="output", prefix="my_simulation")

# CSVデータ出力
export_all_csv_data(evs, gp, mi, results, P_base, enforce_blocks,
                    output_dir="output", prefix="my_simulation")

# 放電パターン（外部使用）を含むデモ
from demo_discharge import build_demo_with_discharge
evs, gp, mi, R_bid, enforce_blocks, P_base = build_demo_with_discharge(
    n_evs=5, hours=12, include_v2g=True, discharge_intensity=0.6
)
```

## 主な特徴

1. **分散最適化**: 各EVが個別に最適化しつつ、全体制約を満たす
2. **予備力サービス**: 充電停止により上げ調整力を提供
3. **ベースライン追従**: 入札時のベースライン電力パターンを維持
4. **リカバリー考慮**: 予備力提供後の充電補償を事前に計画
5. **L1ペナルティ**: 線形計画法で解ける形に定式化
6. **EV放電機能**: 外部使用パターンとV2G（Vehicle-to-Grid）機能

## 可視化機能

新たに追加された可視化機能により、以下のグラフが生成できます：

1. **個別EV充電スケジュール** - 各EVの時間別充電電力、利用可能性、ブロック境界
2. **SOC進化** - 各EVの充電状態（State of Charge）の時間変化
3. **集約分析** - 全体の充電電力、予備力容量、リカバリーエネルギー、市場価格
4. **ADMM収束履歴** - アルゴリズムの収束過程と残差の変化

### 画像保存機能

- **時刻フォルダ保存**: 実行時刻の名前のフォルダ（`output/run_YYYYMMDD_HHMMSS/`）に結果を整理保存
- **詳細結果ファイル**: 最適化結果の詳細をテキストファイル（`detailed_results.txt`）として保存
- **個別保存**: 各プロット関数に`save_path`パラメータを指定
- **一括保存**: `save_all_plots()`関数で全グラフを一度に保存
- **高解像度**: 300dpiの高品質PNG形式で保存

### 保存される内容

各実行時に以下が`output/run_YYYYMMDD_HHMMSS/`フォルダに保存されます：

- `ev_charging_schedule.png` - 個別EV充電スケジュール
- `soc_evolution.png` - SOC進化グラフ
- `aggregate_analysis.png` - 集約分析（全体充電・予備力・価格）
- `convergence_history.png` - ADMM収束履歴
- `detailed_results.txt` - 詳細な最適化結果（テキスト形式）
- `csv_data/` - 時系列データのCSVファイル群

### CSV出力データ

`csv_data/`フォルダには以下のCSVファイルが保存されます：

- `timeslot_data.csv` - 5分毎の時系列データ（集約充電電力、ベースライン、価格等）
- `block_data.csv` - 30分ブロック毎のデータ（予備力容量、リカバリーエネルギー等）
- `individual_ev_data.csv` - 個別EVの充電スケジュールとSOC進化
- `market_data.csv` - 市場価格とコスト・収益データ
- `admm_convergence.csv` - ADMM収束履歴の詳細
- `summary_statistics.csv` - 全体のサマリー統計

## 依存関係

- PuLP (線形計画法ソルバー)
- matplotlib (可視化)
- numpy (数値計算)
- Python 3.7+
