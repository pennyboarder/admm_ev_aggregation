# フォルダ構成整理案

## 提案する新しい構成

```
admm_ev_aggregation/
├── src/                           # メインソースコード
│   ├── core/                      # コアモデルとクラス
│   │   ├── __init__.py
│   │   ├── models.py              # EVAgent, GlobalParams, MarketInputs
│   │   └── exceptions.py          # カスタム例外クラス（新規）
│   │
│   ├── solvers/                   # 最適化ソルバー
│   │   ├── __init__.py
│   │   ├── admm_coordinator.py    # ADMM協調最適化
│   │   ├── local_solver.py        # 従来の個別EV最適化
│   │   ├── local_solver_oop.py    # OOP版個別EV最適化
│   │   └── bidding_optimizer.py   # 入札最適化
│   │
│   └── utils/                     # ユーティリティ
│       ├── __init__.py
│       ├── utils.py               # ブロック構築等
│       ├── visualization.py       # 可視化機能
│       ├── csv_export.py          # データエクスポート
│       ├── demo.py                # デモデータ生成
│       └── demo_discharge.py      # 放電デモデータ
│
│   ├── main/                      # メイン実行スクリプト
│   │   ├── __init__.py
│   │   ├── main.py                # メイン実行（統合版）
│   │   ├── main_with_bidding.py   # 入札機能版
│   │   └── main_auto_bidding.py   # 自動入札版
│
├── examples/                      # 使用例・サンプルコード
│   ├── __init__.py
│   ├── basic_example.py           # 基本的な使用例
│   ├── advanced_example.py        # 高度な使用例
│   ├── bidding_example.py         # 入札最適化の例
│   └── discharge_example.py       # 放電制御の例
│
├── tests/                         # テストコード
│   ├── __init__.py
│   ├── integration/               # 統合テスト
│   │   ├── __init__.py
│   │   ├── test_discharge.py      # 放電シナリオテスト
│   │   ├── test_parallel_performance.py
│   │   ├── test_tqdm_convergence.py
│   │   ├── test_nested_tqdm.py
│   │   └── test_mutual_exclusion.py
│   │
│   └── unit/                      # 単体テスト
│       ├── __init__.py
│       ├── test_models.py         # モデルテスト（新規統合）
│       ├── test_solvers.py        # ソルバーテスト（新規統合）
│       ├── test_utils.py          # ユーティリティテスト（新規統合）
│       ├── test_ev_agent_oop.py
│       ├── test_l2_regularization.py
│       ├── test_l2_comprehensive.py
│       ├── test_baseline_enforcement.py
│       ├── test_baseline_enforcement_enhanced.py
│       ├── test_reserve_baseline.py
│       ├── test_admm_baseline_enforcement.py
│       ├── test_baseline_constraint_simple.py
│       ├── test_bidding_block_baseline.py
│       ├── test_blocks_mapping.py
│       ├── test_direct_baseline_enforcement.py
│       ├── test_bidding_optimization.py
│       ├── test_fixed_availability.py
│       └── test_targeted_control.py
│
├── debug/                         # デバッグツール
│   ├── __init__.py
│   ├── debug_baseline.py
│   ├── debug_l2.py
│   ├── check_capacity.py
│   └── check_availability.py
│
├── docs/                          # ドキュメント
│   ├── README.md                  # プロジェクト概要
│   ├── API.md                     # API仕様書
│   └── examples.md                # 使用例
│
├── outputs/                       # 実行結果出力（既存）
├── __init__.py                    # プロジェクトルート
├── admm_ev_aggregator.py          # 旧統合ファイル（移行後削除予定）
└── FOLDER_STRUCTURE.md            # この構成説明
```

## 整理の利点

### 1. **可読性向上**
- 機能別にファイルが分離され、目的の機能が見つけやすい
- **全てのコードがsrc下に整理**され、構造が明確
- テストコードとメインコードが分離

### 2. **保守性向上**
- コア機能、ソルバー、ユーティリティが明確に分離
- 単体テストと統合テストが分離され、テスト戦略が明確

### 3. **拡張性向上**
- 新しいソルバーやユーティリティを追加しやすい
- examplesフォルダに用途別のサンプルを配置可能

### 4. **プロフェッショナルな構成**
- **src/レイアウト**: モダンなPythonプロジェクト標準
- **パッケージ化対応**: setuptools, poetryなどに最適
- CI/CDやDockerビルドが容易

## 実行方法（src/レイアウト）

```bash
# メイン実行
python -m src.main.main

# 入札版実行
python -m src.main.main_with_bidding

# または、ルートに実行スクリプト作成
python run.py  # → src.main.mainを呼び出し
```

## 移行手順

1. **段階1**: コアファイルの移動 (models.py等)
2. **段階2**: ソルバーファイルの移動  
3. **段階3**: メイン実行ファイルの移動
4. **段階4**: ユーティリティとテストの移動
5. **段階5**: インポートパスの更新
6. **段階6**: 実行スクリプト(run.py)作成

## ファイル分類詳細

### コア機能 (src/core/)
- **models.py**: EVAgent, GlobalParams, MarketInputs - システムの基本データ構造

### ソルバー (src/solvers/)
- **admm_coordinator.py**: ADMM分散最適化の協調機能
- **local_solver.py**: 個別EV最適化（従来版）
- **local_solver_oop.py**: 個別EV最適化（OOP版）
- **bidding_optimizer.py**: 市場入札の最適化

### ユーティリティ (src/utils/)
- **utils.py**: ブロック構築等の補助機能
- **visualization.py**: グラフ作成と可視化
- **csv_export.py**: 結果のCSVエクスポート
- **demo.py / demo_discharge.py**: テスト用デモデータ生成
