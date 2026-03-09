# Melbourne Housing Price Predictor

基于 Victorian Property Sales Report (VPSR) 的墨尔本房价预测项目，包含：
- 数据清洗（多份 `.xls` 合并）
- 特征工程（时序滞后与滚动特征）
- 模型对比（线性回归 / 随机森林 / GBDT / XGBoost / LightGBM）
- FastAPI 预测接口

## 1. 项目结构

```text
.
├─ dataset/                  # 你的 VPSR 原始数据（.xls）
├─ artifacts/                # 训练输出（自动生成）
│  ├─ clean_long_data.csv
│  ├─ best_model.joblib
│  ├─ latest_history.csv
│  └─ metrics.json
├─ src/
│  ├─ data_pipeline.py       # 数据清洗与标准化
│  ├─ train.py               # 特征工程 + 模型训练/对比
│  └─ api_fastapi.py         # FastAPI 服务
├─ notebooks/
│  └─ melbourne_housing_model.ipynb   # Jupyter Notebook 工作流
└─ requirements.txt
```

## 0. 使用 Jupyter Notebook（推荐）

```bash
jupyter notebook
```

然后打开：
- `notebooks/melbourne_housing_model.ipynb`

## 2. 安装依赖

```bash
pip install -r requirements.txt
```

## 3. 训练模型

```bash
python src/train.py
```

训练后会在 `artifacts/metrics.json` 里看到模型对比指标（MAE/RMSE/R2）和最佳模型。

## 4. 启动 API

```bash
uvicorn src.api_fastapi:app --reload --host 0.0.0.0 --port 8000
```

接口：
- `GET /health`：健康检查
- `POST /predict`：预测某个区域某季度的中位房价

## 5. 预测请求示例

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "locality": "ABBOTSFORD",
    "year": 2026,
    "quarter": 2
  }'
```

说明：
- `lag_1` / `lag_2` / `rolling_mean_2` 可不传，接口会尝试从 `artifacts/latest_history.csv` 自动补齐。
- 如果某个区域历史数据不足，会返回 400，并提示手动提供 lag 特征。

## 6. 当前实现的关键点

- 清洗层会自动识别季度列（如 `Jan-Mar 2024`、`Apr-Jun` + 年份行），并过滤销售量/变化率列。
- 多来源重复季度数据按 `locality-year-quarter` 求均值去重。
- 训练采用按时间切分（避免随机切分造成数据泄漏）。

## 7. 下一步可选优化

- 引入地理特征（Postcode、距CBD、学区）
- 引入宏观变量（利率、人口增长、库存）
- 使用更强的时序模型（LightGBM / XGBoost / Temporal CV）
- 增加可视化看板（Streamlit）
