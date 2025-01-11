import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os
from dotenv import load_dotenv
import shap
import matplotlib.pyplot as plt

# 加载 .env 文件
load_dotenv()
# 获取配置
THRESHOLD = float(os.getenv("THRESHOLD", 0.14))

# 加载模型
model_save_path = r"rf_model.pkl"
model = joblib.load(model_save_path)

# 特征列顺序
feature_columns = [
    "Gender", "Age", "Residence", "BMI", "smoke", "drink", "FX", "BM", "LWY", "FIT"
]

# 特征范围定义 (包含自定义标签)
feature_ranges = {
    "Gender": {"type": "categorical", "options": [0, 1]},
    "Age": {"type": "categorical", "options": [0, 1, 2, 3]},
    "Residence": {"type": "categorical", "options": [0, 1]},
    "BMI": {"type": "categorical", "options": [0, 1]},
    "smoke": {"type": "categorical", "options": [0, 1]},
    "drink": {"type": "categorical", "options": [0, 1]},
    "FX": {"type": "categorical", "options": [0, 1], "label": "History of chronic diarrhea"},
    "BM": {"type": "categorical", "options": [0, 1], "label": "History of chronic constipation"},
    "LWY": {"type": "categorical", "options": [0, 1], "label": "History of chronic appendicitis or appendectomy"},
    "FIT": {"type": "categorical", "options": [0, 1]},
}

# 显示特征映射（用于 SHAP 图）
feature_label_mapping = {
    "FX": "Chronic diarrhea",
    "BM": "Chronic constipation",
    "LWY": "Appendicitis",
}

# Streamlit 界面
st.title("Prediction Model")

# 动态生成输入项
st.header("Enter the following feature values:")
feature_values = []

# 严格按照 feature_columns 顺序生成输入项
for feature in feature_columns:
    if feature in feature_ranges:
        properties = feature_ranges[feature]
        feature_label = properties.get("label", feature)
        value = st.selectbox(
            label=f"{feature_label} (Select a value)",
            options=properties["options"],
        )
        feature_values.append(value)
    else:
        st.error(f"Feature '{feature}' is not defined in feature_ranges.")

# 校验特征值与列名对齐
def validate_features(feature_values, feature_columns):
    if len(feature_values) != len(feature_columns):
        raise ValueError("Feature values do not match the expected feature columns!")
    return True

# 校验特征值是否在预定义范围内
def validate_features_with_range(feature_values, feature_ranges):
    for feature, value in zip(feature_ranges.keys(), feature_values):
        properties = feature_ranges[feature]
        if properties["type"] == "categorical" and value not in properties["options"]:
            raise ValueError(f"Invalid value for {feature}: {value}. Expected: {properties['options']}")
    return True

try:
    validate_features(feature_values, feature_columns)
    validate_features_with_range(feature_values, feature_ranges)
except ValueError as e:
    st.error(f"Validation failed: {e}")
    st.stop()

# 转换为模型输入格式
features = pd.DataFrame([feature_values], columns=feature_columns)

# 替换 SHAP 图中的列名
def replace_feature_labels(features):
    renamed_features = features.rename(columns=feature_label_mapping)
    return renamed_features

# 生成 SHAP 图
def generate_shap_plot(model, features):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(features)

    # 替换 SHAP 图中的列名
    features_with_labels = replace_feature_labels(features)

    # 生成 SHAP 力图
    plt.figure(figsize=(15, 6))  # 设置图形大小
    shap.force_plot(
        explainer.expected_value[1],
        shap_values[1],
        features_with_labels,
        matplotlib=True,
    )

    # 添加标注，解释 SHAP 图中用到的特征
    plt.figtext(0.5, -0.05, 
                "History of chronic diarrhea -> Chronic diarrhea\n"
                "History of chronic constipation -> Chronic constipation\n"
                "History of chronic appendicitis or appendectomy -> Chronic appendicitis",
                wrap=True, horizontalalignment='center', fontsize=10)

    st.pyplot(plt.gcf())
    plt.clf()

if st.button("Predict"):
    if hasattr(model, 'best_estimator_'):
        best_model = model.best_estimator_
    else:
        best_model = model

    predicted_class = best_model.predict(features)[0]
    predicted_proba = best_model.predict_proba(features)[0]
    probability = predicted_proba[1]
    risk_level = "High Risk" if probability > THRESHOLD else "Low Risk"

    st.subheader("Prediction Results:")
    st.write(f"Predicted Probability: {probability:.2f}")
    st.write(f"Risk Level: {risk_level}")

    try:
        generate_shap_plot(best_model, features)
    except Exception as e:
        st.error(f"SHAP plot generation failed: {e}")