###Streamlit应用程序开发
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Load the model
model = joblib.load('Vc_7.pkl')
scaler = joblib.load('scaler7.pkl') 

# Define feature names
feature_names = ['Age','Weight','Pre_Na','Post_UAR','Post_ALB','Post_P','Post_PLT']

# Streamlit user interface
st.title("DBDLT-AKI Phase 2 Predictor")

# Age: numerical input
Age = st.number_input("Age (years):", min_value=18, max_value=100, value=50)
# Weight: numerical input
Weight = st.number_input("Weight (kg):", min_value=40, max_value=100, value=65)
# Pre_Na: numerical input
Pre_Na = st.number_input("Pre_Na (mmol/L):", min_value=110.0, max_value=150.0, value=130.0)
# Post_UAR：numerical input
Post_UAR = st.number_input("Post_UAR (mg/g):", min_value=0.0, max_value=40.0, value=6.5)
# Post_ALB: numerical input
Post_ALB = st.number_input("Post_ALB (g/L):", min_value=0.0, max_value=50.0, value=35.0)
# Post_P：numerical input
Post_P = st.number_input("Post_P (mmol/L):", min_value=0.00, max_value=3.00, value=1.00)
# Post_PLT：numerical input
Post_PLT = st.number_input("Post_PLT (10^9/L):", min_value=0, max_value=400, value=50)

# 准备输入特征
feature_values = [Age,Weight,Pre_Na,Post_UAR,Post_ALB,Post_P,Post_PLT]
features = np.array([feature_values])

# 关键修改：使用 pandas DataFrame 来确保列名
features_df = pd.DataFrame(features, columns=feature_names)
standardized_features_1 = scaler.transform(features_df)

# 关键修改：确保 final_features 是一个二维数组，并且用 DataFrame 传递给模型
standardized_features = pd.DataFrame(standardized_features_1, columns=feature_names)

if st.button("Predict"):    
    # 标准化特征
    # standardized_features = scaler.transform(features)

    # Predict class and probabilities    
    predicted_class = model.predict(standardized_features)[0]   
    predicted_proba = model.predict_proba(standardized_features)[0]

    # Display prediction results    
    st.write(f"**Predicted Class:** {predicted_class} (1: Disease, 0: No Disease)")   
    formatted_proba = ", ".join(f"{prob:.2f}" for prob in predicted_proba)
    st.write(f"**Prediction Probabilities:** {formatted_proba}")

    probability = predicted_proba[predicted_class] * 100
    # Generate advice based on prediction results  
    if predicted_class == 1:        
        advice = (            
            f"According to the model, you are at high risk of developing acute kidney injury (AKI) after heart transplant surgery. "            
            f"The model predicts a {probability:.1f}% probability of AKI. "            
            "It is recommended to closely monitor kidney function indicators and maintain communication with your medical team for timely prevention or intervention."        
        )    
    else:        
        advice = (           
            f"According to the model, you are at low risk of developing acute kidney injury (AKI) after heart transplant surgery. "            
            f"The model predicts a {probability:.1f}% probability of not developing AKI. "            
            "However, it is still important to closely monitor kidney function post-surgery and follow the guidance of your medical team to ensure a smooth     recovery."        
        )      
    st.write(advice)

    # SHAP Explanation
    st.subheader("SHAP Force Plot Explanation")

    # 创建SHAP解释器
    # 假设 X_train 是用于训练模型的特征数据
    df=pd.read_csv('train.csv',encoding='utf8')
    trainy=df.AKI123
    x_train=df.drop('AKI123',axis=1)
    from sklearn.preprocessing import StandardScaler
    continuous_cols = [1,2,3,4,5,6,7]
    trainx = x_train.copy()
    scaler = StandardScaler()
    trainx.iloc[:, continuous_cols] = scaler.fit_transform(x_train.iloc[:, continuous_cols])

    explainer_shap = shap.KernelExplainer(model.predict_proba, trainx)
    
    # 获取SHAP值
    shap_values = explainer_shap.shap_values(pd.DataFrame(standardized_features,columns=feature_names))
    
    # 将标准化前的原始数据存储在变量中
    original_feature_values = pd.DataFrame(features, columns=feature_names)

    # Display the SHAP force plot for the predicted class    
    if predicted_class == 1:        
        shap.force_plot(explainer_shap.expected_value[1], shap_values[:,:,1], original_feature_values, matplotlib=True)    
    else:        
        shap.force_plot(explainer_shap.expected_value[0], shap_values[:,:,0], original_feature_values, matplotlib=True)    
    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)    
    st.image("shap_force_plot.png", caption='SHAP Force Plot Explanation')

