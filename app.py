import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.figure_factory as ff
from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
)

# Model Imports
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


# --- Data Loading and Preprocessing ---
def preprocess_data(file_path):
    """
    Loads and preprocesses the bank churn data as per the user's original notebook.
    This function handles column dropping, feature engineering, one-hot encoding,
    and min-max scaling for the entire dataset before splitting.
    """
    df = pd.read_csv(file_path, delimiter=',')

    # Drop the specified columns
    df = df.drop(["RowNumber", "CustomerId", "Surname"], axis=1)

    # Feature Engineering as in the user's notebook
    # Note: Using df.Age directly as per the original notebook, not df.Age-18
    df['BalanceSalaryRatio'] = df.Balance / df.EstimatedSalary
    df['TenureByAge'] = df.Tenure / df.Age
    df['CreditScoreGivenAge'] = df.CreditScore / df.Age

    # Handle infinite values created by division by zero
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()

    # One-hot encode the categorical variables
    df.loc[df.HasCrCard == 0, 'HasCrCard'] = -1
    df.loc[df.IsActiveMember == 0, 'IsActiveMember'] = -1
    df['HasCrCard'] = df['HasCrCard'].astype('int64')
    df['IsActiveMember'] = df['IsActiveMember'].astype('int64')

    lst = ['Geography', 'Gender']
    df_encoded = df.copy()
    for i in lst:
        if df_encoded[i].dtype == object:
            for j in df_encoded[i].unique():
                df_encoded[i+'_'+j] = np.where(df_encoded[i] == j, 1, -1)
    df_encoded = df_encoded.drop(lst, axis=1)

    # Reorder columns for consistency
    continuous_vars = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary',
                       'BalanceSalaryRatio', 'TenureByAge', 'CreditScoreGivenAge']
    cat_vars = ['HasCrCard', 'IsActiveMember', 'Geography_Spain', 'Geography_France',
                'Geography_Germany', 'Gender_Female', 'Gender_Male']
    df_processed = df_encoded[['Exited'] + continuous_vars + cat_vars]

    # MinMax scaling continuous variables
    scaler = MinMaxScaler()
    df_processed[continuous_vars] = scaler.fit_transform(df_processed[continuous_vars])

    return df_processed, continuous_vars


def train_models(df_train):
    """
    Fits the models and returns a dictionary of trained models.
    The parameters are taken from the user's GridSearchCV results.
    """
    X_train = df_train.loc[:, df_train.columns != 'Exited']
    y_train = df_train.Exited

    models = {}

    # Logistic Regression with polynomial features
    poly2 = PolynomialFeatures(degree=2)
    X_train_pol2 = poly2.fit_transform(X_train)
    log_pol2 = LogisticRegression(C=10, max_iter=300, penalty='l2', tol=0.0001, solver='liblinear')
    log_pol2.fit(X_train_pol2, y_train)
    models['Logistic Regression'] = log_pol2
    models['poly_transform'] = poly2

    # Random Forest
    rf_model = RandomForestClassifier(max_depth=8, max_features=9, min_samples_split=6, n_estimators=50)
    rf_model.fit(X_train, y_train)
    models['Random Forest'] = rf_model

    # XGBoost
    xgb_model = XGBClassifier(gamma=0.01, learning_rate=0.1, max_depth=7, min_child_weight=5, n_estimators=20)
    xgb_model.fit(X_train, y_train)
    models['XGBoost Classifier'] = xgb_model

    # SVM with RBF Kernel
    svm_rbf = SVC(C=100, gamma=0.1, probability=True, kernel='rbf')
    svm_rbf.fit(X_train, y_train)
    models['SVM (RBF)'] = svm_rbf

    return models


# --- Main Data and Model Preparation ---
file_path = 'dataset/Churn_Modelling.csv'
df_processed, continuous_vars = preprocess_data(file_path)

# Split the data into train and test sets
df_train = df_processed.sample(frac=0.8, random_state=200)
df_test = df_processed.drop(df_train.index)

X_train = df_train.loc[:, df_train.columns != 'Exited']
y_train = df_train.Exited

X_test = df_test.loc[:, df_test.columns != 'Exited']
y_test = df_test.Exited

# Fit models
trained_models = train_models(df_train)
feature_names = X_train.columns.tolist()

# Get metrics for all models on test data
model_results = []
for name, model in trained_models.items():
    if name == 'poly_transform':
        continue
    
    if name == 'Logistic Regression':
        X_test_transformed = trained_models['poly_transform'].transform(X_test)
        predictions = model.predict(X_test_transformed)
        probabilities = model.predict_proba(X_test_transformed)[:, 1]
    else:
        predictions = model.predict(X_test)
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(X_test)[:, 1]
        else:
            probabilities = predictions # For models without predict_proba

    model_results.append({
        'Model': name,
        'Accuracy': accuracy_score(y_test, predictions),
        'Recall': recall_score(y_test, predictions),
        'Precision': precision_score(y_test, predictions, zero_division=0),
        'F1-Score': f1_score(y_test, predictions),
        'ROC-AUC': roc_auc_score(y_test, probabilities) if hasattr(model, 'predict_proba') else 'N/A',
        'predictions': predictions,
        'probabilities': probabilities,
    })

metrics_df = pd.DataFrame(model_results).round(4)


# --- Dashboard Layout ---
app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
server = app.server

header = dbc.Navbar(
    dbc.Container(
        [
            html.Div(
                [
                    html.Span("🏦", className="me-2"),
                    dbc.NavbarBrand("Preventing Bank Customer Churn",
                                    class_name="fw-bold text-wrap", style={"color": "black"}),
                ], className="d-flex align-items-center"
            ),
            dbc.Badge("Dashboard", color="primary", className="ms-auto")
        ]
    ),
    color="light",
    class_name="shadow-sm mb-3"
)

# 1. ASK Tab
ask_tab = dcc.Markdown(
    """
    ### ❓ **ASK** — The Business Question

    This dashboard aims to answer a critical business question: **"Why are customers leaving our bank, and which ones are most likely to leave in the future?"**

    **Business Task**: Our goal is to predict which customers are at high risk of **churning**, meaning they will close their bank accounts or stop using our services. Losing customers is expensive, so identifying them early allows us to take proactive measures to keep them.

    **Stakeholders**: The key users of this dashboard are **Customer Service**, **Marketing**, and **Executive Leadership**. Customer service can use the predictions to target "low-hanging fruit"—customers who are likely to churn but can be easily retained. Marketing can design specific campaigns based on the factors that drive churn.

    **Deliverables**: This interactive dashboard provides a comprehensive analysis, from initial data exploration to final model recommendations, all in a clear, easy-to-understand format.
    """, className="p-4"
)

# 2. PREPARE Tab
prepare_tab = html.Div(
    children=[
        html.H4(
            ["📝 ", html.B("PREPARE"), " — Getting the Data Ready"],
            className="mt-4"
        ),
        html.P("Before we can build a predictive model, we need to understand and clean our data."),
        html.H5("Data Source"),
        html.P(
            ["We are using a dataset of 10,000 bank customers. We've split the data into a ", html.B("training set"), " (80% of the data) for building our models and a separate ", html.B("test set"), " (20%) to check if our models work on new, unseen data."]
        ),
        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader("Training Dataset"),
                            dbc.CardBody(
                                [
                                    html.P(f"Rows: {df_train.shape[0]}"),
                                    html.P(f"Features: {df_train.shape[1]}")
                                ]
                            ),
                        ], className="mb-3"
                    )
                ),
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader("Test Dataset"),
                            dbc.CardBody(
                                [
                                    html.P(f"Rows: {df_test.shape[0]}"),
                                    html.P(f"Features: {df_test.shape[1]}")
                                ]
                            ),
                        ], className="mb-3"
                    )
                ),
            ]
        ),
        html.H5("Key Data Preparations"),
        html.P(
            ["We performed several key steps to prepare the data for modeling:",
             html.Ul(
                 [
                     html.Li(
                         [
                             html.B("Feature Engineering:"),
                             " We created new, more insightful features from the existing ones, such as the ", html.B("BalanceSalaryRatio"), " (a customer's bank balance compared to their estimated salary), which proved to be a powerful predictor of churn."
                         ]
                     ),
                     html.Li(
                         [
                             html.B("One-Hot Encoding:"),
                             " We converted categorical variables like `Geography` (country) and `Gender` into a numerical format that our models can understand. For variables like `HasCrCard` and `IsActiveMember`, we changed '0' to '-1' to signify a negative relationship, helping models better interpret the data."
                         ]
                     ),
                     html.Li(
                         [
                             html.B("Scaling:"),
                             " We normalized our continuous data (e.g., `Age`, `CreditScore`) so that all features have a similar range. This prevents models from giving too much importance to features with larger values."
                         ]
                     )
                 ]
             )
            ]
        ),
    ], className="p-4"
)

# 3. ANALYZE Tab
analyze_tab = html.Div(
    children=[
        html.H4(
            ["📈 ", html.B("ANALYZE"), " — Finding Patterns and Building Models"],
            className="mt-4"
        ),
        dbc.Tabs([
            dbc.Tab(label="Exploratory Data Analysis", children=[
                html.Div(
                    children=[
                        html.H5("Churn Distribution and Key Factors", className="mt-4"),
                        html.P(
                            ["The pie chart below shows that about **20%** of our customers have churned. This is an important number because it means our data is ", html.B("imbalanced"), ". A simple model could be 80% accurate by just predicting that no one will ever churn, which is why we need more advanced metrics to evaluate our models."]
                        ),
                        dbc.Row([
                            dbc.Col(dcc.Graph(id="churn-pie-chart",
                                              figure=go.Figure(
                                                  data=[go.Pie(labels=['Retained', 'Exited'],
                                                               values=[df_processed.Exited[df_processed['Exited'] == 0].count(),
                                                                       df_processed.Exited[df_processed['Exited'] == 1].count()],
                                                               marker=dict(colors=['#1f77b4', '#ff7f0e'], line=dict(color="white", width=1.3)),
                                                               hoverinfo="label+percent", hole=0.5)],
                                                  layout=go.Layout(title="Proportion of Customer Churn", height=400, margin=dict(t=50, b=50))
                                              )), md=6),
                            dbc.Col(dcc.Graph(id="correlation-matrix"), md=6),
                        ]),
                        html.P(
                            ["The **Correlation Matrix** shows how strongly each feature relates to every other feature. A dark color indicates a strong relationship. We can see that `Age` and `NumOfProducts` have a noticeable correlation with `Exited` (churn)."]
                        ),
                        html.H5("Impact of Key Variables on Churn", className="mt-4"),
                        html.P("These plots help us understand which factors are most closely associated with customer churn."),
                        dbc.Row([
                            dbc.Col(dcc.Graph(id="categorical-churn-plot"), md=6),
                            dbc.Col(dcc.Graph(id="continuous-churn-plot"), md=6),
                        ]),
                        html.P(
                            ["Based on our analysis of the data, we found several interesting trends:",
                             html.Ul(
                                 [
                                     html.Li("Customers from **Germany** have a disproportionately higher churn rate compared to their population."),
                                     html.Li("The proportion of **female** customers churning is greater than that of male customers."),
                                     html.Li("Customers with **higher bank balances** are more likely to churn, which is a significant concern for the bank's capital."),
                                     html.Li("Older customers and those with either very short or very long **tenure** are more likely to churn.")
                                 ]
                             )
                            ]
                        )
                    ], className="p-4"
                )
            ]),
            dbc.Tab(label="Model Performance", children=[
                html.Div(
                    children=[
                        html.H5("Understanding Model Performance Metrics", className="mt-4"),
                        html.P(
                            ["For our imbalanced data, **Accuracy** (the percentage of correct predictions) isn't the best metric. We must focus on a more comprehensive set of metrics:",
                             html.Ul(
                                 [
                                     html.Li(
                                         [
                                             html.B("Precision:"),
                                             " Of all the customers our model predicted would churn, what percentage actually did? High precision means fewer false alarms."
                                         ]
                                     ),
                                     html.Li(
                                         [
                                             html.B("Recall:"),
                                             " Of all the customers who actually churned, what percentage did our model successfully identify? High recall means our model is great at catching churners."
                                         ]
                                     ),
                                     html.Li(
                                         [
                                             html.B("F1-Score:"),
                                             " A single score that balances both precision and recall. It's a great overall measure."
                                         ]
                                     ),
                                     html.Li(
                                         [
                                             html.B("ROC-AUC:"),
                                             " Measures how well the model distinguishes between churners and non-churners. A score of 1.0 is perfect, while 0.5 is no better than a random guess."
                                         ]
                                     )
                                 ]
                             )
                            ]
                        ),
                        dbc.Row([dbc.Col(dcc.Graph(id="model-metrics-bar"), md=12)]),
                        html.P("The chart above shows the performance of our models. We're looking for a model that has a good balance of high scores, particularly for Recall and Precision on the churn class."),
                        dcc.Dropdown(
                            id="model-selector",
                            options=[{'label': name, 'value': name} for name in metrics_df['Model']],
                            value='Random Forest'
                        ),
                        dbc.Row([
                            dbc.Col(dcc.Graph(id="confusion-matrix-plot"), md=6),
                            dbc.Col(dcc.Graph(id="roc-curve-plot"), md=6),
                        ]),
                    ], className="p-4"
                )
            ]),
        ])
    ]
)

# 4. ACT Tab
act_tab = dcc.Markdown(
    """
    ### 🚀 **ACT** — What to Do Next

    This section translates our data insights and model performance into a clear, actionable business strategy.

    **Key Findings**:
    - Our analysis shows that older customers, those with higher balances, and female customers are at a higher risk of churning.
    - The most effective model is the **Random Forest Classifier**, which achieved the best balance between identifying actual churners (high recall) and minimizing false alarms (high precision).
    
    **Actionable Recommendations**:

    1. **Proactive Customer Retention:** Use the **Random Forest** model to generate a daily list of customers with a high probability of churning. The customer service team can then proactively reach out to these individuals with personalized offers or to address any dissatisfaction.

    2. **Targeted Campaigns:** Develop specific marketing campaigns to address the identified churn drivers. For example, create special promotions for older customers or a "High-Value Customer" program for those with large balances to increase their sense of loyalty.

    3. **Continuous Improvement:** Regularly retrain the model with new data to ensure its predictions remain accurate over time. Monitoring the model's performance on a dashboard like this is crucial for its long-term success.
    """, className="p-4"
)

app.layout = dbc.Container(
    [
        header,
        dbc.Tabs(
            [
                dbc.Tab(ask_tab, label="Ask"),
                dbc.Tab(prepare_tab, label="Prepare"),
                dbc.Tab(analyze_tab, label="Analyze"),
                dbc.Tab(act_tab, label="Act")
            ]
        ),
    ],
    fluid=True,
)


# --- Callbacks ---
@app.callback(
    Output("correlation-matrix", "figure"),
    Input("churn-pie-chart", "id") # Dummy input to trigger on load
)
def update_corr_matrix(dummy):
    correlation = df_processed.corr()
    fig = ff.create_annotated_heatmap(
        z=correlation.values.round(2),
        x=list(correlation.columns),
        y=list(correlation.index),
        colorscale="Viridis",
        showscale=True,
        reversescale=True
    )
    fig.update_layout(title="Correlation Matrix", height=500, margin=dict(t=50, b=50))
    return fig


@app.callback(
    Output("categorical-churn-plot", "figure"),
    Input("churn-pie-chart", "id")
)
def update_categorical_plot(dummy):
    fig = go.Figure()
    
    # Geography
    geo_data = df_processed.groupby(['Geography_Germany', 'Exited']).size().reset_index(name='count')
    geo_data['label'] = geo_data['Geography_Germany'].map({1: 'Germany', -1: 'Other'})
    fig.add_trace(go.Bar(x=geo_data.loc[geo_data.Exited == 1, 'label'], y=geo_data.loc[geo_data.Exited == 1, 'count'], name='Exited'))
    fig.add_trace(go.Bar(x=geo_data.loc[geo_data.Exited == 0, 'label'], y=geo_data.loc[geo_data.Exited == 0, 'count'], name='Retained'))
    fig.update_layout(barmode='group', title='Churn by Geography', height=400, margin=dict(t=50, b=50))
    return fig


@app.callback(
    Output("continuous-churn-plot", "figure"),
    Input("churn-pie-chart", "id")
)
def update_continuous_plot(dummy):
    fig = go.Figure()
    
    # Box plots for Age and Balance
    for var in ['Age', 'Balance']:
        retained = df_processed.loc[df_processed['Exited'] == 0, var]
        exited = df_processed.loc[df_processed['Exited'] == 1, var]
        fig.add_trace(go.Box(y=retained, name=f'{var} (Retained)', marker_color='blue'))
        fig.add_trace(go.Box(y=exited, name=f'{var} (Exited)', marker_color='red'))
    
    fig.update_layout(
        title="Distribution of Age and Balance for Churned vs. Retained Customers",
        yaxis_title="Scaled Value",
        height=400,
        margin=dict(t=50, b=50),
        boxmode='group',
        boxgroupgap=0.5
    )
    return fig


@app.callback(
    Output("model-metrics-bar", "figure"),
    Output("confusion-matrix-plot", "figure"),
    Output("roc-curve-plot", "figure"),
    Input("model-selector", "value")
)
def update_model_performance_plots(selected_model_name):
    # Bar chart for all model metrics
    metrics_bar = go.Figure()
    for metric in ['Accuracy', 'Recall', 'Precision', 'F1-Score', 'ROC-AUC']:
        metrics_bar.add_trace(go.Bar(
            y=metrics_df['Model'],
            x=metrics_df[metric],
            orientation='h',
            name=metric
        ))
    metrics_bar.update_layout(
        barmode='group',
        title="Model Performance Metrics (Test Data)",
        height=450,
        margin=dict(l=150, r=20, t=50, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    # Plots for the selected model
    selected_model_data = metrics_df[metrics_df['Model'] == selected_model_name].iloc[0]
    predictions = selected_model_data['predictions']
    probabilities = selected_model_data['probabilities']
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, predictions)
    cm_fig = ff.create_annotated_heatmap(
        z=cm, x=["Retained", "Exited"], y=["Retained", "Exited"],
        colorscale='blues',
        showscale=False
    )
    cm_fig.update_layout(title=f"Confusion Matrix ({selected_model_name})", height=450, margin=dict(t=50, b=50))

    # ROC Curve
    if isinstance(probabilities, np.ndarray):
        fpr, tpr, _ = roc_curve(y_test, probabilities)
        roc_fig = go.Figure(data=[
            go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC Curve'),
            go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash'), name='Random Guess')
        ])
        roc_fig.update_layout(
            title=f"ROC Curve ({selected_model_name})",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            height=450,
            margin=dict(t=50, b=50)
        )
    else:
        roc_fig = go.Figure(go.Scatter())
        roc_fig.update_layout(title="ROC Curve Not Available", height=450, margin=dict(t=50, b=50))

    return metrics_bar, cm_fig, roc_fig


if __name__ == "__main__":
    app.run(debug=True)