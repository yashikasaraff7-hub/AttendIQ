import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="AttendIQ",
    page_icon="🎓",
    layout="wide"
)

# ==========================================
# 2. CUSTOM DARK THEME
# ==========================================

st.markdown("""
<style>

.stApp {
    background-color: #0B1120;
    color: white;
}

[data-testid="stSidebar"] {
    background-color: #111827;
}

[data-testid="stMetric"] {
    background-color: #1E293B;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #334155;
}

h1, h2, h3 {
    color: white;
}

p, label {
    color: #CBD5E1;
}

.stButton button {
    background-color: #3B82F6;
    color: white;
    border-radius: 8px;
}

.stButton button:hover {
    background-color: #2563EB;
    color: white;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# 3. LOAD DATASET
# ==========================================

DATA_PATH = Path(__file__).parent / "data" / "student.attendence.csv"


def generate_sample_data():

    np.random.seed(42)

    n = 808

    df = pd.DataFrame({

        "student_id": [f"STU{i:04d}" for i in range(1, n + 1)],

        "name": [f"Student {i}" for i in range(1, n + 1)],

        "gender": np.random.choice(
            ["Male", "Female"], n
        ),

        "age": np.random.randint(17, 25, n),

        "branch": np.random.choice(
            ["CSE", "ECE", "Mechanical", "Civil", "BBA"],
            n
        ),

        "study_hours": np.random.uniform(2, 20, n).round(1),

        "failures": np.random.choice(
            [0, 1, 2, 3], n,
            p=[0.70, 0.18, 0.09, 0.03]
        ),

        "travel_time": np.random.randint(5, 120, n),

        "participation": np.random.randint(1, 6, n),

        "gpa": np.random.uniform(4, 10, n).round(2),

        "sleep": np.random.uniform(4, 10, n).round(1),

    })

    df["attendance_pct"] = np.clip(
        np.random.normal(76, 16, n),
        25,
        100
    ).round(2)

    df["at_risk"] = np.where(
        df["attendance_pct"] < 75,
        1,
        0
    )

    df["risk_status"] = df["at_risk"].map({
        0: "Safe",
        1: "At Risk"
    })

    return df


@st.cache_data
def load_data():

    if DATA_PATH.exists() and DATA_PATH.stat().st_size > 0:

        try:
            data = pd.read_csv(DATA_PATH)

            if data.empty:
                raise ValueError("Dataset is empty")

            return data, False

        except Exception:
            pass

    data = generate_sample_data()

    DATA_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    data.to_csv(DATA_PATH, index=False)

    return data, True


df, using_sample_data = load_data()

# ==========================================
# 4. DATA CLEANING
# ==========================================

df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
)

if "attendance_pct" not in df.columns:

    for col in ["attendance_percentage", "attendance", "attendance_percent"]:

        if col in df.columns:
            df.rename(
                columns={col: "attendance_pct"},
                inplace=True
            )
            break

if "attendance_pct" not in df.columns:

    st.error(
        "Attendance percentage column not found. "
        "Please check your CSV column names."
    )

    st.stop()


df["attendance_pct"] = pd.to_numeric(
    df["attendance_pct"],
    errors="coerce"
)

df.dropna(
    subset=["attendance_pct"],
    inplace=True
)

df["at_risk"] = (
    df["attendance_pct"] < 75
).astype(int)

df["risk_status"] = df["at_risk"].map({
    0: "Safe",
    1: "At Risk"
})


# ==========================================
# 5. SIDEBAR NAVIGATION
# ==========================================

st.sidebar.markdown("# 🎓 AttendIQ")

st.sidebar.caption(
    "STUDENT ATTENDANCE ANALYTICS"
)

st.sidebar.divider()

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Dashboard",
        "Student Records",
        "Attendance Analytics",
        "At-Risk Prediction",
        "Model Performance",
        "About Project"
    ]
)

st.sidebar.divider()

st.sidebar.caption("Machine Learning Mini Project")
st.sidebar.caption("Version 2.0")

st.sidebar.success("Dataset Loaded Successfully")


# ==========================================
# 6. DASHBOARD
# ==========================================

if page == "Dashboard":

    st.title("🎓 AttendIQ")

    st.subheader(
        "Student Attendance Analysis & Risk Prediction"
    )

    st.write(
        "Monitor attendance, analyze patterns "
        "and identify students at academic risk."
    )

    st.divider()

    total = len(df)

    average = df["attendance_pct"].mean()

    at_risk = (df["attendance_pct"] < 75).sum()

    safe = total - at_risk

    st.markdown("### 📊 Dashboard Overview")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Students", total)

    c2.metric("Average Attendance", f"{average:.2f}%")

    c3.metric("At-Risk Students", int(at_risk))

    c4.metric("Safe Students", int(safe))

    st.divider()

    left, right = st.columns(2)

    with left:

        st.markdown("### Attendance Overview")

        pie_data = pd.DataFrame({
            "Category": ["Safe", "At Risk"],
            "Students": [safe, at_risk]
        })

        fig = px.pie(
            pie_data,
            names="Category",
            values="Students",
            hole=0.55,
            color="Category",
            color_discrete_map={
                "Safe": "#22C55E",
                "At Risk": "#EF4444"
            }
        )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0B1120"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with right:

        st.markdown("### Attendance Distribution")

        fig = px.histogram(
            df,
            x="attendance_pct",
            nbins=20,
            color_discrete_sequence=["#3B82F6"]
        )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0B1120",
            xaxis_title="Attendance Percentage",
            yaxis_title="Number of Students"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.divider()

    st.markdown("### Department-wise Attendance")

    if "branch" in df.columns:

        branch_data = (
            df.groupby("branch")["attendance_pct"]
            .mean()
            .reset_index()
        )

        fig = px.bar(
            branch_data,
            x="branch",
            y="attendance_pct",
            color="attendance_pct",
            text_auto=".2f"
        )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0B1120",
            xaxis_title="Department",
            yaxis_title="Average Attendance (%)"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    if using_sample_data:

        st.warning(
            "You are currently viewing synthetic sample data. "
            "Replace it with your actual student dataset."
        )


# ==========================================
# 7. STUDENT RECORDS
# ==========================================

elif page == "Student Records":

    st.title("👨‍🎓 Student Records")

    st.write(
        "Explore and filter student attendance records."
    )

    search = st.text_input(
        "🔍 Search Student"
    )

    filtered = df.copy()

    if "name" in filtered.columns and search:

        filtered = filtered[
            filtered["name"].astype(str).str.contains(
                search,
                case=False,
                na=False
            )
        ]

    if "branch" in filtered.columns:

        branches = st.multiselect(
            "Select Department",
            filtered["branch"].dropna().unique()
        )

        if branches:

            filtered = filtered[
                filtered["branch"].isin(branches)
            ]

    status = st.selectbox(
        "Attendance Status",
        ["All", "Safe", "At Risk"]
    )

    if status != "All":

        filtered = filtered[
            filtered["risk_status"] == status
        ]

    st.metric(
        "Records Found",
        len(filtered)
    )

    st.dataframe(
        filtered,
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "Download Student Records",
        data=filtered.to_csv(index=False),
        file_name="student_records.csv",
        mime="text/csv"
    )


# ==========================================
# 8. ATTENDANCE ANALYTICS
# ==========================================

elif page == "Attendance Analytics":

    st.title("📈 Attendance Analytics")

    st.write(
        "Explore detailed attendance patterns."
    )

    st.divider()

    st.subheader("Attendance Distribution")

    fig = px.histogram(
        df,
        x="attendance_pct",
        nbins=20,
        color_discrete_sequence=["#3B82F6"]
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0B1120"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.divider()

    if "branch" in df.columns:

        st.subheader("Department-wise Attendance")

        branch_data = (
            df.groupby("branch")["attendance_pct"]
            .mean()
            .reset_index()
        )

        fig = px.bar(
            branch_data,
            x="branch",
            y="attendance_pct",
            color="branch",
            text_auto=".2f"
        )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0B1120",
            showlegend=False
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.divider()

    st.subheader("Attendance Categories")

    category_data = pd.DataFrame({
        "Category": [
            "Below 60%",
            "60–75%",
            "75–90%",
            "Above 90%"
        ],
        "Students": [
            ((df["attendance_pct"] < 60)).sum(),

            ((df["attendance_pct"] >= 60) &
             (df["attendance_pct"] < 75)).sum(),

            ((df["attendance_pct"] >= 75) &
             (df["attendance_pct"] < 90)).sum(),

            (df["attendance_pct"] >= 90).sum()
        ]
    })

    fig = px.bar(
        category_data,
        x="Category",
        y="Students",
        color="Category",
        text_auto=True
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0B1120",
        showlegend=False
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.divider()

    st.subheader("Attendance Statistics")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Average",
        f"{df['attendance_pct'].mean():.2f}%"
    )

    c2.metric(
        "Highest",
        f"{df['attendance_pct'].max():.2f}%"
    )

    c3.metric(
        "Lowest",
        f"{df['attendance_pct'].min():.2f}%"
    )


# ==========================================
# 9. AT-RISK PREDICTION
# ==========================================

elif page == "At-Risk Prediction":

    st.title("⚠️ At-Risk Student Prediction")

    st.write(
        "Enter student attendance details to check academic risk."
    )

    st.divider()

    with st.form("prediction_form"):

        attendance = st.number_input(
            "Attendance Percentage",
            min_value=0.0,
            max_value=100.0,
            value=75.0
        )

        study_hours = st.number_input(
            "Daily Study Hours",
            min_value=0.0,
            max_value=24.0,
            value=5.0
        )

        branch = st.selectbox(
            "Department",
            ["CSE", "ECE", "Mechanical", "Civil", "BBA"]
        )

        submitted = st.form_submit_button(
            "Analyze Student Risk"
        )

    if submitted:

        if attendance < 75:

            st.error("⚠️ Student is At Risk")

            st.write(
                "The student's attendance is below 75%."
            )

        else:

            st.success("✅ Student is Safe")

            st.write(
                "The student's attendance meets the 75% requirement."
            )

        st.progress(
            int(attendance)
        )

        st.metric(
            "Attendance Percentage",
            f"{attendance}%"
        )

    st.caption(
        "Note: This is rule-based attendance checking, "
        "not a trained ML prediction."
    )


# ==========================================
# 10. MODEL PERFORMANCE
# ==========================================

elif page == "Model Performance":

    st.title("🤖 Machine Learning Model Performance")

    st.write(
        "Compare Logistic Regression, Decision Tree "
        "and Random Forest."
    )

    # Exclude attendance percentage to avoid
    # directly giving the model the target rule.

    exclude = [
        "at_risk",
        "risk_status",
        "attendance_pct",
        "name",
        "student_id"
    ]

    features = [
        col for col in df.columns
        if col not in exclude
    ]

    if len(features) == 0:

        st.warning(
            "Not enough predictor columns for model training."
        )

    elif df["at_risk"].nunique() < 2:

        st.warning(
            "Both Safe and At-Risk students are needed "
            "for model evaluation."
        )

    else:

        X = df[features]

        y = df["at_risk"]

        numeric_features = X.select_dtypes(
            include=np.number
        ).columns.tolist()

        categorical_features = X.select_dtypes(
            exclude=np.number
        ).columns.tolist()

        numeric_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

        categorical_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ])

        preprocessor = ColumnTransformer([
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ])

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        models = {
            "Logistic Regression": LogisticRegression(
                max_iter=1000
            ),

            "Decision Tree": DecisionTreeClassifier(
                max_depth=5,
                random_state=42
            ),

            "Random Forest": RandomForestClassifier(
                n_estimators=100,
                random_state=42
            )
        }

        results = []

        predictions = {}

        for name, model in models.items():

            pipeline = Pipeline([
                ("preprocessor", preprocessor),
                ("model", model)
            ])

            pipeline.fit(X_train, y_train)

            pred = pipeline.predict(X_test)

            predictions[name] = pred

            results.append({

                "Model": name,

                "Accuracy": accuracy_score(
                    y_test, pred
                ),

                "Precision": precision_score(
                    y_test, pred, zero_division=0
                ),

                "Recall": recall_score(
                    y_test, pred, zero_division=0
                ),

                "F1 Score": f1_score(
                    y_test, pred, zero_division=0
                )
            })

        results_df = pd.DataFrame(results)

        st.subheader("Model Comparison")

        st.dataframe(
            results_df.style.format({
                "Accuracy": "{:.2%}",
                "Precision": "{:.2%}",
                "Recall": "{:.2%}",
                "F1 Score": "{:.2%}"
            }),
            use_container_width=True
        )

        st.divider()

        st.subheader("Model Accuracy Comparison")

        fig = px.bar(
            results_df,
            x="Model",
            y="Accuracy",
            color="Model",
            text_auto=".2%"
        )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0B1120"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        selected_model = st.selectbox(
            "Select Model for Confusion Matrix",
            list(models.keys())
        )

        cm = confusion_matrix(
            y_test,
            predictions[selected_model],
            labels=[0, 1]
        )

        fig = px.imshow(
            cm,
            text_auto=True,
            x=["Safe", "At Risk"],
            y=["Safe", "At Risk"],
            color_continuous_scale="Blues",
            title=f"{selected_model} Confusion Matrix"
        )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0B1120"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.info(
            "Models are evaluated on a held-out test set. "
            "Results depend on the dataset and its features."
        )


# ==========================================
# 11. ABOUT PROJECT
# ==========================================

elif page == "About Project":

    st.title("ℹ️ About AttendIQ")

    st.subheader("Project Overview")

    st.write("""
    AttendIQ is a Machine Learning based Student
    Attendance Analysis and Risk Prediction platform.

    It analyzes student-related data, visualizes
    attendance patterns and identifies students
    who may be at risk of low attendance.
    """)

    st.divider()

    st.subheader("Technologies Used")

    st.write("""
    - Python
    - Pandas
    - NumPy
    - Scikit-learn
    - Plotly
    - Streamlit
    """)

    st.divider()

    st.subheader("Machine Learning Algorithms")

    st.write("""
    1. Logistic Regression
    2. Decision Tree
    3. Random Forest
    """)

    st.divider()

    st.subheader("Project Objectives")

    st.write("""
    - Analyze student attendance.
    - Visualize attendance trends.
    - Identify at-risk students.
    - Compare machine learning algorithms.
    - Support data-driven academic decisions.
    """)

    st.success(
        "AttendIQ - Making Student Attendance Smarter."
    )