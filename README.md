# AttendIQ

AttendIQ is a Streamlit dashboard for student attendance analytics and academic risk prediction. It includes the sample dataset at `data/student.attendence.csv`.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Push this folder to a GitHub repository.
2. In [Streamlit Community Cloud](https://share.streamlit.io/), choose **Create app** and connect that repository.
3. Select the branch you pushed and set the app file path to `app.py`.
4. Choose **Deploy**. The dependencies are listed in `requirements.txt`.

The sample data is synthetic. Replace `data/student.attendence.csv` with your approved dataset before publishing if you want the live app to show real records. Do not commit private student data or credentials to a public repository.
