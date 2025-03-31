# copy this code to a app.py file inside the folder that will be uploaded to Hugging Face
import joblib
from pathlib import Path
import gradio as gr
import json
import pandas as pd
    
# load model 
model = joblib.load("best_model.pkl")

final_columns = ['U_DEM_AGE',
 'B_JOB_CAT_NEW3',
 'M_ED_MR_MAJOR_ED_CAT_NEW2',
 'L_ED_HS_SCHOOL_ST_CTRY_CD',
 'E_JOB_EMPLR_SIZE',
 'E_JOB_EMPLR_NON_EDUC_INST_TYPE'] # numeric_cols+categorical_cols

# load info dict containing selection categories of the selected columns
with open("info_dict.json", "r") as json_file:
    info_dict = json.load(json_file)


# Define the prediction function
def predict_salary(*inputs):
    # Convert inputs into a DataFrame
    input_data = {name: [value] for name, value in zip(final_columns, inputs)}
    df = pd.DataFrame(input_data)

    # Transform data before prediction
    df_transformed = model.named_steps['preprocessor'].transform(df)
    prediction = model.named_steps['model'].predict(df_transformed)

    return f"Predicted Salary: ${prediction[0]:,.2f}"


inputs = (
    [gr.Number(label=info_dict[final_columns[0]]["DESCRIPTION"], value=23)]+
    [gr.Dropdown(choices=info_dict[column]["Choices"], label=f'{info_dict[column]["DESCRIPTION"]} - (Actual or expected in the next three months)') for column in final_columns[1:]]  # Drop-downs
)



app = gr.Interface(
    fn=predict_salary,
    inputs=inputs,
    outputs="text",
    title="Salary Prediction App",
    description="Enter your details to predict the loan applicant potential salary. If something does not apply to the applicant select 'Not applicable' or the most possible option to happen in the next months."
)

if __name__ == "__main__":
    app.launch()