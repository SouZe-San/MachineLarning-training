from tkinter import *
import joblib as jb
import numpy as np


# Load Model
model = jb.load('./Logistic_regression_model.pkl')

# Create APp
root = Tk()


def prediction():
    input_data = [PregnanciesValue, GlucoseValue, BloodPressureValue,
                  SkinThicknessValue, InsulinValue, DiabetesPedigreeFunctionValue, AgeValue]

    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    prediction = model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 0):
        result.config(text="The person is not diabetic")

    else:
        result.config(text="The person is diabetic")


root.geometry("644x344")
# Heading
Label(root, text="Have I Diabetes ?",
      font="Arial 13 bold", pady=15).grid(row=0, column=3)

# Text for our form

Pregnancies = Label(root, text='Number of Pregnancies')
Glucose = Label(root, text='Glucose Level')
BloodPressure = Label(root, text='Blood Pressure value')
SkinThickness = Label(root, text='Skin Thickness value')
Insulin = Label(root, text='Insulin Level')
BMI = Label(root, text='BMI value')
DiabetesPedigreeFunction = Label(root, text='Diabetes Pedigree Function value')
Age = Label(root, text='Age of the Person')
result_label = Label(root, text="Result")
result = Label(root, text="")
# Pack text for our form

Pregnancies.grid(row=1, column=2)
Glucose.grid(row=2, column=2)
BloodPressure.grid(row=3, column=2)
SkinThickness.grid(row=4, column=2)
Insulin.grid(row=5, column=2)
BMI.grid(row=6, column=2)
DiabetesPedigreeFunction.grid(row=7, column=2)
Age.grid(row=8, column=2)
result_label.grid(row=9, column=2)


# Tkinter variable for storing entries
PregnanciesValue = StringVar()
GlucoseValue = StringVar()
BloodPressureValue = StringVar()
SkinThicknessValue = StringVar()
InsulinValue = StringVar()
BMIValue = StringVar()
DiabetesPedigreeFunctionValue = StringVar()
AgeValue = StringVar()


# Entries for our form

PregnanciesEntry = Entry(root, textvariable=PregnanciesValue)
GlucoseEntry = Entry(root, textvariable=GlucoseValue)
BloodPressureEntry = Entry(root, textvariable=BloodPressureValue)
SkinThicknessEntry = Entry(root, textvariable=SkinThicknessValue)
InsulinEntry = Entry(root, textvariable=InsulinValue)
BMIEntry = Entry(root, textvariable=BMIValue)
DiabetesPedigreeFunctionEntry = Entry(
    root, textvariable=DiabetesPedigreeFunctionValue)
AgeEntry = Entry(root, textvariable=AgeValue)


# Packing the Entries
PregnanciesEntry.grid(row=1, column=3)
GlucoseEntry.grid(row=2, column=3)
BloodPressureEntry.grid(row=3, column=3)
SkinThicknessEntry.grid(row=4, column=3)
InsulinEntry.grid(row=5, column=3)
BMIEntry.grid(row=6, column=3)
DiabetesPedigreeFunctionEntry.grid(row=7, column=3)
AgeEntry.grid(row=8, column=3)
result.grid(row=9, column=3)


# Button & packing it and assigning it a command
Button(text="Check", command=prediction).grid(row=10, column=3)


root.mainloop()
