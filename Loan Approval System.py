import tkinter as tk
from tkinter import messagebox
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier  # Neural Network algoritham
from sklearn.preprocessing import StandardScaler


# Data load and model training
rawdata = {
    'income': [50000, 100000, 150000, 200000, 80000, 45000, 120000, 300000, 40000, 60000],
    'loan': [500000, 4000000, 5000000, 8000000, 1000000, 3000000, 4000000, 15000000, 2000000, 1200000],
    'score': [700, 750, 800, 780, 680, 600, 720, 850, 550, 650],
    'years': [5, 10, 8, 10, 5, 7, 6, 10, 5, 6],
    'status': [1, 1, 1, 1, 1, 0, 1, 1, 0, 1]
}

data = pd.DataFrame(rawdata)
input_x = data.drop('status', axis=1)
input_y = data['status']

scaler = StandardScaler()
scaled_x = scaler.fit_transform(input_x.values)

#Activation ReLU
loan_model = MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=2000)
loan_model.fit(scaled_x, input_y)

def get_emi(*args):# Equated Monthly Installment 
    try:
        user_amount = entry_loan.get()
        yrs_str = entry_years.get()
        if user_amount and yrs_str:
            amount = float(user_amount)
            yrs = int(yrs_str)
            if 5 <= yrs <= 10:
                rate = 0.12 / 12
                months = yrs * 12
                result = (amount * rate * (1 + rate) ** months) / ((1 + rate) ** months - 1)
                entry_monthly.delete(0, tk.END)
                entry_monthly.insert(0, str(round(result, 2)))
    except:
        entry_monthly.delete(0, tk.END)

def process_check():
    try:
        inc = float(entry_income.get())
        amount = float(entry_loan.get())
        yrs = int(entry_years.get())
        emi = float(entry_monthly.get())

        if not (5 <= yrs <= 10):
            messagebox.showwarning("Error", "Years must be 5-10")
            return

        limit = inc * 0.45
        if emi > limit:
            rate = 0.12 / 12
            months = yrs * 12
            max_amount = (limit * ((1 + rate) ** months - 1)) / (rate * (1 + rate) ** months)
            messagebox.showinfo("Result", f"Rejected: EMI too high!\nLimit: Rs. {round(limit,2)}\nMax Loan: Rs. {round(max_amount,2)}")
        else:
            # Using 700 as default credit score
            inp = scaler.transform([[inc, amount, 700, yrs]])
            out = loan_model.predict(inp)

            if out[0] == 1:
                messagebox.showinfo("Result", "Success: Loan Approved!")
            else:
                messagebox.showinfo("Result", "Rejected: Risk Profile too high")
    except Exception as e:
        messagebox.showerror("Error", "Please fill all fields correctly.")

def reset():
    entry_income.delete(0, tk.END)
    entry_loan.delete(0, tk.END)
    entry_years.delete(0, tk.END)
    entry_monthly.delete(0, tk.END)

# GUI Layout Setup
root = tk.Tk()
root.title("Neural Network Loan Checker")
root.geometry("500x500")
root.configure(padx=25, pady=20, bg="#82D5E0")

tk.Label(root, text="Loan Approval Predictor System",
         font=("Segoe UI", 20, "bold", "underline"),
         bg="#82D5E0", fg="#000000").pack(pady=25)

loan_var, year_var = tk.StringVar(), tk.StringVar()
loan_var.trace_add("write", get_emi)
year_var.trace_add("write", get_emi)

form_frame = tk.Frame(root, bg="#82D5E0")
form_frame.pack(pady=10)

# Monthly Income
tk.Label(form_frame, text="Monthly Income (Rs.)", font=("Berlin Sans FB", 14), bg="#82D5E0").grid(row=0, column=0, sticky="w", pady=8)
entry_income = tk.Entry(form_frame, font=("Berlin Sans FB", 14))
entry_income.grid(row=0, column=1, padx=10, pady=8)

# Loan Amount
tk.Label(form_frame, text="Loan Amount", font=("Berlin Sans FB", 14), bg="#82D5E0").grid(row=1, column=0, sticky="w", pady=8)
entry_loan = tk.Entry(form_frame, textvariable=loan_var, font=("Berlin Sans FB", 14))
entry_loan.grid(row=1, column=1, padx=10, pady=8)

# Years
tk.Label(form_frame, text="Year (Duration 5 -10)", font=("Berlin Sans FB", 14), bg="#82D5E0").grid(row=2, column=0, sticky="w", pady=8)
entry_years = tk.Entry(form_frame, textvariable=year_var, font=("Berlin Sans FB", 14))
entry_years.grid(row=2, column=1, padx=10, pady=8)

# Monthly EMI
tk.Label(form_frame, text="Monthly EMI", font=("Berlin Sans FB", 14), bg="#82D5E0").grid(row=3, column=0, sticky="w", pady=8)
entry_monthly = tk.Entry(form_frame, font=("Berlin Sans FB", 14), fg="blue")
entry_monthly.grid(row=3, column=1, padx=10, pady=8)

btn_frame = tk.Frame(root, bg="#82D5E0")
btn_frame.pack(pady=20)

tk.Button(btn_frame, text="Check Approval", command=process_check,
          bg="#27C461", fg="white", font=("Segoe UI", 14, "bold"), width=31).grid(row=0, column=0, padx=16)

tk.Button(btn_frame, text="Clear", command=reset,
          bg="#E00A21", fg="white", font=("Segoe UI", 14, "bold"), width=31).grid(row=1, column=0, padx=10, pady=8)

root.mainloop()