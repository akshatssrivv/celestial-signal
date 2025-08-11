import pandas as pd
from ai_explainer_utils import format_bond_diagnostics

final_df = pd.read_csv("final_signal.csv")  

sample_row = final_df.iloc[0]
diagnostic_dict = format_bond_diagnostics(sample_row)
print(diagnostic_dict)
