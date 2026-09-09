import pandas as pd
import matplotlib.pyplot as plt
import os
from io import StringIO


# Load the data from a CSV-formatted string (replace with file reading if needed)
data = '''
Level,CAMON,COELA,Embodied,HMAS-2,Do Nothing,Wildfire Alg
Cut Trees: Sparse (Small),"18.00±0.00","17.00±5.00","14.60±3.00","18.00±0.00","0.00±0.00",18
Cut Trees: Sparse (Large),"72.00±4.00","57.67±25.00","50.33±9.00","56.33±13.00","0.00±0.00",75
Cut Trees: Lines (Small),"29.33±2.00","27.33±4.00","28.67±4.00","28.67±4.00","0.00±0.00",30
Cut Trees: Lines (Large),"94.33±7.00","83.67±24.00","90.33±1.00","90.33±6.00","0.00±0.00",105
Scout Fire (Small),"0.00±0.00","0.00±0.00","0.00±0.00","0.00±0.00","0.00±0.00",2
Scout Fire (Large),"0.00±0.00","0.00±0.00","0.67±2.00","0.67±2.00","0.00±0.00",0
Transport Firefighters (Small),"6.00±0.00","3.00±5.00","4.60±4.00","4.40±5.00","0.00±0.00",6
Transport Firefighters (Large),"10.00±0.00","8.33±5.00","9.33±2.00","8.33±5.00","0.00±0.00",10
Rescue Civilians: Known Location (Small),"1.60±2.00","0.00±0.00","1.60±3.00","1.40±1.00","0.00±0.00",2
Rescue Civilians: Search and Rescue,"0.00±0.00","0.00±0.00","0.00±0.00","0.00±0.00","0.00±0.00",2
Rescue Civilians: Search + Rescue + Transport,"0.00±0.00","0.00±0.00","0.00±0.00","0.00±0.00","0.00±0.00",0
'''

# Read into DataFrame
df = pd.read_csv(StringIO(data))

# Extract mean values from the baseline columns for plotting
def extract_mean(score):
    if isinstance(score, str) and '±' in score:
        return float(score.split('±')[0])
    return float(score)

# Create output directory
os.makedirs("plots", exist_ok=True)

# Plot each level
for idx, row in df.iterrows():
    level = row['Level']
    scores = {alg: extract_mean(row[alg]) for alg in df.columns[1:]}

    plt.figure(figsize=(8, 5))
    plt.bar(scores.keys(), scores.values(), color='skyblue')
    plt.title(level, fontsize=10)
    plt.ylabel('Score')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    filename = f"plots/{level.replace(':','-').replace(' ','_').replace('/','_')}.png"
    plt.savefig(filename)
    plt.close()

print("Plots saved in the 'plots' directory.")
