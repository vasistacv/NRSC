import pandas as pd
df = pd.read_csv(r"D:\NEW_NRSC\Final_ground_truth_data.csv")
df["Date"] = pd.to_datetime(df["Date"])
df_2023 = df[(df["Date"].dt.year == 2023) & (df["Date"].dt.month.isin([6,7,8,9]))]
print(f"Max rainfall in 2023 monsoon: {df_2023['Rainfall'].max():.1f} mm")
print(f"\nPer-station max:")
for stn in df_2023["Station"].unique():
    mx = df_2023[df_2023["Station"]==stn]["Rainfall"].max()
    print(f"  {stn}: {mx:.1f} mm")
