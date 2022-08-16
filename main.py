import csv
import pandas as pd
# with open("fire_hd_reviews_pre_cleaned.csv") as csv_file:
#     csv_reader = csv.reader(csv_file, delimiter= ",")
#     for row in csv_reader:
#         print(row)

reviews_df = pd.read_csv("fire_hd_reviews_pre_cleaned.csv", on_bad_lines='skip')
print("Dataframe shape:", reviews_df.shape)










if __name__ == '__main__':
    pass
