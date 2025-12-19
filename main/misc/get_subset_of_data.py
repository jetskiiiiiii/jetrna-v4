val_seq_df = pd.read_csv("/content/validation_sequences.csv")
val_lab_df = pd.read_csv("/content/validation_labels.csv")

train_seq_sample = val_seq_df[:8]
val_seq_sample = val_seq_df[8:10]
test_seq_sample = val_seq_df[10:]

train_lab_sample = pd.DataFrame()
val_lab_sample = pd.DataFrame()
test_lab_sample = pd.DataFrame()

for row in train_seq_sample.itertuples(index=False):
  name = row.target_id
  labels = val_lab_df[val_lab_df["ID"].str.startswith(name)]
  train_lab_sample = pd.concat([train_lab_sample, labels], ignore_index=True)

for row in val_seq_sample.itertuples(index=False):
  name = row.target_id
  labels = val_lab_df[val_lab_df["ID"].str.startswith(name)]
  val_lab_sample = pd.concat([val_lab_sample, labels], ignore_index=True)

for row in test_seq_sample.itertuples(index=False):
  name = row.target_id
  labels = val_lab_df[val_lab_df["ID"].str.startswith(name)]
  test_lab_sample = pd.concat([test_lab_sample, labels], ignore_index=True)

train_seq_sample.to_csv("train_seq_sample.csv")
val_seq_sample.to_csv("val_seq_sample.csv")
test_seq_sample.to_csv("test_seq_sample.csv")

train_lab_sample.to_csv("/train_lab_sample.csv")
val_lab_sample.to_csv("/val_lab_sample.csv")
test_lab_sample.to_csv("/test_lab_sample.csv")
