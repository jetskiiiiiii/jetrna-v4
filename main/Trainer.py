import lightning as L

from Model import jetRNA_v4_Model
from DataModule import jetRNA_v4_DataModule

IN_DIM = 28 
HIDDEN_DIM = 32
OUT_DIM = 3
LEARNING_RATE = 5e-5
BATCH_SIZE = 8 # Batches 2 variable-sized graphs together

EPOCHS = 20

path_to_train_sequences_csv = "./dataset/stanford/train_sequences.csv"
path_to_train_embeddings = "./dataset/stanford/single_representation/train"
path_to_val_sequences_csv = "./dataset/stanford/validation_sequences.csv"
path_to_val_embeddings = "./dataset/stanford/single_representation/val"
path_to_test_sequences_csv = "./dataset/stanford/test_sequences.csv"
path_to_test_embeddings = "./dataset/stanford/single_representation/test"
path_to_train_labels_csv = "./dataset/stanford/train_labels.csv"
path_to_val_labels_csv = "./dataset/stanford/validation_labels.csv"

# Smaller subset of data
#path_to_train_sequences_csv = "./dataset/sample/train_seq_sample.csv"
#path_to_val_sequences_csv = "./dataset/sample/val_seq_sample.csv"
#path_to_test_sequences_csv = "./dataset/sample/test_seq_sample.csv"
#path_to_train_labels_csv = "./dataset/sample/train_lab_sample.csv"
#path_to_val_labels_csv = "./dataset/sample/val_lab_sample.csv"


if __name__ == "__main__":
    data_module = jetRNA_v4_DataModule(
        path_to_train_sequences_csv,
        path_to_train_embeddings,
        path_to_val_sequences_csv,
        path_to_val_embeddings,
        path_to_test_sequences_csv,
        path_to_test_embeddings,
        path_to_train_labels_csv,
        path_to_val_labels_csv,
        BATCH_SIZE,
    )

    model = jetRNA_v4_Model(IN_DIM, HIDDEN_DIM, OUT_DIM, LEARNING_RATE)

    trainer = L.Trainer(
        max_epochs=EPOCHS,
        log_every_n_steps=1,
        enable_checkpointing=True,
        accelerator="mps",
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
    )

    print(f"\n--- Starting Training ({EPOCHS} Epochs) ---")
    trainer.fit(model, data_module)
    print("\n--- Training Complete ---")

