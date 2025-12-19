import torch
import numpy as np
import lightning as L

from Model import jetRNA_v4_Model 
from DataModule import jetRNA_v4_DataModule

BATCH_SIZE = 8 # Batches 2 variable-sized graphs together

path_to_train_sequences_csv = "./dataset/sample/train_seq_sample.csv"
path_to_val_sequences_csv = "./dataset/sample/val_seq_sample.csv"
path_to_test_sequences_csv = "./dataset/sample/test_seq_sample.csv"
path_to_train_labels_csv = "./dataset/sample/train_lab_sample.csv"
path_to_val_labels_csv = "./dataset/sample/val_lab_sample.csv"
path_to_test_labels_csv = "./dataset/sample/test_lab_sample.csv"

def get_prediction_tensor(path_to_model: str, path_to_save: str):
    """
    Only performs inference on test data and saves predictions as tensor.
    """
    data_module = jetRNA_v4_DataModule(
        path_to_train_sequences_csv,
        path_to_val_sequences_csv,
        path_to_test_sequences_csv,
        path_to_train_labels_csv,
        path_to_val_labels_csv,
        path_to_test_labels_csv,
        BATCH_SIZE,
    )


    model = jetRNA_v4_Model.load_from_checkpoint(path_to_model)
    model.eval()
    model.freeze()

    # Using predict_step
    trainer = L.Trainer()
    predictions = trainer.predict(model, data_module) # Returns list containing one Tensor of torch.Size([13, 1, 640, 640])

    all_preds = torch.cat([r[0] for r in predictions], dim=0)
    all_centers = torch.cat([r[2] for r in predictions], dim=0)

    # Denormalize and recenter
    final_coords = (all_preds * 900.0) + all_centers.unsqueeze(1)

    torch.save(final_coords, path_to_save)

    return predictions

if __name__ == "__main__":
    version = "v3"
    idx = 1 
    model_path = f"./lightning_logs/{version}/checkpoints/{version}.ckpt"

    path_to_save_prediction_tensor = f"./predictions/prediction_tensor/{version}/{version}_prediction_tensor.pt"
    path_to_save_single_prediction_tensor = f"./predictions/prediction_tensor/{version}/{version}_{idx}_prediction_tensor.csv"
    predictions = get_prediction_tensor(model_path, path_to_save_prediction_tensor)

    predictions = torch.load(path_to_save_prediction_tensor)
    #print(predictions[0][0][0])
    print(predictions.numpy().shape)
    x = predictions[idx, :, :].numpy()
    print(x.shape)
    np.savetxt(path_to_save_single_prediction_tensor, x, fmt="%.2f")
