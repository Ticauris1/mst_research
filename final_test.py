import torch # type: ignore
import os
from training.kfold import kfold_cross_validation
from sklearn.preprocessing import LabelEncoder # type: ignore
from torchvision import transforms # type: ignore
from collections import Counter
from training.eval import evaluate_model
from training.utils import setup_directories, get_model_with_attention
from dataset.custom_dataset import CustomDataset
from config import DEVICE

def final_test(
    test_data,
    label_encoder,
    save_root,
    model_name="resnet152d",
    attention_type="cbam",
    batch_size=32,
    triplet_embedding_dict=None
):
    """
    Loads the best model from training (fold 1) and evaluates it on the final test set.
    """
    print("\n\n🚀 Starting Final Test on Hold-Out Data")

    X_test, y_test_encoded, z_test = test_data
    num_classes = len(label_encoder.classes_)

    # Find the path to the best model saved during the first fold of training
    _, best_weights_path, graph_dir, predictions_dir = setup_directories(
        base_path=save_root, model_name=model_name, fold=1, attention_type=attention_type
    )

    if not os.path.exists(best_weights_path):
        print(f"❌ Best model weights not found at: {best_weights_path}. Cannot run final test.")
        return

    print(f"✅ Loading best model from: {best_weights_path}")

    # Initialize the model architecture
    model = get_model_with_attention(
        model_name=model_name, num_classes=num_classes, attention_type=attention_type, use_triplet_embedding=True, triplet_embedding_dim=512, fusion_mode="mlp"
    ).to(DEVICE)

    # Load the trained weights
    model.load_state_dict(torch.load(best_weights_path, map_location=DEVICE))

    # Create the test dataset
    test_dataset = CustomDataset(
        image_paths=X_test,
        labels=y_test_encoded,
        metadata=z_test,
        transform=transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        include_skin_vec=True,
        triplet_embedding_dict=triplet_embedding_dict,
        num_classes=num_classes
    )

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Evaluate the model on the final test data
    evaluate_model(
        model=model,
        test_loader=test_loader,
        device=DEVICE,
        label_encoder=label_encoder,
        save_dir=os.path.join(predictions_dir, "final_test_results"),
        model_name=f"{model_name}_{attention_type}_FINAL_TEST",
        graph_dir=os.path.join(graph_dir, "final_test_graphs"),
        fold_classes=list(range(num_classes))
    )
