""" import mlflow
import os
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', type=str, default='unet')
    parser.add_argument('--encoder', type=str, default='efficientnet-b0')
    parser.add_argument('--loss', type=str, default="dice")
    parser.add_argument('--batch', type=int, default=1)
    opt = parser.parse_args()
    return opt



def select_best():
    config = parse_arguments()

    client = mlflow.tracking.MlflowClient()
    model_name = 'model_' + config.network + '_' + config.encoder + '_' + config.loss + '_bsz' + str(config.batch)
    
    # Get a list of all registered models
    filter_string = f"name='{model_name}'"
    registered_models = client.search_model_versions(filter_string)

    best_iou = 0
    best_model_version = None
    
    # Iterate over each model version
    for model in registered_models:
        # Get metrics for each model version
        metrics = client.get_run(model.run_id).data.metrics
        iou_score = metrics.get('val_iou', None)

        # Check if this model has the best IOU so far
        if iou_score and iou_score > best_iou:
            best_iou = iou_score
            best_model_version = model.version
    

    
    
    # Transition the best model to production
    if best_model_version:
        client.transition_model_version_stage(
            name=model_name,
            version=best_model_version,
            stage="Production",
            archive_existing_versions=True  # Archive other versions
        )
        print(f"Model version {best_model_version} transitioned to Production with IOU: {best_iou}")
    else:
        print("No suitable model version found.")

if __name__ == "__main__":
    # Specify the registered model name
    select_best() """