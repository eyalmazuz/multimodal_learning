

def get_task_config(task):
    if task == 'cancer':
        eval_arguments = [
            '--no_features_scaling',
            '--dataset_type', 'classification',
            '--extra_metrics', 'prc-auc',
            '--split_type', 'scaffold_balanced',
            '--ignore_columns', 'drugBank_id',
        ]

        train_arguments = [
            '--no_features_scaling',
            '--dataset_type', 'classification',
            '--extra_metrics', 'prc-auc',
            '--split_type', 'scaffold_balanced',
            '--ignore_columns', 'drugBank_id',
        ]

    elif task == 'target':
        eval_arguments = [
            '--dataset_type', 'regression',
            '--split_type', 'random',
        ]

        train_arguments = [
            '--dataset_type', 'regression',
            '--split_type', 'random',
        ]

    elif task == 'yeast':
        eval_arguments = [
            '--no_features_scaling',
            '--dataset_type', 'classification',
            '--extra_metrics', 'prc-auc',
            '--split_type', 'scaffold_balanced',
        ]

        train_arguments = [
            '--no_features_scaling',
            '--dataset_type', 'classification',
            '--extra_metrics', 'prc-auc',
            '--split_type', 'scaffold_balanced',
        ]



    return eval_arguments, train_arguments