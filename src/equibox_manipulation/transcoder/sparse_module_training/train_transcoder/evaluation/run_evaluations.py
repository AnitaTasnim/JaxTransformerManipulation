from equibox_manipulation.transcoder.sparse_module_training.train_transcoder.evaluation.ModuleEvaluator import ModuleEvaluator


def run_evaluations(state, model, activation_store, wandb_log_frequency):
        """Run evaluation less frequently to save compute."""
        if (state["n_training_steps"] + 1) % (wandb_log_frequency * 10) == 0:
            evaluator = ModuleEvaluator(state["sparse_autoencoder"], model, activation_store)
            evaluator.run_evals(state["n_training_steps"], state["key"], training=False)
        return state