def update_progress_bar(state):
    """Update tqdm bar description for current step."""
    sa = state["sparse_autoencoder"]
    sae_out, feature_acts, mse_loss, l1_loss, ghost_grad_loss, spacon_loss = state["aux"]

    if state["pbar"] is not None:
        if sa.cfg.is_sparse_connection:
            state["pbar"].set_description(
                f"{state['n_training_steps']}| MSE Loss {float(mse_loss):.3f} | "
                f"L1 {float(l1_loss):.3f} | SCST {float(spacon_loss):.3f}"
            )
        else:
            state["pbar"].set_description(
                f"{state['n_training_steps']}| MSE Loss {float(mse_loss):.3f} | "
                f"L1 {float(l1_loss):.3f}"
            )
        state["pbar"].update(state["batch_size"])
    return state