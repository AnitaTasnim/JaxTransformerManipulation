import optax
from typing import Optional, Callable

class LearningRateScheduler:

    def get_scheduler(
        scheduler_name: Optional[str],
        base_lr: float,
        **kwargs,
    ) -> Callable[[int], float]:
        """
        Returns a learning rate schedule function for use with optax.
        """
        if scheduler_name is None or scheduler_name.lower() == "constant":
            return lambda step: base_lr
        
        #elif scheduler_name.lower() == "constantwithwarmup":
        #    warm_up_steps = kwargs.get("warm_up_steps", 0)
        #    return optax.linear_schedule(
        #        init_value=0.0,
        #        end_value=base_lr,
        #        transition_steps=warm_up_steps,
        #    )
        
        elif scheduler_name.lower() == "constantwithwarmup":
            warm_up_steps = kwargs.get("warm_up_steps", 0)

            warmup = optax.linear_schedule(
                init_value=0.0,
                end_value=base_lr,
                transition_steps=warm_up_steps,
            )

            constant = optax.constant_schedule(base_lr)

            return optax.join_schedules(
                schedules=[warmup, constant],
                boundaries=[warm_up_steps],
            )

        elif scheduler_name.lower() == "linearwarmupdecay":
            raise NotImplementedError("linearwarmupdecay not implemented yet")
        elif scheduler_name.lower() == "cosineannealing":
            raise NotImplementedError("cosineannealing not implemented yet")
        elif scheduler_name.lower() == "cosinewithwarmup":
            raise NotImplementedError("cosinewithwarmup not implemented yet")
        elif scheduler_name.lower() == "cosineannealingwarmrestarts":
            raise NotImplementedError("cosineannealingwarmrestarts not implemented yet")
        else:
            raise ValueError(f"Unknown scheduler name: {scheduler_name}")

    
    def update_learning_rate(state):
        """Handle LR reset ramp after neuron resampling."""
        if state["steps_before_reset"] > 0 and state["n_training_steps"] > 0:
            state["steps_before_reset"] -= 1
            state["lr_multiplier"] += state.get("increment", 0.0) / state["schedule"](state["n_training_steps"])
            if state["steps_before_reset"] == 0:
                state["lr_multiplier"] = 1.0
        else:
            state["lr_multiplier"] = 1.0
        return state