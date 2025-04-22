def create_linear_decay_schedule(start_val: float, end_val: float, n_steps: int):
    """
    Creates a scheduling function that linearly decreases a value from
    start_val to end_val over n_steps, then holds at end_val.

    Args:
        start_val: The starting value at step 1.
        end_val: The value to decay to by step n_steps and hold afterwards.
        n_steps: The number of steps over which the linear decay occurs.

    Returns:
        A function that takes the current step (int) and returns the
        scheduled value (float).
    """
    if not 0.0 <= start_val <= 1.0:
        raise ValueError("start_val must be between 0.0 and 1.0")
    if not 0.0 <= end_val <= 1.0:
        raise ValueError("end_val must be between 0.0 and 1.0")
    if n_steps <= 0:
        raise ValueError("n_steps must be positive")

    # Calculate the change per step during the decay phase
    # Avoid division by zero if n_steps is 1
    if n_steps == 1:
        decay_per_step = 0.0
    else:
        decay_per_step = (start_val - end_val) / (n_steps - 1)

    def schedule_function(current_step: int) -> float:
        """
        Calculates the value for the current step based on the schedule.
        """
        if current_step <= 0:
             # Or handle as an error, depending on desired behavior for step 0 or negative
             return start_val
        elif current_step >= n_steps:
            return end_val
        else:
            # Linearly decay from start_val to end_val
            # current_value = start_val - (change_per_step * steps_elapsed)
            steps_elapsed = current_step - 1
            current_val = start_val - (decay_per_step * steps_elapsed)
            # Ensure we don't overshoot end_val due to potential floating point inaccuracies
            # This depends on whether start_val > end_val or vice versa
            if start_val >= end_val:
                return max(end_val, current_val)
            else: # Handles linear increase case too
                return min(end_val, current_val)

    return schedule_function