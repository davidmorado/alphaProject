

def in_train_phase(x, alt, training):
  """Selects `x` in train phase, and `alt` otherwise.
  Note that `alt` should have the *same shape* as `x`.
  Arguments:
      x: What to return in train phase
          (tensor or callable that returns a tensor).
      alt: What to return otherwise
          (tensor or callable that returns a tensor).
      training: Optional scalar tensor
          (or Python boolean, or Python integer)
          specifying the learning phase.
  Returns:
      Either `x` or `alt` based on the `training` flag.
      the `training` flag defaults to `K.learning_phase()`.
  """
    #   if training is None:
    #     training = learning_phase()

  if training == 1 or training is True:
    if callable(x):
      return x()
    else:
      return x

  elif training == 0 or training is False:
    if callable(alt):
      return alt()
    else:
      return alt