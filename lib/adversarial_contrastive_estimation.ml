open Core

let adversarial_mixture_noise ~discriminator ~generator ~x ~y_pos ~lambda =
  let discriminator_model =
    Discriminator.update ~model:discriminator ~x ~y_pos ~y_neg:(Generator.sample ~model:generator ~x)
  in
  let y_neg_nce = nce_sample ~x ~y_pos ~generator ~lambda in
  let discriminator_reward = Discriminator.score ~model:discriminator_model ~x ~y_pos ~y_neg:y_neg_nce in
  let generator_reward = -discriminator_reward in
  let generator_model =
    Generator.update ~model:generator ~x ~y_pos ~y_neg:y_neg_nce ~reward:generator_reward
  in
  discriminator_model, generator_model

let nce_sample ~x ~y_pos ~generator ~lambda =**
  let batch_size = List.length x
  let noise = Random.vector (batch_size * negative_samples) ~dim:(Vector.dimension (List.hd x))
  let noise_list = List.split noise batch_size
  let negative_samples = List.map (fun noise -> Generator.sample ~model:generator ~x:noise) noise_list

  let positive_probs = List.map (fun _ -> exp (Discriminator.score ~model:discriminator ~x ~y:y_pos)) x
  let noise_probs = List.map (fun y_neg -> exp (Discriminator.score ~model:discriminator ~x ~y:y_neg)) negative_samples

  let positive_prob_sum = List.fold_left (+) 0.0 positive_probs
  let noise_prob_sum = List.fold_left (+) 0.0 noise_probs
  let positive_weight = positive_prob_sum /. (positive_prob_sum +. noise_prob_sum)
  let noise_weights = List.map (fun prob -> prob /. noise_prob_sum) noise_probs

  let sampled_idx = Random.choice (List.mapi noise_weights List.iter)
  let sampled_negative = List.nth negative_samples sampled_idx

  let y_neg = if Random.bool () then y_pos else sampled_negative

  y_neg