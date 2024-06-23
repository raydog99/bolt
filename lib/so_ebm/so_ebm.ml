open Torch

let gaussian_pdf mean sig y_sched =
  let pi = 4.0 *. atan 1.0 in
  let var = sig ** 2.0 in
  let exponent = -((y_sched - mean) ** 2.0) / (2.0 *. var) in
  (exp exponent) / (sqrt (2.0 *. pi *. var))

let gaussian_cdf mean sig y_sched =
  let half = 0.5 in
  let arg = (y_sched - mean) / (sig *. sqrt 2.0) in
  half *. (1.0 + (Owl_stats.Rf.erf arg))

let task_loss_expectation y_sched mean sig params =
  let gamma_under = params.["gamma_under"] in
  let gamma_over = params.["gamma_over"] in
  let pdf = gaussian_pdf mean sig y_sched in
  let cdf = gaussian_cdf mean sig y_sched in
  let loss = ((gamma_under +. gamma_over) *. ((sig ** 2.0 *. pdf) +. ((y_sched -. mean) *. cdf))) -. 
             (gamma_under *. (y_sched -. mean)) +. 
             (0.5 *. (((y_sched -. mean) ** 2.0) +. (sig ** 2.0))) in
  loss

let task_loss y_sched y_actual params =
  let gamma_under = params.["gamma_under"] in
  let gamma_over = params.["gamma_over"] in
  let pos_diff = max (y_actual - y_sched) 0.0 in
  let neg_diff = max (y_sched - y_actual) 0.0 in
  let squared_diff = 0.5 *. ((y_sched - y_actual) ** 2.0) in
  (gamma_under *. pos_diff) +. (gamma_over *. neg_diff) +. squared_diff

let langevin_dynamics model z variables params ~steps ~step_size ~num_samples =
  Torch.eval model;
  List.iter (fun p -> Torch.set_requires_grad p false) (Torch.parameters model);

  let noise = Torch.randn_like z in
  let mean, sig = model variables.["X_train_"] in
  let z = Torch.repeat z [|num_samples; 1|] in
  let z = Torch.set_requires_grad z true in
  let mean = Torch.repeat mean [|num_samples; 1|] in
  let sig = Torch.repeat sig [|num_samples; 1|] in

  for _ = 1 to steps do
    Torch.normal_ noise ~mean:0.0 ~std:0.01;
    Torch.add_ z z noise;

    let out_z = Torch.task_loss_expectation z mean sig params |> Torch.sum1 |> Torch.mean0 in
    Torch.backward out_z;

    Torch.add_ z (Torch.mul (Torch.get_grad z) (-.step_size));
    Torch.detach (Torch.get_grad z);
    Torch.zero_ (Torch.get_grad z);
  done;

  List.iter (fun p -> Torch.set_requires_grad p true) (Torch.parameters model);
  Torch.train model;
  z