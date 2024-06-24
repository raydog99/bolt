open Torch

module EBMTrainer = struct
  type t = {
    step_size: float;
    num_steps: int;
    data_augmentation: Tensor.t -> Tensor.t;
    stop_gradient: Tensor.t -> Tensor.t;
    ebm: Tensor.t -> Tensor.t;
    replay_buffer: Tensor.t list ref;
    optimizer: Optimizer.t;
  }

  let create ~step_size ~num_steps ~data_augmentation ~stop_gradient ~ebm ~init_params =
    {
      step_size;
      num_steps;
      data_augmentation;
      stop_gradient;
      ebm;
      replay_buffer = ref [];
      optimizer = Optimizer.adam init_params ~lr:0.001;
    }

  let sample_from_buffer buffer =
    if Random.float 1.0 < 0.99 then
      List.nth buffer (Random.int (List.length buffer))
    else
      Tensor.randn [1; 784] (* Assuming 28x28 images *)

  let nearest_neighbor_entropy x_i buffer =
    let distances = List.map (fun x -> Tensor.mse_loss x_i x) buffer in
    Tensor.of_float (-.log (List.fold_left min Float.max_float distances))

  let train_step trainer x_pos =
    let x_neg = sample_from_buffer !(trainer.replay_buffer) in
    let x_neg_augmented = trainer.data_augmentation x_neg in

    let x_k = ref x_neg_augmented in
    for _ = 1 to trainer.num_steps do
      let x_prev = !x_k in
      let grad = Tensor.grad_of_fn trainer.ebm x_prev in
      let noise = Tensor.randn_like x_prev in
      x_k := Tensor.(sub x_prev (mul_scalar grad trainer.step_size) + (mul_scalar noise (sqrt (Float.mul 2. trainer.step_size))));
    done;

    let x_neg_final = trainer.stop_gradient !x_k in
    let x_neg_detached = Tensor.detach !x_k in

    let l_cd = Tensor.(
      mean (sub (trainer.ebm x_pos) (trainer.ebm x_neg_final))
    ) in

    let l_kl = Tensor.(
      sub
        (trainer.ebm x_neg_detached)
        (nearest_neighbor_entropy x_neg_detached !(trainer.replay_buffer))
    ) in

    let loss = Tensor.(add l_cd l_kl) in

    Optimizer.zero_grad trainer.optimizer;
    Tensor.backward loss;
    Optimizer.step trainer.optimizer;

    trainer.replay_buffer := x_neg_detached :: !(trainer.replay_buffer);

    loss

  let train trainer data_dist max_iterations =
    let rec loop i =
      if i >= max_iterations then ()
      else begin
        let x_pos = data_dist () in
        let loss = train_step trainer x_pos in
        if i mod 100 = 0 then
          Printf.printf "Iteration %d, Loss: %f\n" i (Tensor.float_value loss);
        loop (i + 1)
      end
    in
    loop 0
end

let data_augmentation x =
  x

let stop_gradient x =
  Tensor.detach x

let ebm x =
  let module N = Neural.Packed in
  let net = N.(sequential [
    linear 784 256 ~bias:true;
    relu;
    linear 256 128 ~bias:true;
    relu;
    linear 128 1 ~bias:true;
  ]) in
  N.forward net x