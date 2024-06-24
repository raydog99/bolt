open Torch

module Encoder = struct
  let create () =
    Layer.sequential []

  let forward encoder x =
    Layer.forward encoder x
end

module HDGE = struct
  type t = {
    f: Layer.t;
    mutable queue: Tensor.t list;
    b: int;
    k: int;
    c: int;
    m: float;
    t: float;
    optimizer: Optimizer.t;
  }

  let create encoder batch_size queue_size num_classes momentum temperature =
    {
      f = encoder;
      queue = [];
      b = batch_size;
      k = queue_size;
      c = num_classes;
      m = momentum;
      t = temperature;
      optimizer = Optimizer.adam (Layer.parameters encoder) ~lr:0.01;
    }

  let augment x =
    x

  let transform_to_onehot y c =
    Tensor.one_hot y ~num_classes:c

  let normalize logits =
    Tensor.normalize logits ~p:2 ~dim:[1]

  let enqueue t probs =
    t.queue <- Tensor.detach probs :: t.queue

  let dequeue t =
    match t.queue with
    | _ :: rest when List.length t.queue = t.k -> t.queue <- rest
    | _ -> ()

  let train t loader =
    Dataset_helper.iter loader ~f:(fun (x, y) ->
      let x = augment x in
      let y = transform_to_onehot y t.c in

      let ce_logits = Encoder.forward t.f x in
      let ce_loss = Tensor.cross_entropy_loss ce_logits y in

      let probs = normalize ce_logits in
      let l_pos = Tensor.logsumexp (Tensor.mul probs y) ~dim:[1] ~keepdim:true in

      let queue_tensor = Tensor.stack (List.map Tensor.clone t.queue) in
      let l_neg = Tensor.einsum "nc,ck->nck" [|y; Tensor.detach queue_tensor|] in
      let l_neg = Tensor.logsumexp l_neg ~dim:[1] in

      let logits = Tensor.cat [l_pos; l_neg] ~dim:1 in
      let labels = Tensor.zeros [t.k] in

      let cl_loss = Tensor.cross_entropy_loss (Tensor.div logits t.t) labels in

      let loss = Tensor.add ce_loss cl_loss in

      Optimizer.zero_grad t.optimizer;
      Tensor.backward loss;
      Optimizer.step t.optimizer;

      enqueue t probs;
      dequeue t
    )
end